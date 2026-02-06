from __future__ import annotations

import os
import time
import traceback
from typing import Optional, Union

import torch
from loguru import logger
from kernelbench import timing
from kernelbench.eval import (
    KernelExecResult,
    _process_input_tensor,
    get_error_name,
    get_tolerance_for_precision,
    graceful_eval_cleanup,
    load_custom_model,
    load_custom_model_with_tempfile,
    load_original_model_and_inputs,
    register_and_format_exception,
    set_seed,
)


def eval_kernel_against_ref_logged(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    measure_performance: bool = False,
    timing_method: str = "cuda_event",  # see timing.py
    verbose: bool = False,
    build_dir: os.PathLike = None,
    device: Union[torch.device, int] = (
        torch.cuda.current_device() if torch.cuda.is_available() else None
    ),  # have to run on GPU
    backend: str = "cuda",  # can be 'cuda', 'triton', 'tilelang', or 'cute'
    precision: torch.dtype = torch.float32,
    # Guard against potential reward hacking [optional but ongoing enhancement]
    check_for_excessive_speedup: bool = True,
    excessive_speedup_threshold: float = 10,  # flag if the kernel is more than <threshold>x faster than the reference
) -> KernelExecResult:
    """
    Logged clone of kernelbench.eval.eval_kernel_against_ref.

    Adds phase-level timing and per-trial debug logs while preserving upstream behavior.
    """
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"

    if backend.lower() == "tilelang":
        assert (
            precision == torch.float16 or precision == torch.bfloat16
        ), "TileLang only supports fp16 or bfloat16"

    torch.set_printoptions(
        precision=4,
        threshold=10,
        edgeitems=3,
        linewidth=80,
    )

    logger.info(
        "KernelBench eval start (backend={backend}, precision={precision}, "
        "correct_trials={num_correct_trials}, perf_trials={num_perf_trials}, "
        "measure_performance={measure_performance}, timing_method={timing_method}).",
        backend=backend,
        precision=str(precision),
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        measure_performance=measure_performance,
        timing_method=timing_method,
    )
    logger.info(
        "KernelBench eval env: TORCH_EXTENSIONS_DIR={torch_ext_dir}, build_dir={build_dir}.",
        torch_ext_dir=os.environ.get("TORCH_EXTENSIONS_DIR", "unset"),
        build_dir=str(build_dir) if build_dir is not None else "none",
    )

    # set CUDA device
    torch.cuda.set_device(device)

    uses_tempfile = backend.lower() in ["triton", "tilelang", "cute"]
    metadata: dict = {}
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)

    if uses_tempfile:
        if isinstance(device, int):
            device_num = device
        elif isinstance(device, torch.device):
            assert (
                device.type == "cuda"
            ), "CUDA is not availible on device, cannot run Eval"
            device_num = device.index
        else:
            raise ValueError(
                f"device must be an int or torch.device, got {type(device)}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    context = {}

    if verbose:
        print(f"[Eval] Start Evalulation! on device: {device}")
        print("[Eval] Loading Original Model")

    load_start_s = time.perf_counter()
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        original_model_src, context
    )
    set_seed(seed_num)
    init_inputs = get_init_inputs()
    init_inputs = [
        _process_input_tensor(x, device, backend, precision) for x in init_inputs
    ]

    with torch.no_grad():
        set_seed(seed_num)
        original_model = Model(*init_inputs)
        assert hasattr(original_model, "forward")
        if verbose:
            print("[Eval] Original Model Loaded")
    logger.info(
        "KernelBench eval: original model loaded in {elapsed_s:.2f}s.",
        elapsed_s=time.perf_counter() - load_start_s,
    )

    if verbose:
        print("[Eval] Loading and Compiling New Model with Custom CUDA Kernel")

    compile_start_s = time.perf_counter()
    logger.info(
        "KernelBench eval: compiling/loading custom model (this can take a while)."
    )
    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        tempfile = None
        backend_lower = backend.lower()
        if backend_lower in ["triton", "tilelang", "cute"]:
            ModelNew, tempfile = load_custom_model_with_tempfile(
                custom_model_src, entry_point="ModelNew"
            )
        else:
            ModelNew = load_custom_model(custom_model_src, context, build_dir)
        torch.cuda.synchronize(device=device)
    except Exception as e:
        print(
            f"Failed to compile custom CUDA kernel: Record as compilation failure. \nError: {e}"
        )
        if "lock" in str(e) or "No such file or directory" in str(e):
            print(
                f"[Eval] Lock file error during compilation, Please retry. Error: {e}"
            )
            graceful_eval_cleanup(context, device, tempfile)
            return None
        metadata["compilation_error_name"] = get_error_name(e)
        metadata["compilation_error"] = e
        graceful_eval_cleanup(context, device, tempfile)
        logger.info(
            "KernelBench eval: compilation failed in {elapsed_s:.2f}s.",
            elapsed_s=time.perf_counter() - compile_start_s,
        )
        return KernelExecResult(compiled=False, metadata=metadata)

    logger.info(
        "KernelBench eval: custom model compiled/loaded in {elapsed_s:.2f}s.",
        elapsed_s=time.perf_counter() - compile_start_s,
    )

    if ModelNew is None:
        print(
            "Failed to load custom model: Syntax error or ModelNew not found in generated code. "
            "Record as compilation failure."
        )
        metadata["compilation_error_name"] = "SyntaxError"
        metadata["compilation_error"] = (
            "Syntax error in custom generated code or ModelNew not found"
        )
        graceful_eval_cleanup(context, device, tempfile)
        return KernelExecResult(compiled=False, metadata=metadata)

    instantiate_start_s = time.perf_counter()
    try:
        with torch.no_grad():
            set_seed(seed_num)
            custom_model = ModelNew(*init_inputs)
            assert hasattr(custom_model, "forward")
            original_model = original_model.to(device=device, dtype=precision)
            custom_model = custom_model.to(device=device, dtype=precision)
            torch.cuda.synchronize(device=device)
        if verbose:
            print("[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        print(
            "Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. "
            f"\nError: {e}"
        )
        graceful_eval_cleanup(context, device, tempfile)
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        logger.info(
            "KernelBench eval: custom model instantiation failed in {elapsed_s:.2f}s.",
            elapsed_s=time.perf_counter() - instantiate_start_s,
        )
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)
    logger.info(
        "KernelBench eval: custom model instantiated in {elapsed_s:.2f}s.",
        elapsed_s=time.perf_counter() - instantiate_start_s,
    )

    kernel_exec_result = None

    if verbose:
        print("[Eval] Checking Correctness")
    correctness_start_s = time.perf_counter()
    try:
        kernel_exec_result = run_and_check_correctness(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=seed_num,
            device=device,
            backend=backend,
            precision=precision,
        )
    except Exception as e:
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        kernel_exec_result = KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )
    logger.info(
        "KernelBench eval: correctness check finished in {elapsed_s:.2f}s "
        "(correct={correctness}, trials={trials}).",
        elapsed_s=time.perf_counter() - correctness_start_s,
        correctness=kernel_exec_result.correctness if kernel_exec_result else False,
        trials=metadata.get("correctness_trials", "unknown"),
    )

    if measure_performance:
        try:
            if kernel_exec_result and kernel_exec_result.correctness:
                if verbose:
                    print("[Eval] Measuring Performance as Sample is Correct")

                perf_start_s = time.perf_counter()
                torch.cuda.synchronize(device=device)
                set_seed(seed_num)
                inputs = get_inputs()
                inputs = [
                    _process_input_tensor(x, device, backend, precision) for x in inputs
                ]

                model_new = custom_model.to(device=device, dtype=precision)
                torch.cuda.synchronize(device=device)

                timing_fn = timing.get_timing_function(timing_method)
                elapsed_times = timing_fn(
                    model_new,
                    inputs,
                    num_trials=num_perf_trials,
                    verbose=verbose,
                    device=device,
                )
                runtime_stats = timing.get_timing_stats(elapsed_times, device=device)

                if verbose:
                    print(f"[Eval] Performance Stats: {runtime_stats}")
                kernel_exec_result.runtime = runtime_stats["mean"]
                kernel_exec_result.runtime_stats = runtime_stats
                logger.info(
                    "KernelBench eval: custom perf timing finished in {elapsed_s:.2f}s "
                    "(runtime_us={runtime_us:.2f}).",
                    elapsed_s=time.perf_counter() - perf_start_s,
                    runtime_us=kernel_exec_result.runtime,
                )
        except Exception as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            kernel_exec_result.metadata["error_during_performance"] = e

    if measure_performance and check_for_excessive_speedup:
        if verbose:
            print("[Eval] Additional checks to flag excessive speedup")

        ref_start_s = time.perf_counter()
        torch.cuda.synchronize(device=device)
        set_seed(seed_num)
        inputs = get_inputs()
        inputs = [_process_input_tensor(x, device, backend, precision) for x in inputs]

        model_new = custom_model.to(device=device, dtype=precision)
        torch.cuda.synchronize(device=device)

        timing_fn = timing.get_timing_function(timing_method)
        reference_elapsed_times = timing_fn(
            original_model,
            inputs,
            num_trials=num_perf_trials,
            verbose=verbose,
            device=device,
        )
        reference_runtime_stats = timing.get_timing_stats(
            reference_elapsed_times, device=device
        )
        kernel_exec_result.ref_runtime = reference_runtime_stats["mean"]
        kernel_exec_result.ref_runtime_stats = reference_runtime_stats

        effective_speedup = kernel_exec_result.ref_runtime / kernel_exec_result.runtime

        if verbose:
            print(
                f"[Eval] Effective Speedup is {effective_speedup:.2f}x using timing method {timing_method}"
            )
        if effective_speedup > excessive_speedup_threshold:
            kernel_exec_result.metadata["excessive_speedup"] = True
            print(
                f"[WARNING] Excessive speedup {effective_speedup:.2f}x over "
                f"{excessive_speedup_threshold}x threshold detected"
            )
            print(
                "[WARNING] Double check your kernel carefully to ensure it is not reward hacking."
            )
        logger.info(
            "KernelBench eval: reference perf timing finished in {elapsed_s:.2f}s "
            "(ref_runtime_us={ref_runtime_us:.2f}, effective_speedup={effective_speedup:.2f}).",
            elapsed_s=time.perf_counter() - ref_start_s,
            ref_runtime_us=kernel_exec_result.ref_runtime,
            effective_speedup=effective_speedup,
        )

    cleanup_start_s = time.perf_counter()
    graceful_eval_cleanup(context, device, tempfile)
    logger.info(
        "KernelBench eval: cleanup finished in {elapsed_s:.2f}s.",
        elapsed_s=time.perf_counter() - cleanup_start_s,
    )
    return kernel_exec_result


def run_and_check_correctness(
    original_model_instance: torch.nn.Module,
    new_model_instance: torch.nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose: bool = False,
    seed: int = 42,
    device: Optional[torch.device] = None,
    backend: str = "cuda",
    precision: torch.dtype = torch.float32,
) -> KernelExecResult:
    """
    Logged clone of kernelbench.eval.run_and_check_correctness.
    """
    pass_count = 0

    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]

    with torch.no_grad():
        for trial in range(num_correct_trials):
            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Eval] Generating Random Input with seed {trial_seed}")
            logger.debug(
                "KernelBench eval: correctness trial {trial_idx}/{num_trials} (seed={seed}).",
                trial_idx=trial + 1,
                num_trials=num_correct_trials,
                seed=trial_seed,
            )

            set_seed(trial_seed)
            inputs = get_inputs_fn()
            inputs = [
                _process_input_tensor(x, device, backend, precision) for x in inputs
            ]

            set_seed(trial_seed)
            model = original_model_instance.to(device=device, dtype=precision)

            set_seed(trial_seed)
            model_new = new_model_instance.to(device=device, dtype=precision)

            output = model(*inputs)
            torch.cuda.synchronize(device=device)

            try:
                output_new = model_new(*inputs)
                torch.cuda.synchronize(device=device)
                if output.shape != output_new.shape:
                    metadata = register_and_format_exception(
                        "correctness_issue",
                        f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}",
                        metadata,
                    )
                    metadata["correctness_issue_name"] = "correctness_issue"
                    if verbose:
                        print(
                            f"[FAIL] trial {trial}: Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
                        )
                    logger.debug(
                        "KernelBench eval: trial {trial_idx} failed (shape mismatch).",
                        trial_idx=trial + 1,
                    )
                    return KernelExecResult(
                        compiled=True, correctness=False, metadata=metadata
                    )

                tolerance = get_tolerance_for_precision(precision)
                if not torch.allclose(
                    output, output_new, atol=tolerance, rtol=tolerance
                ):
                    max_diff = torch.max(torch.abs(output - output_new)).item()
                    avg_diff = torch.mean(torch.abs(output - output_new)).item()
                    metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                    metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                    metadata["correctness_issue"] = "Output mismatch"
                    if verbose:
                        print(f"[FAIL] trial {trial}: Output mismatch")
                    logger.debug(
                        "KernelBench eval: trial {trial_idx} failed (output mismatch, "
                        "max_diff={max_diff:.6f}, avg_diff={avg_diff:.6f}).",
                        trial_idx=trial + 1,
                        max_diff=max_diff,
                        avg_diff=avg_diff,
                    )
                else:
                    pass_count += 1
                    if verbose:
                        print(f"[PASS] trial {trial}: New Model matches Model")
                    logger.debug(
                        "KernelBench eval: trial {trial_idx} passed.",
                        trial_idx=trial + 1,
                    )

            except Exception as e:
                print("[Error] Exception happens during correctness check")
                print(f"Error in launching kernel for ModelNew: {e}")
                print("\n[Full Traceback]:")
                traceback.print_exc()
                print("\n")

                metadata = register_and_format_exception(
                    "runtime_error", e, metadata, truncate=True
                )
                metadata["runtime_error_name"] = get_error_name(e)
                metadata["runtime_error_traceback"] = traceback.format_exc()
                logger.debug(
                    "KernelBench eval: trial {trial_idx} failed with exception {error}.",
                    trial_idx=trial + 1,
                    error=repr(e),
                )
                return KernelExecResult(
                    compiled=True, correctness=False, metadata=metadata
                )

    if verbose:
        print(
            f"[Eval] Pass count: {pass_count}, num_correct_trials: {num_correct_trials}"
        )

    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    if pass_count == num_correct_trials:
        return KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    return KernelExecResult(compiled=True, correctness=False, metadata=metadata)
