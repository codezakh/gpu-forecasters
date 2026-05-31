"""``BinFiller``: drives one goal-directed PUCT search per ``fill`` call.

Construction-time inputs (``__init__``) are stable across one pipeline
run: pack runtime, reference text, hardware metadata claim, search
dynamics, and provider knobs. The Modal eval session is opened in
``__enter__`` and shared across all bins — Modal container warmup is
paid once.

Per-call inputs (``BinFillRequest``) carry only what varies between
bins: target bin, harvested seed pool, output destination.

The mutation provider is per-request (its prompt is goal-conditioned on
``target_bin``), so its asyncio loop opens and closes per ``fill``
call. The goal-conditioned eval wrapper is also per-request — it's a
stateless adapter over the shared inner Modal provider.

Pack-generic over ``(TestArgsT, CaseSpeedupT)``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generic, Self

from loguru import logger

from gpu_forecasters.gpu_mode_kernel.core import (
    CaseSpeedupT,
    GpuModeKernelObservation,
    SuccessFeedback,
)
from gpu_forecasters.gpu_mode_kernel.kernel_pack import TestArgsT
from gpu_forecasters.gpu_mode_kernel.modal_scoring import PackedModalRuntime
from gpu_forecasters.gpu_mode_kernel.providers import GpuModeKernelModalProvider
from gpu_forecasters.hill_climbing.domain import Evaluation
from gpu_forecasters.invocation_sink import FilesystemInvocationSink
from gpu_forecasters.landscape_map.v1.domain import HardwareContext
from gpu_forecasters.max_reward_puct.v2.config import SearchConfig
from gpu_forecasters.max_reward_puct.v2.event_log import FileEventLog
from gpu_forecasters.max_reward_puct.v2.search import SearchDriver

from .domain import (
    BinFillRequest,
    BinFillResult,
    EvaluationProviderSpec,
    MutationProviderSpec,
    speedup_band_for_bin,
)
from .goal_conditioned_evaluation import GoalConditionedEvaluationProvider
from .goal_conditioned_mutation.provider import (
    GoalConditionedMutationProvider,
    render_prompt,
)
from .seed_selection import SelectedSeed, select_seed
from .summary import (
    compute_run_summary_from_event_log,
    extract_in_target_kernels_from_event_log,
)


class BinFiller(Generic[TestArgsT, CaseSpeedupT]):
    """Pack-generic goal-directed bin filler.

    Open one filler per pipeline run, then issue many ``fill`` calls
    against the same Modal session. Each ``fill`` runs a per-bin
    goal-conditioned PUCT search, persists durable artifacts under the
    request's ``output_dir``, and returns the produced fills as
    ``KernelRuntimeComparison`` rows the eval-set JSONL can consume
    directly.
    """

    def __init__(
        self,
        *,
        pack_runtime: PackedModalRuntime[TestArgsT, CaseSpeedupT],
        reference_code: str,
        hardware: HardwareContext,
        search: SearchConfig,
        mutation: MutationProviderSpec,
        evaluation: EvaluationProviderSpec,
        invocation_sink_dir: Path | None = None,
    ) -> None:
        """Construct a filler. Providers are opened in ``__enter__``;
        the Modal session lives for the duration of the context.

        ``invocation_sink_dir`` is filler-wide (one ``modal_invocations/``
        directory shared across all bin runs). Pass ``None`` to skip
        invocation tracking. We do not split by bin here because the
        ``InvocationSink`` is owned by the long-lived Modal provider;
        rotating it per ``fill`` would require re-opening the session.
        Records carry ``code_sha256`` so a downstream consumer can
        attribute them to bins by joining against ``events.jsonl``.
        """
        self._pack_runtime = pack_runtime
        self._reference_code = reference_code
        self._hardware = hardware
        self._search_config = search
        self._mutation_spec = mutation
        self._evaluation_spec = evaluation
        self._invocation_sink_dir = invocation_sink_dir

        self._inner_evaluation_provider: (
            GpuModeKernelModalProvider[TestArgsT, CaseSpeedupT] | None
        ) = None

    # --- Lifecycle ------------------------------------------------------

    def __enter__(self) -> Self:
        sink = (
            FilesystemInvocationSink(self._invocation_sink_dir)
            if self._invocation_sink_dir is not None
            else None
        )
        provider: GpuModeKernelModalProvider[TestArgsT, CaseSpeedupT] = (
            GpuModeKernelModalProvider(
                pack_runtime=self._pack_runtime,
                aggregator=self._evaluation_spec.aggregator,
                gpu=self._evaluation_spec.eval_gpu,
                max_in_flight=self._evaluation_spec.max_in_flight,
                max_containers=self._evaluation_spec.max_containers,
                get_timeout_s=self._evaluation_spec.get_timeout_s,
                invocation_sink=sink,
            )
        )
        _ = provider.__enter__()
        self._inner_evaluation_provider = provider
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._inner_evaluation_provider is not None:
            self._inner_evaluation_provider.__exit__(exc_type, exc_val, exc_tb)
            self._inner_evaluation_provider = None

    # --- Per-request fill ----------------------------------------------

    def fill(self, request: BinFillRequest) -> BinFillResult:
        if self._inner_evaluation_provider is None:
            raise RuntimeError(
                f"{type(self).__name__} must be entered as a context manager before fill()."
            )
        request.output_dir.mkdir(parents=True, exist_ok=True)

        band = speedup_band_for_bin(request.target_bin)
        logger.info(
            "BinFiller.fill: target_bin={bin} band={band} midpoint={mid:.4f}× output_dir={dir}",
            bin=request.target_bin.name,
            band=band.display,
            mid=band.midpoint,
            dir=request.output_dir,
        )

        # 1. Pick seed.
        seed = select_seed(
            request.harvested,
            target_bin=request.target_bin,
            pack=self._pack_runtime.pack,
        )
        if seed.source is not None:
            logger.info(
                "Picked seed from harvest: source_id={sid} aggregated_speedup={s:.4f}× true_bin={b}",
                sid=seed.source.source_id,
                s=seed.source.aggregated_speedup,
                b=seed.source.true_bin.name,
            )
            _ = (request.output_dir / "seed.json").write_text(
                seed.source.model_dump_json(indent=2)
            )
        else:
            logger.info("Cold-start: harvest empty, using pack.seed_kernel_code")

        # 2. Pre-flight legibility prompt.
        sample_prompt = render_prompt(
            pack=self._pack_runtime.pack,
            target_bin=request.target_bin,
            parent_code=seed.program_code,
            evaluation=_synthesize_seed_evaluation(seed, self._pack_runtime.pack.case_speedup_type),
            gpu_name=self._mutation_spec.gpu_name,
            triton_version=self._mutation_spec.triton_version,
        )
        _ = (request.output_dir / "sample_prompt.txt").write_text(sample_prompt)

        # 3. Per-request mutation provider (target_bin is baked into
        # the prompt) and goal-conditioned eval wrapper around the
        # construction-time inner provider.
        mutation_provider = GoalConditionedMutationProvider[TestArgsT, CaseSpeedupT](
            pack=self._pack_runtime.pack,
            target_bin=request.target_bin,
            model_slug=self._mutation_spec.model_slug,
            gpu_name=self._mutation_spec.gpu_name,
            triton_version=self._mutation_spec.triton_version,
            max_llm_concurrency=self._mutation_spec.max_llm_concurrency,
            num_retries=self._mutation_spec.num_retries,
            request_timeout_s=self._mutation_spec.request_timeout_s,
            temperature=self._mutation_spec.temperature,
            max_tokens=self._mutation_spec.max_tokens,
        )
        wrapper = GoalConditionedEvaluationProvider[CaseSpeedupT](
            inner_evaluation_provider=self._inner_evaluation_provider,
            target_bin=request.target_bin,
        )

        # 4. Event log + driver.
        observation_type = GpuModeKernelObservation[
            self._pack_runtime.pack.case_speedup_type  # pyright: ignore[reportInvalidTypeArguments]
        ]
        event_log: FileEventLog[GpuModeKernelObservation[CaseSpeedupT]] = FileEventLog(
            request.output_dir / "events.jsonl",
            observation_type=observation_type,
        )
        search_driver: SearchDriver[GpuModeKernelObservation[CaseSpeedupT]] = (
            SearchDriver(
                self._search_config,
                mutation_provider=mutation_provider,
                evaluation_provider=wrapper,
                event_log=event_log,
                observation_type=observation_type,
            )
        )

        # 5. Run. The mutation provider's asyncio loop opens for this
        # bin only; the inner Modal session stays open across bins.
        wall_clock_start = time.perf_counter()
        with mutation_provider:
            _final_state = search_driver.run(initial_program=seed.program_code)
        wall_clock_seconds = time.perf_counter() - wall_clock_start
        logger.info(
            "BinFiller.fill complete in {s:.1f}s (target_bin={bin})",
            s=wall_clock_seconds,
            bin=request.target_bin.name,
        )

        # 6. Summary + extract.
        events = event_log.read_all()
        summary = compute_run_summary_from_event_log(
            events,
            target_bin=request.target_bin,
            target_band_lo=band.lo,
            target_band_hi=band.hi,
            target_midpoint_speedup=band.midpoint,
            seed_source_id=seed.source.source_id if seed.source else None,
            seed_speedup_at_harvest=(
                seed.source.aggregated_speedup if seed.source else None
            ),
            model_slug=self._mutation_spec.model_slug,
            search_config=self._search_config.model_dump(),
            k_per_parent=self._search_config.k_per_parent,
            archive_capacity=self._search_config.archive_capacity,
            wall_clock_seconds=wall_clock_seconds,
            observation_type=observation_type,
        )
        _ = (request.output_dir / "summary.json").write_text(
            summary.model_dump_json(indent=2)
        )

        in_target = extract_in_target_kernels_from_event_log(
            events,
            target_bin=request.target_bin,
            reference_code=self._reference_code,
            hardware=self._hardware,
            source_id_prefix=f"binfiller/{request.target_bin.name}",
        )

        return BinFillResult(
            target_bin=request.target_bin,
            in_target_kernels=in_target,
            events_log_path=event_log.path,
            summary=summary,
        )


def _synthesize_seed_evaluation(
    seed: SelectedSeed,
    case_speedup_type: type[CaseSpeedupT],
) -> Evaluation[GpuModeKernelObservation[CaseSpeedupT]]:
    """Build a synthetic Evaluation around the seed's harvest-time speedup
    (or 1.0× if cold-start) for rendering the cold-start sample prompt.
    Not used by the search itself.
    """
    del case_speedup_type  # Type parameter only; SuccessFeedback is generic.
    speedup = seed.source.aggregated_speedup if seed.source is not None else 1.0
    feedback = SuccessFeedback[CaseSpeedupT](
        aggregated_speedup=speedup,
        aggregation_method="geomean",
        per_case_speedups=[],
    )
    observation = GpuModeKernelObservation[CaseSpeedupT](feedback=feedback)
    return Evaluation[GpuModeKernelObservation[CaseSpeedupT]](
        observation=observation,
        reward=speedup,
    )
