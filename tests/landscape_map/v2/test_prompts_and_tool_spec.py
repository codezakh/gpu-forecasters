"""Smoke tests for prompt rendering and tool-spec schema generation."""

from __future__ import annotations

from gpu_forecasters.landscape_map.v2.domain import (
    HardwareContext,
    KernelImplementation,
    KernelRuntimeQuery,
    KernelTaskInfo,
)
from gpu_forecasters.landscape_map.v2.prompt_rendering import (
    render_system_prompt,
    render_user_prompt,
)
from gpu_forecasters.landscape_map.v2.tool_spec import (
    TOOL_NAME,
    cookbook_tool_spec,
    openai_tool_spec,
    parameters_schema,
)


def _query(with_hardware: bool = True) -> KernelRuntimeQuery:
    return KernelRuntimeQuery(
        task=KernelTaskInfo(op_name="trimul", level_id=2, task_id=99),
        reference=KernelImplementation(
            kernel_name="ref", code="x = a + b", runtime_ms=1.0
        ),
        candidate=KernelImplementation(
            kernel_name="cand", code="x = fused_add(a, b)", runtime_ms=0.6
        ),
        hardware=(
            HardwareContext(
                device_name="NVIDIA A100-SXM4-80GB",
                compute_capability=(8, 0),
                total_global_memory_gb=79.3,
                multiprocessor_count=108,
                max_threads_per_multiprocessor=2048,
                clock_rate_ghz=1.41,
                memory_clock_rate_ghz=1.59,
                memory_bus_width_bits=5120,
            )
            if with_hardware
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def test_system_prompt_mentions_tool_name_and_simplex() -> None:
    sys_prompt = render_system_prompt()
    assert TOOL_NAME in sys_prompt
    # The "must sum to 1" instruction is in the probability-distribution
    # section. Lock this in so the prompt can't drift away from it.
    assert "sum to 1" in sys_prompt


def test_user_prompt_includes_code_and_op_name() -> None:
    query = _query()
    rendered = render_user_prompt(query)
    assert "trimul" in rendered
    assert "fused_add(a, b)" in rendered
    assert TOOL_NAME in rendered


def test_user_prompt_includes_hardware_table_when_present() -> None:
    rendered = render_user_prompt(_query(with_hardware=True))
    assert "NVIDIA A100-SXM4-80GB" in rendered
    assert "Compute Capability" in rendered


def test_user_prompt_omits_hardware_table_when_absent() -> None:
    rendered = render_user_prompt(_query(with_hardware=False))
    assert "Hardware Context" not in rendered


# ---------------------------------------------------------------------------
# Tool spec
# ---------------------------------------------------------------------------


def test_parameters_schema_is_flat_no_refs() -> None:
    schema = parameters_schema()
    assert "$defs" not in schema
    # The top-level required list must mention all eight per-bin floats
    # plus predicted_bin and reasoning.
    required = set(schema["required"])
    assert "predicted_bin" in required
    assert "reasoning" in required
    for fname in (
        "p_severe_slowdown",
        "p_significant_slowdown",
        "p_moderate_slowdown",
        "p_minor_slowdown",
        "p_minor_speedup",
        "p_significant_speedup",
        "p_high_speedup",
        "p_extreme_speedup",
    ):
        assert fname in required


def test_openai_tool_spec_shape() -> None:
    spec = openai_tool_spec()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == TOOL_NAME
    assert spec["function"]["parameters"] == parameters_schema()


def test_cookbook_tool_spec_shape() -> None:
    spec = cookbook_tool_spec()
    assert spec["name"] == TOOL_NAME
    assert spec["parameters"] == parameters_schema()
