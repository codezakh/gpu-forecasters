from unittest.mock import MagicMock

from arid_badger.ttt_discover.v2.domain.context import TaskPromptContext
from arid_badger.ttt_discover.v2.domain.problem import TriMulProblem
from arid_badger.ttt_discover.v2.renderers.task_static import (
    StaticTaskPromptRenderer,
)


def _problem() -> TriMulProblem:
    return TriMulProblem(
        base_prompt_text="BASE PROMPT TEXT",
        test_cases=(),
        gpu_name="A100-80GB",
        triton_version="3.3.1",
        target_runtime_us=2500.0,
    )


def test_static_task_renderer_appends_rules() -> None:
    renderer = StaticTaskPromptRenderer()
    ctx = TaskPromptContext(
        problem=_problem(),
        archive=MagicMock(),
        parent=None,
        timestep=0,
    )
    rendered = renderer.render(ctx)
    assert rendered.startswith("BASE PROMPT TEXT")
    assert "Rules:" in rendered
    assert "A100-80GB" in rendered
    assert "triton 3.3.1" in rendered
