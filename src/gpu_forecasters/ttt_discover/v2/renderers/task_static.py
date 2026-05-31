"""The default task-prompt renderer: a fixed base prompt + a rules block
parameterised on the problem's ``gpu_name`` / ``triton_version``.

The base prompt is passed through the context's ``problem`` — the renderer
does not vendor its own copy. Experiments that want a different task
prompt (e.g. one that dynamically focuses the model on the slowest case
from the archive's current best) replace this renderer with their own
``TaskPromptRenderer`` implementation.
"""

from __future__ import annotations

from gpu_forecasters.ttt_discover.v2.domain.context import TaskPromptContext
from gpu_forecasters.ttt_discover.v2.interfaces.renderer import TaskPromptRenderer
from gpu_forecasters.typing_utils import implements

_RULES_TEMPLATE = """\
Rules:
- The tensors arguments passed in will be already on your cuda device.
- Define all of your code in one final ```python ``` block.
- We will test the correctness of your kernel on multiple input shapes, make sure to support different potential test cases.
- You are allowed to use mixed precision computations, but make sure your final output is in float32.
- You must use triton {triton_version} and these kernels will be run on an Nvidia {gpu_name}.
- You do not have to implement everything in triton, you may choose to have some of the operations done in pytorch. However, you must implement at least part of the operations in a kernel.
- Include a short docstring at the top summarizing your algorithm.
"""


class StaticTaskPromptRenderer:
    def render(self, ctx: TaskPromptContext) -> str:
        rules = _RULES_TEMPLATE.format(
            gpu_name=ctx.problem.gpu_name,
            triton_version=ctx.problem.triton_version,
        )
        return ctx.problem.base_prompt_text.rstrip() + "\n\n" + rules


_ = implements(TaskPromptRenderer)(StaticTaskPromptRenderer)
