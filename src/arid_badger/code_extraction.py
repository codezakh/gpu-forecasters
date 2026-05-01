"""Shared code extraction helpers for LLM-emitted programs.

The ``extract_last_python_codeblock`` helper picks the LAST python block
from a model response. The mutation prompts used by the v2 KernelBench
and gpu_mode_kernel providers both instruct the model to put its final
code in one trailing block, but reasoning models often emit drafts above
that block — taking the last match is the rule that makes "final answer"
extraction robust.
"""

from __future__ import annotations

import re

_PYTHON_CODEBLOCK_RE = re.compile(
    r"```python\n(?!```)(.*?)(?:\n```)?(?=\n```|$)",
    re.DOTALL,
)


def extract_last_python_codeblock(text: str) -> str | None:
    matches = list(_PYTHON_CODEBLOCK_RE.finditer(text))
    if not matches:
        return None
    code = matches[-1].group(1).rstrip()
    return code or None
