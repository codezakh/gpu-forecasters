"""Extract the last ``python`` code block from a model response.

Mirrors the regex semantics used by v1's
``last_codeblock_postprocess`` (dataset_builder.py:139) and
``trimul_feedback_mutation._extract_last_python_codeblock`` — picks the
*last* block because the rules prompt the model to put its final code in
one trailing block, and reasoning models often emit drafts above that.
"""

from __future__ import annotations

import re

from gpu_forecasters.ttt_discover.v2.interfaces.extractor import CodeExtractor
from gpu_forecasters.typing_utils import implements

_PYTHON_CODEBLOCK_RE = re.compile(
    r"```python\n(?!```)(.*?)(?:\n```)?(?=\n```|$)",
    re.DOTALL,
)


class LastPythonBlockExtractor:
    def extract(self, raw_response: str) -> str | None:
        if not raw_response:
            return None
        matches = list(_PYTHON_CODEBLOCK_RE.finditer(raw_response))
        if not matches:
            return None
        code = matches[-1].group(1).rstrip()
        return code or None


_ = implements(CodeExtractor)(LastPythonBlockExtractor)
