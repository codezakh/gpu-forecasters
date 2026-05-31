"""Causal depthwise 1D convolution kernel — port of
``gpu-mode/reference-kernels/problems/helion/causal_conv1d_py``.

Package layout intentionally mirrors ``gpu_forecasters.trimul`` 1:1:

- ``cases.py``           — ``TypedDict`` + correctness/benchmark literals
- ``comparison.py``      — numerical utilities + ``DeterministicContext``
- ``reference.py``       — ``ref_kernel``, ``generate_input``,
                           ``check_implementation``
- ``seed_kernel.py``     — cold-start program (PyTorch reference)
- ``core.py``            — feedback union + exec result + ``Stats``
- ``scoring.py``         — single-case scoring entry point
- ``modal_scoring.py``   — Modal harness

Each module's docstring identifies which TriMul sibling it duplicates.
The duplication is deliberate: the second port exists to expose the
shared shape so that the ``gpu_forecasters.gpu_mode_kernel`` extraction in
the follow-up (gh070-A task #3) has two concrete instances to abstract
over rather than one.
"""

from __future__ import annotations
