"""Pack-name → (PackedModalRuntime, CaseSpeedupT) dispatch for runbook scripts.

The runbook scripts take ``pack`` as a Literal string in their config.
This module is the single place that resolves that string to the
concrete pack runtime and per-case-speedup type the library's PUCT
search drivers want.

Imports are lazy at function-call time. Each pack module pulls in
``modal`` and the pack's PyTorch reference at import time, which is
cheap but unnecessary for the other five packs in a given search.
"""

from __future__ import annotations

from typing import Any


def get_pack_runtime_and_case_type(pack: str) -> tuple[Any, type[Any]]:
    """Return ``(pack_runtime, case_speedup_type)`` for ``pack``.

    ``pack`` is one of:
    ``trimul``, ``cross_entropy``, ``gdn_chunk_fwd_h``,
    ``gdn_chunk_fwd_o``, ``gdn_recompute_w_u``, ``fp8_quant``.
    """
    if pack == "trimul":
        from gpu_forecasters.gpu_mode_kernel.packs.trimul import (
            TRIMUL_RUNTIME,
            TriMulCaseSpeedup,
        )

        return TRIMUL_RUNTIME, TriMulCaseSpeedup
    if pack == "cross_entropy":
        from gpu_forecasters.gpu_mode_kernel.packs.cross_entropy import (
            CROSS_ENTROPY_RUNTIME,
            CrossEntropyCaseSpeedup,
        )

        return CROSS_ENTROPY_RUNTIME, CrossEntropyCaseSpeedup
    if pack == "gdn_chunk_fwd_h":
        from gpu_forecasters.gpu_mode_kernel.packs.gated_deltanet_chunk_fwd_h import (
            GDN_CHUNK_FWD_H_RUNTIME,
            GdnChunkFwdHCaseSpeedup,
        )

        return GDN_CHUNK_FWD_H_RUNTIME, GdnChunkFwdHCaseSpeedup
    if pack == "gdn_chunk_fwd_o":
        from gpu_forecasters.gpu_mode_kernel.packs.gated_deltanet_chunk_fwd_o import (
            GDN_CHUNK_FWD_O_RUNTIME,
            GdnChunkFwdOCaseSpeedup,
        )

        return GDN_CHUNK_FWD_O_RUNTIME, GdnChunkFwdOCaseSpeedup
    if pack == "gdn_recompute_w_u":
        from gpu_forecasters.gpu_mode_kernel.packs.gated_deltanet_recompute_w_u import (
            GDN_RECOMPUTE_W_U_RUNTIME,
            GdnRecomputeWUCaseSpeedup,
        )

        return GDN_RECOMPUTE_W_U_RUNTIME, GdnRecomputeWUCaseSpeedup
    if pack == "fp8_quant":
        from gpu_forecasters.gpu_mode_kernel.packs.fp8_quant import (
            FP8_QUANT_RUNTIME,
            Fp8QuantCaseSpeedup,
        )

        return FP8_QUANT_RUNTIME, Fp8QuantCaseSpeedup
    raise ValueError(f"unknown pack name: {pack!r}")


__all__ = ["get_pack_runtime_and_case_type"]
