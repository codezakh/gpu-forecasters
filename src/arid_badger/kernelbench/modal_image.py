"""Shared Modal infrastructure for kernel evaluation.

Defines the Modal app, container image, and GPU architecture mapping used
across all Modal-based kernel evaluators. Separated from scoring logic so
Phase 2 (compile on CPU, benchmark on GPU) can reuse the same image.
"""

from __future__ import annotations

import os

import modal

# ---------------------------------------------------------------------------
# Paths (resolved at import time on the local machine for image construction)
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# src/arid_badger/kernelbench → src/arid_badger → src → 15-arid-badger
_LIBRARY_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
_KERNELBENCH_DIR = os.path.normpath(
    os.path.join(_LIBRARY_ROOT, "third_party", "KernelBench")
)
_KERNELBENCH_SRC_DIR = os.path.join(_KERNELBENCH_DIR, "src")
_KERNELBENCH_DATASET_DIR = os.path.join(_KERNELBENCH_DIR, "KernelBench")

# ---------------------------------------------------------------------------
# GPU architecture mapping
# Maps Modal GPU names to torch CUDA arch names used by set_gpu_arch().
# ---------------------------------------------------------------------------

GPU_ARCH_MAPPING: dict[str, list[str]] = {
    "T4": ["Turing"],
    "A10G": ["Ampere"],
    "A100": ["Ampere"],
    "L4": ["Ada"],
    "L40S": ["Ada"],
    "H100": ["Hopper"],
    "H200": ["Hopper"],
}

# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------

app = modal.App("arid-badger-kernel-eval")

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry(
        # Python 3.13 to match our local environment. KernelBench's
        # requires-python >= 3.10 is satisfied by 3.13.
        "nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.13"
    )
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .uv_sync(uv_project_dir=_KERNELBENCH_DIR, extras=["gpu"])
    # arid_badger's additional deps not covered by KernelBench. We can't
    # uv_sync our own pyproject.toml because it uses uv workspaces, which
    # Modal doesn't support.
    .uv_pip_install("loguru", "pydantic", "pytest")
    .env({"PYTHONPATH": "/root/kernelbench_src"})
    # add_local_* must come last — Modal mounts these at startup rather than
    # baking them into the image, so no build steps can follow.
    .add_local_python_source("arid_badger")
    .add_local_dir(_KERNELBENCH_SRC_DIR, remote_path="/root/kernelbench_src")
    .add_local_dir(_KERNELBENCH_DATASET_DIR, remote_path="/root/KernelBench")
)
