"""Per-kernel ``KernelPack`` declarations.

Each module here declares one ``KernelPack`` constant for one
gpu-mode/reference-kernels problem, plus the Modal app + benchmarker
cls that bind the pack to a remote container. Generic infra in
``arid_badger.gpu_mode_kernel`` consumes the pack — no per-kernel
mirror packages are needed elsewhere in the tree.
"""
