"""
Device selection utility for CMS-FlowSim-GPU.

Supports:
  - NVIDIA GPUs via CUDA
  - AMD GPUs via ROCm  (PyTorch ROCm builds expose AMD GPUs as "cuda" devices,
                        so torch.cuda.is_available() returns True on ROCm too)
  - CPU fallback (automatic — no separate wheel needed)

Use one of two container images, install PyTorch accordingly:
  NVIDIA (CUDA container):    pip install torch>=2.9.0
  AMD   (ROCm 7.2 container): pip install torch>=2.9.0 --index-url https://download.pytorch.org/whl/rocm7.2

CPU is always available as a fallback within either container.
"""

import torch


def get_device(gpu_requested: bool = True) -> torch.device:
    """
    Return the best available compute device.

    Parameters
    ----------
    gpu_requested : bool
        When True (default), use a GPU if one is available.
        When False, always return CPU (useful for debugging).

    Returns
    -------
    torch.device
    """
    if not gpu_requested:
        device = torch.device("cpu")
        print("GPU not requested — using CPU.")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        _print_gpu_info()
    else:
        device = torch.device("cpu")
        print("No GPU found — falling back to CPU.")

    return device


def _print_gpu_info():
    """Print a short summary of the GPU(s) that will be used."""
    # torch.version.hip is set when PyTorch was built with ROCm
    backend = "ROCm/AMD" if getattr(torch.version, "hip", None) else "CUDA/NVIDIA"
    n = torch.cuda.device_count()
    print(f"Using {backend} backend — {n} GPU(s) available:")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / 1024**3
        print(f"  [{i}] {props.name}  ({vram_gb:.1f} GB)")
