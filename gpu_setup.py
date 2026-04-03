"""
GPU Setup — must be imported BEFORE cupy on Windows.

Adds NVIDIA pip-package DLL directories to Python's DLL search path so that
cupy can find cublas, curand, nvrtc, etc. without needing CUDA Toolkit installed.

Usage:
    import gpu_setup          # at the top of any script
    import cupy as cp         # then works normally
"""

import os
import glob
import sys
import warnings

_SETUP_DONE = False


def setup():
    global _SETUP_DONE
    if _SETUP_DONE:
        return True

    site_packages = os.path.join(sys.exec_prefix, "Lib", "site-packages", "nvidia")

    if not os.path.isdir(site_packages):
        warnings.warn(
            "NVIDIA pip packages not found. Run:\n"
            "  pip install nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 "
            "nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cuda-nvrtc-cu12 "
            "nvidia-cuda-runtime-cu12",
            RuntimeWarning, stacklevel=2
        )
        return False

    added = []
    for dll_path in glob.glob(os.path.join(site_packages, "**", "bin", "*.dll"),
                               recursive=True):
        dll_dir = os.path.dirname(dll_path)
        if dll_dir not in added:
            try:
                os.add_dll_directory(dll_dir)
                added.append(dll_dir)
            except Exception:
                pass

    _SETUP_DONE = True
    return True


# Auto-run on import
setup()
