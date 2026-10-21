"""Environment Manager: inspects the runtime and guards GPU benchmarks."""
import sys
import platform


def get_environment_info() -> dict:
    """Collect software and GPU details. Never raises if CUDA is missing."""
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        info["torch_version"] = None
        info["cuda_available"] = False
        return info

    try:
        import triton
        info["triton_version"] = triton.__version__
    except ImportError:
        info["triton_version"] = None

    if info["cuda_available"]:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)
        info["compute_capability"] = f"{props.major}.{props.minor}"
        info["multiprocessor_count"] = props.multi_processor_count
    return info


def print_environment_info(info: dict) -> None:
    print("=" * 50)
    print(" ENVIRONMENT REPORT")
    print("=" * 50)
    for key, value in info.items():
        print(f"{key:22s}: {value}")
    print("=" * 50)


def require_gpu() -> dict:
    """Return env info if a CUDA GPU + Triton are usable, else raise a clear error."""
    info = get_environment_info()
    if info.get("torch_version") is None:
        raise RuntimeError("PyTorch is not installed in this runtime.")
    if info.get("triton_version") is None:
        raise RuntimeError("Triton is not installed in this runtime.")
    if not info["cuda_available"]:
        raise RuntimeError(
            "No CUDA GPU detected. In Colab: Runtime > Change runtime type > "
            "Hardware accelerator = T4 GPU, then Runtime > Restart session."
        )
    return info
