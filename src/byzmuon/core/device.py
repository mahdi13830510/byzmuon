import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """CUDA, then Intel XPU, then CPU. A config value other than 'auto' overrides."""
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")
