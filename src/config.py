"""Configuration dataclass for borg-cube."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Mapping
import warnings

import torch


_WARNED_BFLOAT16_FALLBACKS: set[str] = set()


@lru_cache(maxsize=None)
def _supports_bfloat16(device: str) -> bool:
    target = torch.device(device)
    if target.type == "cuda":
        if not torch.cuda.is_available():
            return False
        try:
            with torch.cuda.device(target):
                return torch.cuda.is_bf16_supported()
        except (RuntimeError, TypeError):
            return False

    if target.type == "mps" and not torch.backends.mps.is_available():
        return False
    if target.type not in ("cpu", "mps", "xpu"):
        return False

    try:
        values = torch.ones((2, 2), device=target)
        with torch.autocast(device_type=target.type, dtype=torch.bfloat16):
            result = values @ values
        return result.dtype is torch.bfloat16
    except (AttributeError, NotImplementedError, RuntimeError, TypeError):
        return False


@dataclass
class BorgConfig:
    model_name: str = "microsoft/deberta-v3-large"
    max_seq_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 10
    warmup_ratio: float = 0
    adapter_reduction_factor: int = 16
    seed: int = 42
    lang: str = "en"
    dtype: torch.dtype = torch.bfloat16
    device: str = "auto"  # "auto", "cpu", "cuda"
    components: List[str] = field(
        default_factory=lambda: ["tokenizer", "tagger", "parser", "lemmatizer"]
    )
    eval_batch_size: int = 32

    def to_dict(self) -> Dict[str, Any]:
        """Return checkpoint metadata with a JSON-compatible dtype."""
        values = dict(self.__dict__)
        values["dtype"] = str(self.dtype)
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "BorgConfig":
        """Restore checkpoint metadata, including its PyTorch dtype."""
        fields = {k: v for k, v in values.items() if k in cls.__dataclass_fields__}
        if "dtype" in fields:
            dtype = fields["dtype"]
            if isinstance(dtype, str):
                name = dtype[len("torch."):] if dtype.startswith("torch.") else dtype
                dtype = getattr(torch, name, None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"Invalid checkpoint dtype: {fields['dtype']!r}")
            fields["dtype"] = dtype
        return cls(**fields)

    def resolve_device(self) -> str:
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device

    def resolve_dtype(self, device: str | None = None) -> torch.dtype:
        """Resolve the requested dtype for a runtime device."""
        runtime_device = device or self.resolve_device()
        if self.dtype is not torch.bfloat16 or _supports_bfloat16(runtime_device):
            return self.dtype

        if runtime_device not in _WARNED_BFLOAT16_FALLBACKS:
            warnings.warn(
                f"Device {runtime_device!r} does not support bfloat16 autocast; "
                "using float32 for this run. BorgConfig.dtype remains bfloat16.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_BFLOAT16_FALLBACKS.add(runtime_device)
        return torch.float32

    def autocast_context(self, device: str | None = None):
        """Create an autocast context using a dtype supported by the runtime device."""
        runtime_device = device or self.resolve_device()
        device_type = torch.device(runtime_device).type
        dtype = self.resolve_dtype(runtime_device)
        enabled = (
            device_type in ("cuda", "cpu", "mps", "xpu")
            and dtype in (torch.float16, torch.bfloat16)
        )
        return torch.autocast(
            device_type=device_type,
            dtype=dtype,
            enabled=enabled,
        )