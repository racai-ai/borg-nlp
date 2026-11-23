"""Hugging Face configuration for exported borg-cube pipelines."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig


class BorgHFConfig(PretrainedConfig):
    model_type = "borg_cube"

    def __init__(
        self,
        borg_config: Optional[Dict[str, Any]] = None,
        components: Optional[List[str]] = None,
        checkpoint: str = "best",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.borg_config = borg_config or {}
        self.components = components if components is not None else []
        self.checkpoint = checkpoint
