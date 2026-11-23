"""Lightweight AutoModel used to load a custom borg-cube pipeline from the Hub."""
from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedModel

from .configuration_borg import BorgHFConfig


class BorgHFModel(PreTrainedModel):
    config_class = BorgHFConfig
    base_model_prefix = "borg_cube"

    def __init__(self, config: BorgHFConfig):
        super().__init__(config)
        self.register_parameter(
            "_pipeline_marker",
            torch.nn.Parameter(torch.zeros(1), requires_grad=False),
        )
        self.post_init()

    def forward(self, *args: Any, **kwargs: Any):
        raise RuntimeError(
            "BorgHFModel is a Hub loading marker; inference runs through "
            "the registered BorgHFPipeline."
        )
