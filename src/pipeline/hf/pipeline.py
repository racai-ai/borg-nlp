"""Transformers Pipeline adapter for borg-cube checkpoints."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from huggingface_hub import snapshot_download
from transformers.pipelines import Pipeline


_COMPONENTS = ("tokenizer", "mwt", "tagger", "parser", "lemmatizer")


def _read_component(path: Path) -> str:
    with (path / "borg_config.json").open(encoding="utf-8") as source:
        component = json.load(source).get("component")
    if component not in _COMPONENTS:
        raise ValueError(f"Checkpoint at {str(path)!r} has an invalid component")
    return component


def _discover_checkpoints(
    model_path: Path,
    components: Sequence[str],
    checkpoint: str,
) -> Dict[str, Path]:
    if not model_path.is_dir():
        raise FileNotFoundError(f"Borg model directory not found: {model_path}")

    direct_config = model_path / "borg_config.json"
    if direct_config.is_file():
        component = _read_component(model_path)
        if component not in components:
            raise ValueError(
                f"The checkpoint contains {component!r}, which is not in components"
            )
        return {component: model_path}

    if model_path.name in _COMPONENTS:
        checkpoint_path = model_path / checkpoint
        if (checkpoint_path / "borg_config.json").is_file():
            return {model_path.name: checkpoint_path}

    found: Dict[str, Path] = {}
    for component in components:
        component_path = model_path / component
        if (component_path / "borg_config.json").is_file():
            found[component] = component_path
            continue

        checkpoint_path = component_path / checkpoint
        if (checkpoint_path / "borg_config.json").is_file():
            found[component] = checkpoint_path

    if not found:
        raise FileNotFoundError(
            f"No {checkpoint!r} Borg checkpoints found under {model_path}"
        )
    return found


def _load_config(path: Path):
    from src.config import BorgConfig

    with (path / "borg_config.json").open(encoding="utf-8") as source:
        return BorgConfig.from_dict(json.load(source))


def _sentence_to_dict(sentence) -> Dict[str, Any]:
    return {
        "comments": list(sentence.comments),
        "tokens": [
            {
                "id": token.id,
                "form": token.form,
                "lemma": token.lemma,
                "upos": token.upos,
                "xpos": token.xpos,
                "feats": token.feats_str(),
                "head": token.head,
                "deprel": token.deprel,
                "deps": token.deps,
                "misc": token.misc,
                "space_after": token.space_after,
                "is_mwt": token.is_mwt,
            }
            for token in sentence.tokens
        ],
    }


class _PassthroughTokenizer:
    pad_token_id = 0
    padding_side = "right"


def _repository_path(model) -> Path:
    model_id = getattr(model.config, "_name_or_path", None)
    if not model_id:
        raise ValueError("The Borg model config does not identify its Hub repository")

    path = Path(model_id).expanduser()
    if path.is_dir():
        return path
    return Path(
        snapshot_download(
            model_id,
            revision=getattr(model.config, "_commit_hash", None),
        )
    )


def _runtime_device(config, device) -> Any:
    import torch

    if device is not None:
        if isinstance(device, int):
            config.device = "cpu" if device < 0 else f"cuda:{device}"
        else:
            config.device = str(device)
    return torch.device(config.resolve_device())


def _adapter_repo_id(base_repo_id: str, component: str, variant: Optional[str]) -> str:
    if variant is None:
        return f"{base_repo_id}-{component}"
    if (
        not isinstance(variant, str)
        or not variant
        or variant.strip() != variant
        or "/" in variant
        or "\\" in variant
    ):
        raise ValueError("adapter_variant must be a non-empty repository name suffix")
    return f"{base_repo_id}-{component}-{variant}"


def _load_engine(model, device, adapter_variant: Optional[str] = None):
    repo_path = _repository_path(model)
    repo_string = str(repo_path)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)

    from src.config import BorgConfig
    from src.pipeline.pipeline import BorgPipeline

    adapter_checkpoints = []
    for component in model.config.components:
        adapter_repo_id = _adapter_repo_id(
            model.config._name_or_path,
            component,
            adapter_variant,
        )
        adapter_repo_path = Path(snapshot_download(adapter_repo_id))
        checkpoint_path = adapter_repo_path / model.config.checkpoint
        if not (checkpoint_path / "borg_config.json").is_file():
            if not (adapter_repo_path / "borg_config.json").is_file():
                raise FileNotFoundError(
                    f"Repository {adapter_repo_id!r} has no "
                    f"{model.config.checkpoint!r} checkpoint for {component!r}"
                )
            checkpoint_path = adapter_repo_path
        adapter_checkpoints.append((component, checkpoint_path))

    if not adapter_checkpoints:
        raise ValueError("The Borg pipeline config does not list any adapters")

    if model.config.borg_config:
        config = BorgConfig.from_dict(model.config.borg_config)
    else:
        config = _load_config(adapter_checkpoints[0][1])
    runtime_device = _runtime_device(config, device)
    engine = BorgPipeline(config)
    for component, checkpoint_path in adapter_checkpoints:
        engine.load_component(component, str(checkpoint_path))

    return engine, runtime_device


class BorgHFPipeline(Pipeline):
    """Hub-loadable pipeline for complete borg-cube checkpoints."""

    def __init__(
        self,
        model,
        tokenizer=None,
        framework: Optional[str] = None,
        task: str = "borg-cube",
        device=None,
        adapter_variant: Optional[str] = None,
        **kwargs,
    ):
        self.engine, runtime_device = _load_engine(
            model,
            device,
            adapter_variant=adapter_variant,
        )
        super().__init__(
            model=model,
            tokenizer=(
                tokenizer if tokenizer is not None else _PassthroughTokenizer()
            ),
            framework=framework or "pt",
            task=task,
            device=runtime_device,
            **kwargs,
        )

    def _sanitize_parameters(self, show_progress: bool = False, **kwargs):
        if kwargs:
            raise TypeError(
                f"Unexpected pipeline parameters: {', '.join(sorted(kwargs))}"
            )
        return {}, {"show_progress": show_progress}, {}

    def preprocess(self, input_: Any, **preprocess_parameters) -> Dict[str, Any]:
        if not isinstance(input_, str):
            raise TypeError("BorgHFPipeline inputs must be strings")
        return {"text": input_}

    def _forward(self, input_tensors: Dict[str, Any], show_progress: bool = False):
        texts = input_tensors["text"]
        if isinstance(texts, str):
            texts = [texts]
        return {
            "sentences": [
                self.engine.process(text, show_progress=show_progress)
                for text in texts
            ]
        }

    def postprocess(self, model_outputs: Any) -> List[Dict[str, Any]]:
        if isinstance(model_outputs, dict):
            sentences = model_outputs["sentences"][0]
        else:
            sentences = model_outputs
        return [_sentence_to_dict(sentence) for sentence in sentences]
