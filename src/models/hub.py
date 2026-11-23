"""Helpers for downloading Borg component repositories from the Hugging Face Hub."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

from huggingface_hub import snapshot_download


def adapter_repo_id(
    base_repo_id: str,
    component: str,
    variant: Optional[str] = None,
) -> str:
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


def adapter_checkpoint_path(
    repo_path: Path,
    component: str,
    checkpoint: str,
) -> Path:
    checkpoint_path = repo_path / checkpoint
    if (checkpoint_path / "borg_config.json").is_file():
        return checkpoint_path
    if (repo_path / "borg_config.json").is_file():
        return repo_path
    raise FileNotFoundError(
        f"Adapter repository snapshot at {str(repo_path)!r} has no "
        f"{checkpoint!r} checkpoint for {component!r}"
    )


def download_component_checkpoints(
    base_repo_id: str,
    components: Sequence[str],
    *,
    variant: Optional[str] = None,
    checkpoint: str = "best",
) -> Dict[str, Path]:
    if checkpoint not in ("best", "last"):
        raise ValueError("checkpoint must be 'best' or 'last'")

    downloaded: Dict[str, Path] = {}
    for component in components:
        repo_id = adapter_repo_id(base_repo_id, component, variant)
        repo_path = Path(snapshot_download(repo_id))
        downloaded[component] = adapter_checkpoint_path(
            repo_path,
            component,
            checkpoint,
        )
    return downloaded
