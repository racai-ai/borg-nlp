"""Create a Hub repository directory for ``transformers.pipeline``."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional, Sequence, Union

from src.config import BorgConfig
from src.pipeline.hf.configuration_borg import BorgHFConfig
from src.pipeline.hf.modeling_borg import BorgHFModel
from src.pipeline.hf.pipeline import _COMPONENTS


def export_hf_repo(
    output_dir: Union[str, Path],
    *,
    config: Optional[BorgConfig] = None,
    components: Optional[Sequence[str]] = None,
    checkpoint: str = "best",
) -> Path:
    """Export a Hub pipeline that references separately uploaded adapters.

    The output directory is ready to upload to the base Hub repository. It
    includes the generated ``config.json`` and remote runtime code. Component
    checkpoints are uploaded separately to repositories named
    ``<base-repo>-<component>``.
    """
    if checkpoint not in ("best", "last"):
        raise ValueError("checkpoint must be 'best' or 'last'")

    output_path = Path(output_dir).expanduser()
    selected_components = tuple(
        _COMPONENTS if components is None else components
    )
    if not selected_components:
        raise ValueError("At least one component must be selected")
    if isinstance(components, str):
        raise TypeError("components must be a sequence of component names")
    unknown = set(selected_components) - set(_COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown components: {', '.join(sorted(unknown))}")
    if len(set(selected_components)) != len(selected_components):
        raise ValueError("components must not contain duplicates")

    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Export directory is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    package_root = Path(__file__).resolve().parents[3]
    copy_ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for package in ("src", "eval"):
        shutil.copytree(
            package_root / package,
            output_path / package,
            ignore=copy_ignore,
        )
    for source_name, target_name in (
        ("configuration_borg.py", "configuration_borg.py"),
        ("modeling_borg.py", "modeling_borg.py"),
        ("pipeline.py", "pipeline.py"),
    ):
        shutil.copyfile(
            Path(__file__).with_name(source_name),
            output_path / target_name,
        )
    requirements_path = package_root / "requirements.txt"
    if requirements_path.is_file():
        shutil.copyfile(requirements_path, output_path / "requirements.txt")
    else:
        (output_path / "requirements.txt").write_text(
            "torch>=2.0.0\n"
            "transformers>=4.35.0\n"
            "huggingface_hub>=0.19.0\n"
            "adapters>=0.1.1\n"
            "numpy>=1.24.0\n"
            "tqdm>=4.65.0\n"
            "sentencepiece>=0.1.99\n"
            "protobuf>=3.20.0\n",
            encoding="utf-8",
        )

    with Path(__file__).with_name("config.json").open(encoding="utf-8") as source:
        hub_config_values = json.load(source)
    hub_config_values.update(
        borg_config=config.to_dict() if config is not None else {},
        components=list(selected_components),
        checkpoint=checkpoint,
    )
    hub_config = BorgHFConfig(**hub_config_values)
    hub_model = BorgHFModel(hub_config)
    hub_model.save_pretrained(str(output_path), safe_serialization=False)

    (output_path / "README.md").write_text(
        "# Borg Cube\n\n"
        "This is the pipeline for the BORG Natural Language Processing "
        "framework. For more support, you can go to the official "
        "repo(https://github.com/racai-ai/borg-nlp) and/or use the standalone "
        "PIP package (borg-nlp)\n\n"
        "**Before proceeding**: You will need the following packages "
        "installed: `adapters` `torch` `protobuf` `sentencepiece>=0.1.99`\n\n"
        "```python\n"
        "from transformers import pipeline\n\n"
        'nlp = pipeline(model="racai/borg", trust_remote_code=True)\n'
        'result = nlp("Your input text here")\n'
        "```\n",
        encoding="utf-8",
    )
    return output_path
