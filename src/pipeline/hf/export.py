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
        "This repository contains the borg-cube pipeline configuration and "
        "runtime code, but not its component weights. Publish each component "
        "to a model repository named by appending `-<component>` to this "
        "repository ID (for example, `racai-ai/mixed-tokenizer`). The pipeline "
        "downloads the listed components at runtime. Pass `adapter_variant` "
        "to select repositories with an additional suffix, such as "
        "`racai/borg-tokenizer-combined`; omit it for default repositories. "
        "Its custom pipeline code is loaded by Transformers; review it before enabling "
        "`trust_remote_code`.\n\n"
        "Install the dependencies listed in `requirements.txt`, then run:\n\n"
        "```python\n"
        "from transformers import pipeline\n\n"
        'nlp = pipeline(model="your-username/your-nlp-repo", '
        "trust_remote_code=True, adapter_variant='combined')\n"
        'result = nlp("Your input text here")\n'
        "```\n",
        encoding="utf-8",
    )
    return output_path
