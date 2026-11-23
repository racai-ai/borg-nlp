"""Public Python API for borg-cube.

Usage::

    from borg import cube
    nlp = cube('en')
    sentences = nlp("The cat sat on the mat.")
    print(sentences)
"""
from __future__ import annotations

import os
from typing import List, Mapping, Optional
from urllib.parse import unquote

from src.config import BorgConfig
from src.data.conllu import Sentence
from src.pipeline.pipeline import BorgPipeline


class Cube:
    """High-level interface to the borg-cube NLP pipeline.

    Parameters
    ----------
    lang:
        Language code (used for locating default model files).
    model_path:
        Base directory containing sub-directories for each component
        (``tokenizer/``, ``mwt/``, ``tagger/``, ``parser/``, ``lemmatizer/``).
        If *None*, defaults to ``~/.borg_cube/<lang>/``.
    components:
        List of components to load. Defaults to all pipeline components.
    config:
        Optional :class:`~src.config.BorgConfig` instance.
    """

    DEFAULT_COMPONENTS = ["tokenizer", "mwt", "tagger", "parser", "lemmatizer"]

    def __init__(
        self,
        lang: str = "en",
        model_path: Optional[str] = None,
        components: Optional[List[str]] = None,
        config: Optional[BorgConfig] = None,
        _component_paths: Optional[Mapping[str, str]] = None,
    ):
        self.lang = lang
        if model_path is None and _component_paths is None:
            model_path = os.path.join(os.path.expanduser("~"), ".borg_cube", lang)
        self.model_path = model_path

        if config is None:
            config = BorgConfig(lang=lang)
        self.config = config

        components = self.DEFAULT_COMPONENTS if components is None else components

        self.pipeline = BorgPipeline(config)
        for component in components:
            if _component_paths is not None:
                self.pipeline.load_component(
                    component,
                    _component_paths[component],
                )
            else:
                component_dir = os.path.join(model_path, component)
                if os.path.isdir(component_dir):
                    self.pipeline.load_component(component, component_dir)

    # ------------------------------------------------------------------
    def __call__(self, text: str) -> List[Sentence]:
        """Process *text* and return a list of annotated :class:`Sentence` objects."""
        return self.pipeline.process(text)

    def __repr__(self) -> str:
        loaded = [
            c for c in self.DEFAULT_COMPONENTS
            if getattr(self.pipeline, f"{c}_model") is not None
        ]
        return f"Cube(lang={self.lang!r}, loaded={loaded})"


def cube(
    model: str,
    components: Optional[List[str]] = None,
    config: Optional[BorgConfig] = None,
    *,
    adapter_variant: Optional[str] = None,
    checkpoint: str = "best",
) -> Cube:
    """Load a local model or download component adapters from the Hub.

    ``cube("en")`` loads local components from ``~/.borg_cube/en/``.
    ``cube("file:///path/to/models")`` loads from a local folder.
    A Hub base ID such as ``"racai/borg"`` downloads component repos named
    ``racai/borg-<component>``; ``adapter_variant="combined"`` appends
    ``-combined`` to each component repo name. No base pipeline repo is needed.
    """
    if model.startswith("file://"):
        model_path = unquote(model[len("file://"):])
        if not model_path:
            raise ValueError("A file:// model URI must include a folder path")
        lang = os.path.basename(os.path.normpath(model_path)) or "en"
        return Cube(lang=lang, model_path=model_path, components=components, config=config)

    if model.count("/") == 1 and not model.startswith((".", "/")):
        selected_components = (
            Cube.DEFAULT_COMPONENTS if components is None else components
        )
        if not selected_components:
            return Cube(
                lang=config.lang if config is not None else "en",
                components=[],
                config=config,
                _component_paths={},
            )
        unknown = set(selected_components) - set(Cube.DEFAULT_COMPONENTS)
        if unknown:
            raise ValueError(f"Unknown components: {', '.join(sorted(unknown))}")

        from src.models.hub import download_component_checkpoints

        component_paths = download_component_checkpoints(
            model,
            selected_components,
            variant=adapter_variant,
            checkpoint=checkpoint,
        )
        if config is None:
            import json

            first_checkpoint = next(iter(component_paths.values()))
            with (first_checkpoint / "borg_config.json").open(
                encoding="utf-8"
            ) as source:
                config = BorgConfig.from_dict(json.load(source))
            config.device = "auto"

        return Cube(
            lang=config.lang,
            components=selected_components,
            config=config,
            _component_paths={
                component: str(path)
                for component, path in component_paths.items()
            },
        )

    return Cube(lang=model, components=components, config=config)
