import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

from src.config import BorgConfig
from src.data.conllu import Sentence, Token
from src.pipeline.hf.export import export_hf_repo
from src.pipeline.hf.pipeline import (
    _adapter_repo_id,
    _discover_checkpoints,
    _load_engine,
    _sentence_to_dict,
)


class TestHuggingFacePipeline(unittest.TestCase):
    def test_sentence_output_is_json_serializable_and_preserves_annotations(self):
        sentence = Sentence(
            comments=["# sent_id = 1"],
            tokens=[
                Token(
                    id=1,
                    form="Dogs",
                    lemma="dog",
                    upos="NOUN",
                    feats={"Number": "Plur"},
                    head=2,
                    deprel="nsubj",
                    space_after=False,
                )
            ],
        )

        result = _sentence_to_dict(sentence)

        self.assertEqual(result["comments"], ["# sent_id = 1"])
        self.assertEqual(result["tokens"][0]["form"], "Dogs")
        self.assertEqual(result["tokens"][0]["feats"], "Number=Plur")
        self.assertFalse(result["tokens"][0]["space_after"])
        json.dumps(result)

    def test_discovers_selected_component_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for component in ("tokenizer", "tagger"):
                checkpoint = root / component / "best"
                checkpoint.mkdir(parents=True)
                (checkpoint / "borg_config.json").write_text(
                    json.dumps({"component": component}),
                    encoding="utf-8",
                )

            found = _discover_checkpoints(
                root,
                ("tagger",),
                "best",
            )

        self.assertEqual(list(found), ["tagger"])
        self.assertEqual(found["tagger"], root / "tagger" / "best")

    def test_export_writes_hub_custom_pipeline_metadata_and_code(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "hub_model"

            export_hf_repo(
                output_dir,
                config=BorgConfig(),
                components=("tagger",),
            )

            with (output_dir / "config.json").open(encoding="utf-8") as source:
                config = json.load(source)

            self.assertEqual(config["model_type"], "borg_cube")
            self.assertEqual(config["components"], ["tagger"])
            self.assertEqual(config["checkpoint"], "best")
            self.assertEqual(
                config["auto_map"]["AutoModel"],
                "modeling_borg.BorgHFModel",
            )
            self.assertEqual(
                config["custom_pipelines"]["borg-cube"]["impl"],
                "pipeline.BorgHFPipeline",
            )
            for name in (
                "pipeline.py",
                "configuration_borg.py",
                "modeling_borg.py",
                "pytorch_model.bin",
                "src/pipeline/pipeline.py",
                "eval/conll18_ud_eval.py",
            ):
                self.assertTrue((output_dir / name).is_file(), name)
            self.assertFalse((output_dir / "tagger").exists())

    def test_export_needs_no_local_adapter_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "hub_model"
            export_hf_repo(output_dir, components=("tokenizer", "parser"))

            with (output_dir / "config.json").open(encoding="utf-8") as source:
                config = json.load(source)

            self.assertEqual(config["components"], ["tokenizer", "parser"])
            self.assertEqual(config["borg_config"], {})
            self.assertFalse((output_dir / "tokenizer").exists())

    def test_pipeline_downloads_adapters_from_base_repo_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            def download(repo_id, revision=None):
                repo_path = root / repo_id.rsplit("/", 1)[-1]
                repo_path.mkdir(parents=True, exist_ok=True)
                if repo_id == "racai-ai/mixed":
                    return str(repo_path)
                component = repo_id.rsplit("-", 2)[-2]
                (repo_path / "borg_config.json").write_text(
                    json.dumps({"component": component, "lang": "fr"}),
                    encoding="utf-8",
                )
                return str(repo_path)

            model = SimpleNamespace(
                config=SimpleNamespace(
                    _name_or_path="racai-ai/mixed",
                    _commit_hash=None,
                    borg_config={},
                    components=["tokenizer", "parser"],
                    checkpoint="best",
                )
            )
            with patch(
                "src.pipeline.hf.pipeline.snapshot_download",
                side_effect=download,
            ) as download_mock:
                with patch(
                    "src.pipeline.hf.pipeline._runtime_device",
                    return_value="cpu",
                ):
                    with patch(
                        "src.pipeline.pipeline.BorgPipeline"
                    ) as pipeline_class:
                        _load_engine(model, None, adapter_variant="combined")

        self.assertEqual(
            download_mock.call_args_list,
            [
                call("racai-ai/mixed", revision=None),
                call("racai-ai/mixed-tokenizer-combined"),
                call("racai-ai/mixed-parser-combined"),
            ],
        )
        engine = pipeline_class.return_value
        engine.load_component.assert_has_calls(
            [
                call(
                    "tokenizer",
                    str(root / "mixed-tokenizer-combined"),
                ),
                call("parser", str(root / "mixed-parser-combined")),
            ]
        )
        self.assertEqual(pipeline_class.call_args.args[0].lang, "fr")

    def test_adapter_repo_id_omits_suffix_for_default_variant(self):
        self.assertEqual(
            _adapter_repo_id("racai/borg", "tokenizer", None),
            "racai/borg-tokenizer",
        )
        self.assertEqual(
            _adapter_repo_id("racai/borg", "tokenizer", "combined"),
            "racai/borg-tokenizer-combined",
        )
        with self.assertRaises(ValueError):
            _adapter_repo_id("racai/borg", "tokenizer", "bad/name")


if __name__ == "__main__":
    unittest.main()
