import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from borg.cli import _build_parser, main


class TestCLIUpload(unittest.TestCase):
    def test_component_upload_matches_documented_call(self):
        args = _build_parser().parse_args([
            "upload",
            "data/models/mixed/tokenizer/",
            "racai-ai/borg-tokenizer-mixed",
        ])
        self.assertEqual(args.model_path, "data/models/mixed/tokenizer/")
        self.assertEqual(args.repo_id, "racai-ai/borg-tokenizer-mixed")
        self.assertFalse(args.pipeline)

    def test_upload_sends_component_directory_as_is(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("borg.cli.HfApi") as api_class, patch("builtins.print"):
                main(["upload", temp_dir, "racai-ai/borg-tokenizer-mixed"])

        api = api_class.return_value
        api.create_repo.assert_called_once_with(
            "racai-ai/borg-tokenizer-mixed",
            repo_type="model",
            exist_ok=True,
        )
        api.upload_folder.assert_called_once_with(
            repo_id="racai-ai/borg-tokenizer-mixed",
            repo_type="model",
            folder_path=temp_dir,
        )

    def test_pipeline_upload_needs_no_local_model_directory(self):
        def export(output_dir, checkpoint, components):
            Path(output_dir).mkdir()
            self.assertEqual(checkpoint, "last")
            self.assertIsNone(components)
            return Path(output_dir)

        with patch("src.pipeline.hf.export.export_hf_repo", side_effect=export) as exporter:
            with patch("borg.cli.HfApi") as api_class, patch("builtins.print"):
                main([
                    "upload",
                    "racai-ai/borg-mixed",
                    "--pipeline",
                    "--checkpoint",
                    "last",
                ])

        exporter.assert_called_once()
        self.assertEqual(len(exporter.call_args.args), 1)
        self.assertEqual(
            exporter.call_args.kwargs,
            {"components": None, "checkpoint": "last"},
        )
        api_class.return_value.upload_folder.assert_called_once()
        self.assertEqual(
            api_class.return_value.upload_folder.call_args.kwargs["repo_id"],
            "racai-ai/borg-mixed",
        )


if __name__ == "__main__":
    unittest.main()
