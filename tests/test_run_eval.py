import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from scripts.run_eval import (
    _write_predictions,
    parse_evaluation_counts,
    parse_evaluation_f1,
)


class TestRunEval(unittest.TestCase):
    def test_passes_entire_input_file_to_borg_once(self):
        contents = "First sentence.\nSecond sentence.\n"
        nlp = Mock(return_value=["first", "second"])

        with tempfile.TemporaryDirectory() as temp_dir:
            text_file = Path(temp_dir) / "test.txt"
            prediction_file = Path(temp_dir) / "prediction.conllu"
            text_file.write_text(contents, encoding="utf-8")

            _write_predictions(nlp, text_file, prediction_file)

            nlp.assert_called_once_with(contents)
            self.assertEqual(
                prediction_file.read_text(encoding="utf-8"), "first\nsecond\n"
            )

    def test_parse_evaluator_counts(self):
        output = """Metric     | Correct   |      Gold | Predicted | Aligned
-----------+-----------+-----------+-----------+-----------
Tokens     |        2 |         2 |         2 |          
Sentences  |        1 |         1 |         1 |          
Words      |        2 |         2 |         2 |         2
"""

        counts = parse_evaluation_counts(output)

        self.assertEqual(counts, {"Tokens": 2, "Words": 2, "Sentences": 1})

    def test_parse_f1_from_verbose_output(self):
        output = """Metric     | Precision |    Recall |  F1 Score | AligndAcc
-----------+-----------+-----------+-----------+-----------
UPOS       |     50.00 |     50.00 |     50.00 |     50.00
XPOS       |     10.00 |     10.00 |     10.00 |     10.00
UFeats     |     75.00 |     75.00 |     75.00 |     75.00
Lemmas     |     80.00 |     80.00 |     80.00 |     80.00
UAS        |     60.00 |     60.00 |     60.00 |     60.00
LAS        |     40.00 |     40.00 |     40.00 |     40.00
"""

        f1 = parse_evaluation_f1(output)

        self.assertEqual(f1["UPOS"], "50.00")
        self.assertEqual(f1["UFeats"], "75.00")
        self.assertNotIn("XPOS", f1)

    def test_parse_rejects_missing_metrics(self):
        with self.assertRaisesRegex(ValueError, "missing metrics"):
            parse_evaluation_counts("Metric | Correct | Gold | Predicted | Aligned\n")


if __name__ == "__main__":
    unittest.main()
