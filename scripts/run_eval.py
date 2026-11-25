#!/usr/bin/env python3
"""Run Borg over corpus test inputs and evaluate its CoNLL-U output."""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
EVALUATOR_SCRIPT = Path(__file__).resolve().with_name("conll_ud_eval18.py")
RESULTS_FILE = ROOT / "results.tsv"

COUNT_METRICS = ("Tokens", "Words", "Sentences")
F1_METRICS = (
    "UPOS",
    "UFeats",
    "Lemmas",
    "UAS",
    "LAS",
)


def parse_evaluation_counts(output: str) -> Dict[str, int]:
    """Parse token, word, and sentence totals from evaluator --counts output."""
    metrics: Dict[str, int] = {}
    for line in output.splitlines():
        columns = [column.strip() for column in line.split("|")]
        if len(columns) != 5 or columns[0] not in COUNT_METRICS:
            continue
        metrics[columns[0]] = int(columns[2])

    missing = set(COUNT_METRICS) - metrics.keys()
    if missing:
        raise ValueError(
            "Evaluator output is missing metrics: " + ", ".join(sorted(missing))
        )
    return metrics


def parse_evaluation_f1(output: str) -> Dict[str, str]:
    """Parse F1 percentages directly from evaluator --verbose output."""
    metrics: Dict[str, str] = {}
    for line in output.splitlines():
        columns = [column.strip() for column in line.split("|")]
        if len(columns) != 5 or columns[0] not in F1_METRICS:
            continue
        float(columns[3])
        metrics[columns[0]] = columns[3]

    missing = set(F1_METRICS) - metrics.keys()
    if missing:
        raise ValueError(
            "Evaluator output is missing metrics: " + ", ".join(sorted(missing))
        )
    return metrics


def _run_evaluator(
    gold_file: Path, system_file: Path, option: str
) -> str:
    command = [
        sys.executable,
        str(EVALUATOR_SCRIPT),
        option,
        str(gold_file),
        str(system_file),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"CoNLL-U evaluation failed for {gold_file}:\n"
            f"{result.stderr}{result.stdout}"
        )
    print(result.stdout, end="")
    return result.stdout


def _evaluate(
    gold_file: Path, system_file: Path
) -> Tuple[Dict[str, int], Dict[str, str]]:
    count_output = _run_evaluator(gold_file, system_file, "--counts")
    f1_output = _run_evaluator(gold_file, system_file, "--verbose")
    return parse_evaluation_counts(count_output), parse_evaluation_f1(f1_output)


def _write_predictions(
    nlp: Callable[[str], Iterable[object]],
    text_file: Path,
    prediction_file: Path,
) -> None:
    with text_file.open(encoding="utf-8") as source, prediction_file.open(
        "w", encoding="utf-8"
    ) as output:
        for sentence in nlp(source.read()):
            output.write(f"{sentence}\n")


def main() -> None:
    test_files = sorted(CORPUS_DIR.rglob("test.txt"))
    if not test_files:
        raise FileNotFoundError(f"No test.txt files found under {CORPUS_DIR}")

    test_cases = []
    for text_file in test_files:
        gold_file = text_file.with_suffix(".conllu")
        if not gold_file.is_file():
            raise FileNotFoundError(
                f"Expected gold CoNLL-U file alongside {text_file}: {gold_file}"
            )
        test_cases.append((text_file, gold_file))

    from borg import nlp as load_nlp

    nlp = load_nlp(model="racai/borg")
    rows: List[List[str]] = []
    with tempfile.TemporaryDirectory(prefix="borg-eval-") as temp_dir:
        for index, (text_file, gold_file) in enumerate(test_cases):
            prediction_file = Path(temp_dir) / f"{index}.conllu"
            _write_predictions(nlp, text_file, prediction_file)

            print(f"Evaluating {text_file.relative_to(CORPUS_DIR)}")
            counts, f1 = _evaluate(gold_file, prediction_file)
            rows.append(
                [
                    text_file.relative_to(CORPUS_DIR).as_posix(),
                    str(f1["Tokens"]),
                    str(f1["Words"]),
                    str(f1["Sentences"]),
                    f1["UPOS"],
                    f1["UFeats"],
                    f1["Lemmas"],
                    f1["UAS"],
                    f1["LAS"],
                ]
            )

    with RESULTS_FILE.open("w", encoding="utf-8", newline="") as results:
        writer = csv.writer(results, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "dataset",
                "toks",
                "words",
                "sentences",
                "upos",
                "feats",
                "lemma",
                "uas",
                "las",
            ]
        )
        writer.writerows(rows)
    print(f"Saved results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
