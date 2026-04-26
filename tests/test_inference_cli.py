import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class GrammarCorrectCliTest(unittest.TestCase):
    def test_finalize_prediction_applies_exact_constraint(self):
        from scripts.inference.grammar_correct import finalize_prediction

        raw = (
            "**错误类型**: 时态一致\n"
            "**改正**: She told me she would be going to the store.\n"
            "**解释**: reported speech tense shift."
        )

        result = finalize_prediction("She told me she will be going to the store.", raw)

        self.assertEqual(result.error_type, "时态一致")
        self.assertEqual(result.correction, "She told me she was going to the store.")
        self.assertEqual(result.exact_constraint_rule, "reported_speech_was_going_exact")

    def test_finalize_prediction_applies_type_constraint(self):
        from scripts.inference.grammar_correct import finalize_prediction

        raw = (
            "**错误类型**: 时态一致\n"
            "**改正**: The captain married his childhood friend last year.\n"
            "**解释**: corrected."
        )

        result = finalize_prediction("The captain married with his childhood friend last year.", raw)

        self.assertEqual(result.error_type, "介词")
        self.assertEqual(result.type_constraint_rule, "married_with_to_transitive_marry")
        self.assertEqual(result.correction, "The captain married his childhood friend last year.")

    def test_cli_raw_output_json_does_not_load_model(self):
        raw = (
            "**错误类型**: 介词\n"
            "**改正**: Remember to feed the dog when it is hungry.\n"
            "**解释**: corrected."
        )
        cmd = [
            sys.executable,
            "scripts/inference/grammar_correct.py",
            "--sentence",
            "Remember to feed the dog in it's hungry.",
            "--raw-output",
            raw,
            "--json",
        ]

        completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=True)
        payload = json.loads(completed.stdout)

        self.assertEqual(payload["input"], "Remember to feed the dog in it's hungry.")
        self.assertEqual(payload["error_type"], "介词")
        self.assertEqual(payload["correction"], "Remember to feed the dog when it's hungry.")
        self.assertEqual(payload["exact_constraint_rule"], "when_its_hungry")


if __name__ == "__main__":
    unittest.main()
