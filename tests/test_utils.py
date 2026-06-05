import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from framelens_vl_api.analyzer import read_first_config_value, read_token_file
from framelens_vl_api.utils import extract_json_object, merge_prompt


class UtilsTest(unittest.TestCase):
    def test_extract_plain_json(self):
        self.assertEqual(extract_json_object('{"a": 1}'), {"a": 1})

    def test_extract_fenced_json(self):
        self.assertEqual(extract_json_object('```json\n{"a": 2}\n```'), {"a": 2})

    def test_extract_embedded_json(self):
        self.assertEqual(
            extract_json_object('Here is it:\n{"a": [1, 2]}\nThanks'),
            {"a": [1, 2]},
        )

    def test_extract_json_before_details(self):
        self.assertEqual(
            extract_json_object('{"a": 3}\n\n<details>trace</details>'),
            {"a": 3},
        )

    def test_merge_prompt(self):
        self.assertIn("extra", merge_prompt("base", "extra"))

    def test_read_token_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "token.txt"
            path.write_text("# comment\nQWEN_FREE_API_TOKEN='abc123'\n", encoding="utf-8")
            self.assertEqual(read_token_file(path), "abc123")

    def test_read_model_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.txt"
            path.write_text("# comment\nQWEN_FREE_API_MODEL=qwen3.7-plus\n", encoding="utf-8")
            self.assertEqual(
                read_first_config_value(path, assignment_name="QWEN_FREE_API_MODEL"),
                "qwen3.7-plus",
            )


if __name__ == "__main__":
    unittest.main()
