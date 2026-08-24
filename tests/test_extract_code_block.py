"""Regression tests for core.utils.extract_code_block.

Stdlib-only (unittest). Run directly: python3 tests/test_extract_code_block.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import extract_code_block  # noqa: E402


class ExtractCodeBlockTests(unittest.TestCase):
    def test_python_tagged_fence(self):
        text = "```python\nres_query = 1\n```"
        self.assertEqual(extract_code_block(text), "res_query = 1")

    def test_bare_fence(self):
        text = "```\nres_query = 1\n```"
        self.assertEqual(extract_code_block(text), "res_query = 1")

    def test_bare_fence_with_stray_python_tag_on_its_own_line(self):
        text = "```\npython\nres_query = 1\n```"
        self.assertEqual(extract_code_block(text), "res_query = 1")

    def test_code_starting_with_python_prefixed_identifier_is_not_corrupted(self):
        text = '```\npython_version = "3.10"\nres_query = python_version\n```'
        self.assertEqual(
            extract_code_block(text),
            'python_version = "3.10"\nres_query = python_version',
        )

    def test_python_tagged_fence_with_python_prefixed_identifier(self):
        text = "```python\npython_helper = 2\nres_query = python_helper\n```"
        self.assertEqual(
            extract_code_block(text),
            "python_helper = 2\nres_query = python_helper",
        )

    def test_prose_wrapped_fence(self):
        text = "Sure, here you go:\n```python\nres_query = 1\n```\nLet me know if you need more."
        self.assertEqual(extract_code_block(text), "res_query = 1")

    def test_no_fence_returns_empty_string(self):
        self.assertEqual(extract_code_block("res_query = 1"), "")


if __name__ == "__main__":
    unittest.main()
