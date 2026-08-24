"""Regression tests for core.sandbox.run_user_code.

Stdlib-only (unittest) since the project has no test runner configured yet.
Run directly: python3 tests/test_sandbox.py
"""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("SANDBOX_TIMEOUT_S", "5")

from core.sandbox import run_user_code_subprocess  # noqa: E402


class RunUserCodeSubprocessTests(unittest.TestCase):
    def test_basic_result_round_trips(self):
        out = run_user_code_subprocess("res_query = 1 + 41", {})
        self.assertEqual(out["res_query"], 42)

    def test_module_values_in_local_env_are_usable(self):
        out = run_user_code_subprocess("res_query = os.path.basename('/a/b/c.txt')", {"os": os})
        self.assertEqual(out["res_query"], "c.txt")

    def test_plain_data_in_local_env_round_trips(self):
        out = run_user_code_subprocess("res_query = X + 1", {"X": 10})
        self.assertEqual(out["res_query"], 11)

    def test_exception_surfaces_as_runtime_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            run_user_code_subprocess("res_query = 1 / 0", {})
        self.assertIn("ZeroDivisionError", str(ctx.exception))

    def test_secret_env_vars_are_not_visible_to_generated_code(self):
        os.environ["OPENAI_API_KEY"] = "sk-should-not-leak"
        os.environ["POSTGRES_PASSWORD"] = "hunter2-should-not-leak"
        try:
            out = run_user_code_subprocess("import os\nres_query = dict(os.environ)", {"os": os})
        finally:
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("POSTGRES_PASSWORD", None)
        leaked = [k for k in out["res_query"] if "API_KEY" in k or "PASSWORD" in k]
        self.assertEqual(leaked, [])

    def test_runaway_code_is_killed_after_timeout(self):
        start = time.time()
        with self.assertRaises(RuntimeError) as ctx:
            run_user_code_subprocess("while True:\n    pass", {})
        elapsed = time.time() - start
        self.assertIn("timed out", str(ctx.exception))
        self.assertLess(elapsed, float(os.environ["SANDBOX_TIMEOUT_S"]) + 5)


if __name__ == "__main__":
    unittest.main()
