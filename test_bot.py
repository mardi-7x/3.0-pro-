"""
Automated Unit and Integration Test Suite for Groq Telegram AI Chatbot.
Covers environment validation, conversation isolation, SQLite history trimming,
rate limiting, message splitting, AST deterministic math evaluation, and mock API handling.
"""

import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from main import (
    AppConfig,
    DatabaseManager,
    HallucinationGuard,
    SafeMathEvaluator,
    TelegramMessageFormatter,
    UserRateLimiter,
)


class TestAppConfig(unittest.TestCase):
    """Tests environment variable loading, validation, and secret masking."""

    def test_missing_required_env_vars_raises_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                AppConfig.from_env(validate_now=True)
            self.assertIn("TELEGRAM_BOT_TOKEN", str(ctx.exception))
            self.assertIn("GROQ_API_KEY", str(ctx.exception))

    def test_valid_env_vars_loads_cleanly(self):
        env = {
            "TELEGRAM_BOT_TOKEN": "123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
            "GROQ_API_KEY": "gsk_testsecret1234567890abcdef",
            "GROQ_MODEL": "llama-3.3-70b-versatile",
            "MAX_HISTORY_MESSAGES": "30",
            "PORT": "8080",
        }
        with patch.dict(os.environ, env, clear=True):
            config = AppConfig.from_env(validate_now=True)
            self.assertEqual(config.groq_model, "llama-3.3-70b-versatile")
            self.assertEqual(config.max_history_messages, 30)
            self.assertEqual(config.port, 8080)

    def test_secret_masking_never_exposes_token(self):
        config = AppConfig(
            telegram_bot_token="1234567890:MY_SECRET_BOT_TOKEN",
            groq_api_key="gsk_super_secret_groq_key_99999",
        )
        safe = config.safe_repr()
        self.assertNotIn("MY_SECRET_BOT_TOKEN", safe)
        self.assertNotIn("super_secret_groq_key", safe)
        self.assertIn("***", safe)


class TestDatabaseManager(unittest.TestCase):
    """Tests SQLite persistent conversation storage, history trimming, and user isolation."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db = DatabaseManager(self.temp_db.name)

    def tearDown(self):
        try:
            os.remove(self.temp_db.name)
        except OSError:
            pass

    def test_conversation_isolation_between_users(self):
        user_a = 1001
        user_b = 2002

        self.db.add_message(user_a, "user", "Hello from User A")
        self.db.add_message(user_a, "assistant", "Response to User A")

        self.db.add_message(user_b, "user", "Hello from User B")
        self.db.add_message(user_b, "assistant", "Response to User B")

        hist_a = self.db.get_history(user_a)
        hist_b = self.db.get_history(user_b)

        # Check User A never sees User B's content
        self.assertEqual(len(hist_a), 2)
        self.assertEqual(hist_a[0]["content"], "Hello from User A")
        self.assertEqual(hist_a[1]["content"], "Response to User A")

        self.assertEqual(len(hist_b), 2)
        self.assertEqual(hist_b[0]["content"], "Hello from User B")
        self.assertEqual(hist_b[1]["content"], "Response to User B")

    def test_history_trimming_respects_limit(self):
        user_id = 9999
        for i in range(15):
            self.db.add_message(user_id, "user", f"Question {i}")
            self.db.add_message(user_id, "assistant", f"Answer {i}")

        # Total 30 messages stored, request limit 6
        recent = self.db.get_history(user_id, limit=6)
        self.assertEqual(len(recent), 6)
        # Oldest in the 6-item window should be Question 12
        self.assertEqual(recent[0]["content"], "Question 12")
        self.assertEqual(recent[-1]["content"], "Answer 14")

    def test_clear_history(self):
        user_id = 5555
        self.db.add_message(user_id, "user", "Message 1")
        self.db.add_message(user_id, "assistant", "Message 2")

        self.assertEqual(len(self.db.get_history(user_id)), 2)
        deleted = self.db.clear_history(user_id)
        self.assertEqual(deleted, 2)
        self.assertEqual(len(self.db.get_history(user_id)), 0)


class TestUserRateLimiter(unittest.TestCase):
    """Tests per-user sliding window rate limiting."""

    def test_rate_limiter_allows_and_blocks(self):
        limiter = UserRateLimiter(max_requests=3, window_seconds=60.0)
        user_id = 42

        # 3 requests allowed
        self.assertTrue(limiter.is_allowed(user_id)[0])
        self.assertTrue(limiter.is_allowed(user_id)[0])
        self.assertTrue(limiter.is_allowed(user_id)[0])

        # 4th request blocked
        allowed, wait_sec = limiter.is_allowed(user_id)
        self.assertFalse(allowed)
        self.assertGreater(wait_sec, 0)

        # Another user is not affected (user isolation)
        other_user = 99
        self.assertTrue(limiter.is_allowed(other_user)[0])


class TestDeterministicMathGuard(unittest.TestCase):
    """Tests deterministic AST calculation safeguard against hallucinations."""

    def test_simple_arithmetic(self):
        res = SafeMathEvaluator.try_evaluate("What is 154 * 23?")
        self.assertIsNotNone(res)
        expr, val = res
        self.assertEqual(val, "3542")

    def test_complex_math_functions(self):
        res = SafeMathEvaluator.try_evaluate("calculate sqrt(144) + 18")
        self.assertIsNotNone(res)
        expr, val = res
        self.assertEqual(val, "30")

    def test_division_by_zero_fails_safely(self):
        res = SafeMathEvaluator.try_evaluate("What is 10 / 0?")
        self.assertIsNone(res)

    def test_disallows_code_injection(self):
        res = SafeMathEvaluator.try_evaluate("__import__('os').system('ls')")
        self.assertIsNone(res)


class TestTelegramMessageFormatter(unittest.TestCase):
    """Tests response splitting without breaking code blocks."""

    def test_short_message_not_split(self):
        text = "Hello, world!"
        chunks = TelegramMessageFormatter.split_message(text, max_length=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Hello, world!")

    def test_long_message_splits_at_length(self):
        # 3 lines of 40 chars each = 120 chars
        lines = ["A" * 40, "B" * 40, "C" * 40]
        text = "\n".join(lines)
        chunks = TelegramMessageFormatter.split_message(text, max_length=70)
        self.assertGreater(len(chunks), 1)

    def test_code_block_fence_is_closed_and_reopened(self):
        code_body = "\n".join([f"line_{i} = {i}" for i in range(20)])
        long_code = f"```python\n{code_body}\n```"
        chunks = TelegramMessageFormatter.split_message(long_code, max_length=120)

        self.assertGreater(len(chunks), 1)
        # First chunk should have closed with ```
        self.assertTrue(chunks[0].strip().endswith("```"))
        # Second chunk should have reopened with ```python
        self.assertTrue(chunks[1].strip().startswith("```python"))


class TestHallucinationGuardPrompt(unittest.TestCase):
    """Tests prompt construction and uncertainty guidance."""

    def test_system_prompt_contains_rules(self):
        prompt = HallucinationGuard.get_system_prompt()
        self.assertIn("Never invent facts", prompt)
        self.assertIn("I cannot verify this information with the available evidence", prompt)
        self.assertIn("Bengali (বাংলা)", prompt)


if __name__ == "__main__":
    unittest.main()
