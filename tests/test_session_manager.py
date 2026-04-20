import unittest
from unittest.mock import patch

from chatbot.session_manager import clear_session, process_turn


class SessionManagerTests(unittest.TestCase):
    def tearDown(self):
        clear_session("test-session")

    @patch("chatbot.session_manager.extract_entities_llm")
    def test_process_turn_collects_missing_fields_across_turns(self, mock_extract):
        mock_extract.side_effect = [
            {
                "age": None,
                "condition": None,
                "food": "rice",
                "gibberish": False,
            },
            {
                "age": 45,
                "condition": None,
                "food": None,
                "gibberish": False,
            },
            {
                "age": None,
                "condition": "diabetes",
                "food": None,
                "gibberish": False,
            },
        ]

        first = process_turn("test-session", "rice")
        second = process_turn("test-session", "I am 45")
        third = process_turn("test-session", "diabetes")

        self.assertEqual(first["action"], "ask_missing")
        self.assertIn("age", first["message"].lower())
        self.assertEqual(second["action"], "ask_missing")
        self.assertIn("medical conditions", second["message"].lower())
        self.assertEqual(third["action"], "run_pipeline")
        self.assertEqual(third["session"].food, "rice")
        self.assertEqual(third["session"].age, 45)
        self.assertEqual(third["session"].condition, "diabetes")

    @patch("chatbot.session_manager.extract_entities_llm")
    def test_process_turn_handles_gibberish_prompt(self, mock_extract):
        mock_extract.return_value = {
            "age": None,
            "condition": None,
            "food": None,
            "gibberish": True,
        }

        result = process_turn("test-session", "!!! ???")

        self.assertEqual(result["action"], "ask_missing")
        self.assertIn("couldn't find any health-related information", result["message"].lower())


if __name__ == "__main__":
    unittest.main()
