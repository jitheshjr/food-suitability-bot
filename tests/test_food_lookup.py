import unittest

from chatbot.food_lookup import lookup_food


class FoodLookupTests(unittest.TestCase):
    def test_lookup_known_food_returns_found_record(self):
        result = lookup_food("banana")

        self.assertTrue(result["found"])
        self.assertIn("food_name", result)
        self.assertGreaterEqual(result["gi_value"], 0)

    def test_lookup_unknown_food_uses_default_profile(self):
        result = lookup_food("totally_unknown_food_123")

        self.assertFalse(result["found"])
        self.assertEqual(result["food_name"], "totally_unknown_food_123")
        self.assertEqual(result["gi_value"], 55)


if __name__ == "__main__":
    unittest.main()
