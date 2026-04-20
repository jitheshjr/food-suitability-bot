import unittest

from chatbot.fusion import build_final_response


class FusionTests(unittest.TestCase):
    def test_final_response_includes_deduplicated_sources(self):
        text = build_final_response(
            ml_label="moderate",
            ml_confidence=82.4,
            shap_reasons=[("sugar content", 0.8), ("fiber content", -0.2)],
            food_name="ice cream",
            condition="diabetes",
            rag_results=[
                {"source": "diabetes_diet_guidelines.txt", "text": "a", "score": 0.9},
                {"source": "who_healthy_diet.pdf", "text": "b", "score": 0.8},
                {"source": "diabetes_diet_guidelines.txt", "text": "c", "score": 0.7},
            ],
            llm_explanation="This food is high in sugar and should be limited.",
        )

        self.assertIn("Recommendation: Ice Cream is better as an occasional choice for someone with diabetes.", text)
        self.assertIn("Evidence: diabetes_diet_guidelines.txt; who_healthy_diet.pdf", text)
        self.assertEqual(text.count("diabetes_diet_guidelines.txt"), 1)

    def test_final_response_omits_sources_when_no_rag_results(self):
        text = build_final_response(
            ml_label="safe",
            ml_confidence=71.0,
            shap_reasons=[],
            food_name="banana",
            condition="healthy",
            rag_results=[],
            llm_explanation="This food appears generally suitable.",
        )

        self.assertIn(
            "Recommendation: Banana is generally a reasonable choice for someone without a known medical condition.",
            text,
        )
        self.assertNotIn("Evidence:", text)


if __name__ == "__main__":
    unittest.main()
