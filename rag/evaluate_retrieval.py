import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.retriever import retrieve_context

# Test cases: query → expected keyword in retrieved text
TEST_CASES = [
    ("diabetic ice cream sugar",        "sugar",      "diabetes"),
    ("glycemic index high blood sugar",  "glycemic",   "diabetes"),
    ("sodium hypertension blood pressure","sodium",    "hypertension"),
    ("DASH diet fruits vegetables",      "DASH",       "hypertension"),
    ("protein kidney disease restriction","protein",   "kidney"),
    ("potassium kidney disease avoid",   "potassium",  "kidney"),
    ("fiber diabetes blood glucose",     "fiber",      "diabetes"),
    ("saturated fat heart blood pressure","fat",       "hypertension"),
]

print("RAG RETRIEVAL EVALUATION")
print("="*60)

passed = 0
for query, keyword, expected_source_hint in TEST_CASES:
    results = retrieve_context(query, k=3)
    all_text = " ".join([r['text'].lower() for r in results])
    top_score = results[0]['score'] if results else 0

    hit = keyword.lower() in all_text
    status = "PASS" if hit else "FAIL"
    if hit:
        passed += 1

    print(f"[{status}] Query: '{query[:40]}'")
    print(f"       Keyword '{keyword}' found: {hit} | Top score: {top_score:.3f}")

print(f"\nPrecision@3: {passed}/{len(TEST_CASES)} = {passed/len(TEST_CASES)*100:.1f}%")
print("\nThis score goes into your thesis evaluation section.")
