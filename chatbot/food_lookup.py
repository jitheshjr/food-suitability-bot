import pandas as pd
import os
from difflib import SequenceMatcher

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
FOODS_CSV = os.path.join(BASE_DIR, "..", "data", "processed", "foods_with_gi.csv")

print("Loading food database...")
_foods_df = pd.read_csv(FOODS_CSV)
_foods_df['food_name_lower'] = _foods_df['food_name'].str.lower().str.strip()
print(f"Food database loaded — {len(_foods_df)} items.")

DEFAULT_NUTRITION = {
    'calories':    150,
    'protein_g':   3.0,
    'fat_g':       5.0,
    'carbs_g':     20.0,
    'sugar_g':     8.0,
    'fiber_g':     1.0,
    'sodium_mg':   100,
    'gi_value':    55,
    'gi_category': 'medium',
    'found':       False,
}

SKIP_WORDS = {
    'salty', 'sweet', 'spicy', 'hot', 'cold', 'fresh',
    'fried', 'baked', 'grilled', 'raw', 'plain', 'lite',
    'light', 'low', 'fat', 'free', 'sugar', 'diet',
}


def lookup_food(food_name: str) -> dict:
    if not food_name:
        return DEFAULT_NUTRITION.copy()

    query = food_name.lower().strip()

    # 1. Exact substring match
    mask = _foods_df['food_name_lower'].str.contains(query, na=False, regex=False)
    matches = _foods_df[mask]
    if not matches.empty:
        return _row_to_dict(matches.iloc[0])

    # 2. Word-level match — skip descriptor adjectives
    words = [w for w in query.split() if len(w) >= 4 and w not in SKIP_WORDS]
    for word in words:
        mask = _foods_df['food_name_lower'].str.contains(word, na=False, regex=False)
        matches = _foods_df[mask]
        if not matches.empty:
            return _row_to_dict(matches.iloc[0])

    # 3. Fuzzy match fallback
    best_score = 0
    best_row   = None
    for _, row in _foods_df.iterrows():
        score = SequenceMatcher(None, query, row['food_name_lower']).ratio()
        if score > best_score:
            best_score = score
            best_row   = row

    if best_score > 0.5 and best_row is not None:
        return _row_to_dict(best_row)

    # 4. Nothing found — return defaults with flag
    print(f"  Food not found: '{food_name}' — using defaults")
    result = DEFAULT_NUTRITION.copy()
    result['food_name'] = food_name
    return result


def _row_to_dict(row) -> dict:
    return {
        'food_name':   str(row.get('food_name', 'unknown')),
        'calories':    float(row.get('calories',   0)),
        'protein_g':   float(row.get('protein_g',  0)),
        'fat_g':       float(row.get('fat_g',      0)),
        'carbs_g':     float(row.get('carbs_g',    0)),
        'sugar_g':     float(row.get('sugar_g',    0)),
        'fiber_g':     float(row.get('fiber_g',    0)),
        'sodium_mg':   float(row.get('sodium_mg',  0)),
        'gi_value':    float(row.get('gi_value',   55)),
        'gi_category': str(row.get('gi_category',  'medium')),
        'found':       True,
    }


if __name__ == "__main__":
    tests = ['ice cream', 'white rice', 'banana', 'chicken', 'chips', 'xyz123']
    for food in tests:
        result = lookup_food(food)
        print(f"{food:<15} → sugar={result['sugar_g']}g  "
              f"sodium={result['sodium_mg']}mg  "
              f"gi={result['gi_value']}  "
              f"found={result.get('found', False)}")