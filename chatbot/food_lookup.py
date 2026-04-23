import pandas as pd
import numpy as np
import os
from difflib import SequenceMatcher

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
FOODS_CSV = os.path.join(BASE_DIR, "..", "data", "processed", "foods_with_gi.csv")

print("Loading food database...")
_foods_df = pd.read_csv(FOODS_CSV)

# Build lowercase index on BOTH canonical_name and food_name.
# canonical_name (added by preprocess_foods.py) is the clean short name
# e.g. "lamb rib chop" instead of the 40-word raw USDA string.
# We match against canonical_name first — it gives far better results.
# food_name is kept as a fallback for rows that may not have canonical_name yet.
if 'canonical_name' in _foods_df.columns:
    _foods_df['canonical_lower'] = _foods_df['canonical_name'].str.lower().str.strip()
else:
    # Old foods_with_gi.csv without canonical_name — degrade gracefully
    print("  WARNING: 'canonical_name' column not found. Run updated preprocess_foods.py.")
    _foods_df['canonical_lower'] = _foods_df['food_name'].str.lower().str.strip()

_foods_df['food_name_lower'] = _foods_df['food_name'].str.lower().str.strip()

print(f"Food database loaded — {len(_foods_df)} items.")

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# Used when a food cannot be found at all.
# Potassium and phosphorus defaults are conservative (moderate values) so
# the model doesn't assume the food is kidney-safe when data is missing.
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_NUTRITION = {
    'food_name':         'unknown',
    'canonical_name':    'unknown',
    'calories':          150,
    'protein_g':         3.0,
    'fat_g':             5.0,
    'carbs_g':           20.0,
    'sugar_g':           8.0,
    'fiber_g':           1.0,
    'sodium_mg':         200,
    'potassium_mg':      300,     # conservative default — not zero
    'phosphorus_mg':     150,     # conservative default — not zero
    'gi_value':          55,
    'gi_category':       'medium',
    'gi_match_type':     'unknown',
    'food_category':     'other',
    'preparation_state': None,
    'cooking_method':    None,
    'fat_descriptor':    None,
    'bone_status':       None,
    'enrichment':        None,
    'special_population':None,
    'source':            None,
    'brand':             None,
    'cuisine':           None,
    'is_composite_dish': False,
    'found':             False,
}

# Words to skip during word-level matching — these are descriptors, not food names
SKIP_WORDS = {
    'salty', 'sweet', 'spicy', 'hot', 'cold', 'fresh',
    'fried', 'baked', 'grilled', 'raw', 'plain', 'lite',
    'light', 'low', 'fat', 'free', 'sugar', 'diet', 'with',
    'without', 'and', 'the', 'some', 'little',
}

# Fuzzy match threshold — same as merge_gi.py (0.65)
# Lower than this causes wrong matches (e.g. "rice" → "spice")
FUZZY_THRESHOLD = 0.55   # slightly looser than GI matching since user queries are short


def lookup_food(food_name: str) -> dict:
    """
    Look up nutrition and metadata for a food name.

    Match priority:
      1. Exact substring match on canonical_name  (best — short clean names)
      2. Exact substring match on food_name       (fallback for uncleaned rows)
      3. Word-level match on canonical_name
      4. Word-level match on food_name
      5. Fuzzy match on canonical_name
      6. Return DEFAULT_NUTRITION with found=False
    """
    if not food_name:
        return DEFAULT_NUTRITION.copy()

    query = food_name.lower().strip()

    # ── 1. Exact match on canonical_name ──────────────────────────
    mask = _foods_df['canonical_lower'].str.contains(query, na=False, regex=False)
    matches = _foods_df[mask]
    if not matches.empty:
        return _row_to_dict(matches.iloc[0])

    # ── 2. Exact match on full food_name ──────────────────────────
    mask = _foods_df['food_name_lower'].str.contains(query, na=False, regex=False)
    matches = _foods_df[mask]
    if not matches.empty:
        return _row_to_dict(matches.iloc[0])

    # ── 3. Word-level match on canonical_name ─────────────────────
    words = [w for w in query.split() if len(w) >= 4 and w not in SKIP_WORDS]
    for word in words:
        mask = _foods_df['canonical_lower'].str.contains(word, na=False, regex=False)
        matches = _foods_df[mask]
        if not matches.empty:
            return _row_to_dict(matches.iloc[0])

    # ── 4. Word-level match on food_name (broader) ────────────────
    for word in words:
        mask = _foods_df['food_name_lower'].str.contains(word, na=False, regex=False)
        matches = _foods_df[mask]
        if not matches.empty:
            return _row_to_dict(matches.iloc[0])

    # ── 5. Fuzzy match on canonical_name ──────────────────────────
    # Only scan canonical_lower — much faster than scanning full raw names
    best_score = 0
    best_row   = None
    for _, row in _foods_df.iterrows():
        score = SequenceMatcher(None, query, row['canonical_lower']).ratio()
        if score > best_score:
            best_score = score
            best_row   = row

    if best_score >= FUZZY_THRESHOLD and best_row is not None:
        return _row_to_dict(best_row)

    # ── 6. Not found ──────────────────────────────────────────────
    print(f"  Food not found: '{food_name}' — using conservative defaults")
    result = DEFAULT_NUTRITION.copy()
    result['food_name']      = food_name
    result['canonical_name'] = food_name
    return result


def _safe_float(val, default=0.0) -> float:
    """Convert to float safely, returning default for None/NaN."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if (f != f) else f   # NaN check
    except (ValueError, TypeError):
        return default


def _safe_str(val, default=None):
    """Convert to str safely, returning default for None/NaN."""
    if val is None:
        return default
    if isinstance(val, float) and (val != val):   # NaN
        return default
    return str(val)


def _safe_bool(val, default=False) -> bool:
    if val is None:
        return default
    if isinstance(val, float) and (val != val):
        return default
    return bool(val)


def _row_to_dict(row) -> dict:
    """
    Convert a DataFrame row to the full nutrition dict that predict.py expects.
    All new fields from preprocess_foods.py are extracted here.
    """
    return {
        # Identity
        'food_name':         str(row.get('food_name', 'unknown')),
        'canonical_name':    str(row.get('canonical_name', row.get('food_name', 'unknown'))),

        # Core nutrients
        'calories':          _safe_float(row.get('calories'),     0),
        'protein_g':         _safe_float(row.get('protein_g'),    0),
        'fat_g':             _safe_float(row.get('fat_g'),        0),
        'carbs_g':           _safe_float(row.get('carbs_g'),      0),
        'sugar_g':           _safe_float(row.get('sugar_g'),      0),
        'fiber_g':           _safe_float(row.get('fiber_g'),      0),
        'sodium_mg':         _safe_float(row.get('sodium_mg'),    0),

        # NEW — critical for CKD scoring
        'potassium_mg':      _safe_float(row.get('potassium_mg'),  0),
        'phosphorus_mg':     _safe_float(row.get('phosphorus_mg'), 0),

        # GI
        'gi_value':          _safe_float(row.get('gi_value'),    55),
        'gi_category':       _safe_str(row.get('gi_category'),   'medium'),
        'gi_match_type':     _safe_str(row.get('gi_match_type'), 'unknown'),

        # NEW — food metadata from preprocess_foods.py
        'food_category':     _safe_str(row.get('food_category'),     'other'),
        'preparation_state': _safe_str(row.get('preparation_state'), None),
        'cooking_method':    _safe_str(row.get('cooking_method'),    None),
        'fat_descriptor':    _safe_str(row.get('fat_descriptor'),    None),
        'bone_status':       _safe_str(row.get('bone_status'),       None),
        'enrichment':        _safe_str(row.get('enrichment'),        None),
        'special_population':_safe_str(row.get('special_population'),None),
        'source':            _safe_str(row.get('source'),            None),
        'brand':             _safe_str(row.get('brand'),             None),
        'cuisine':           _safe_str(row.get('cuisine'),           None),
        'is_composite_dish': _safe_bool(row.get('is_composite_dish'), False),

        'found': True,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        'ice cream', 'white rice', 'banana', 'chicken pasta',
        'salmon', 'biryani', 'chips', 'spinach', 'xyz123'
    ]
    print(f"{'Food':<20} {'Found':<6} {'K(mg)':<8} {'P(mg)':<8} {'Na(mg)':<8} {'GI':<6} {'Category'}")
    print("-" * 75)
    for food in tests:
        r = lookup_food(food)
        print(
            f"{food:<20} {str(r['found']):<6} "
            f"{r['potassium_mg']:<8.0f} {r['phosphorus_mg']:<8.0f} "
            f"{r['sodium_mg']:<8.0f} {r['gi_value']:<6.0f} "
            f"{r['food_category']}"
        )