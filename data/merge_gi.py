import pandas as pd
from difflib import SequenceMatcher
import re

foods = pd.read_csv("processed/foods_clean.csv")
gi    = pd.read_csv("raw/glycemic_index.csv")

# Foods that biologically carry near-zero GI
# (proteins / pure fats produce no glucose spike)
NO_GI_KEYWORDS = [
    'meat', 'chicken', 'beef', 'pork', 'fish', 'egg',
    'lamb', 'turkey', 'shrimp', 'tuna', 'salmon', 'trout',
    'sardine', 'crab', 'lobster', 'venison', 'bison',
    'oil', 'butter', 'lard', 'ghee', 'mayonnaise',
    'cheese', 'cream', 'bacon', 'sausage', 'salami',
]

# Raised from 0.5 → 0.65 to avoid bad fuzzy matches
# (e.g. "pork" matching "corn" at lower threshold)
FUZZY_THRESHOLD = 0.65

def is_biological_zero(name: str) -> bool:
    for k in NO_GI_KEYWORDS:
        if re.search(rf'\b{re.escape(k)}\b', name):
            return True
    return False

def best_gi_match(canonical_name: str, gi_df: pd.DataFrame):
    """
    Match a canonical food name to a GI value.

    Match priority:
      1. Near-zero biological GI  → (0,    'low',     'biological_zero')
      2. Exact keyword-in-name    → (value, cat,       'exact')
      3. Fuzzy match >= 0.65      → (value, cat,       'fuzzy')
      4. No match                 → (None,  'unknown',  'unmatched')
         -> logged to gi_unmatched.csv for manual review
         -> NOT silently defaulted to 50 (which corrupted diabetes labels)
    """
    name_lower = str(canonical_name).lower().strip()

    if is_biological_zero(name_lower):
        return 0, 'low', 'biological_zero'

    # -- Priority 1: biological zero ------------------------------------------
    if any(k in name_lower for k in NO_GI_KEYWORDS):
        return 0, 'low', 'biological_zero'

    best_score = 0
    best_gi    = None
    best_cat   = 'unknown'
    best_type  = 'unmatched'

    for _, row in gi_df.iterrows():
        keyword = str(row['food_keyword']).lower().strip()

        # -- Priority 2: exact substring match --------------------------------
        if keyword in name_lower:
            return float(row['gi_value']), row['gi_category'], 'exact'

        # -- Priority 3: fuzzy fallback ---------------------------------------
        score = SequenceMatcher(None, name_lower, keyword).ratio()
        if score > best_score and score >= FUZZY_THRESHOLD:
            best_score = score
            best_gi    = float(row['gi_value'])
            best_cat   = row['gi_category']
            best_type  = 'fuzzy'

    return best_gi, best_cat, best_type


print("Matching GI values to canonical food names...")

# Guard: if canonical_name column is missing (old foods_clean.csv),
# fall back to food_name and warn loudly
if 'canonical_name' not in foods.columns:
    print("WARNING: 'canonical_name' column not found.")
    print("   Run updated preprocess_foods.py first to generate it.")
    print("   Falling back to 'food_name' — GI match quality will be degraded.")
    foods['canonical_name'] = foods['food_name']

results = foods['canonical_name'].apply(
    lambda x: pd.Series(
        best_gi_match(x, gi),
        index=['gi_value', 'gi_category', 'gi_match_type']
    )
)

foods = pd.concat([foods, results], axis=1)

# ──────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ──────────────────────────────────────────────────────────────────────────────

print("\n-- GI match summary -------------------------------------------")
print(foods['gi_match_type'].value_counts())

unmatched = foods[foods['gi_match_type'] == 'unmatched']
if len(unmatched) > 0:
    unmatched.to_csv("processed/gi_unmatched.csv", index=False)
    print(f"\nWARNING: {len(unmatched)} foods have no GI match.")
    print("   Saved -> processed/gi_unmatched.csv")
    print("   Add missing entries to raw/glycemic_index.csv and re-run.")
else:
    print("\nAll foods matched successfully.")

print("\n-- Sample: canonical_name -> GI --------------------------------")
print(foods[['food_name', 'canonical_name', 'gi_value',
             'gi_category', 'gi_match_type']].head(20).to_string())

foods.to_csv("processed/foods_with_gi.csv", index=False)
print(f"\nSaved {len(foods)} foods -> processed/foods_with_gi.csv")