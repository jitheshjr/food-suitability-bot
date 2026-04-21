import re
import pandas as pd
import numpy as np

print("---- Loading USDA data ----")
food          = pd.read_csv("raw/food.csv")
food_nutrient = pd.read_csv("raw/food_nutrient.csv")
nutrient      = pd.read_csv("raw/nutrient.csv")

# ──────────────────────────────────────────────────────────────────────────────
# NUTRIENT IDs  (USDA SR Legacy)
# Potassium (1092) and Phosphorus (1091) added — critical for CKD scoring
# ──────────────────────────────────────────────────────────────────────────────
NUTRIENT_MAP = {
    1008: 'calories',
    1003: 'protein_g',
    1004: 'fat_g',
    1005: 'carbs_g',
    2000: 'sugar_g',
    1079: 'fiber_g',
    1093: 'sodium_mg',
    1092: 'potassium_mg',    # NEW — critical for CKD
    1091: 'phosphorus_mg',   # NEW — critical for CKD
}

print("---- Filtering nutrients ----")
filtered = food_nutrient[food_nutrient['nutrient_id'].isin(NUTRIENT_MAP.keys())].copy()
filtered['nutrient_name'] = filtered['nutrient_id'].map(NUTRIENT_MAP)

print("Pivoting table...")
pivoted = filtered.pivot_table(
    index='fdc_id',
    columns='nutrient_name',
    values='amount',
    aggfunc='first'
).reset_index()

merged = pivoted.merge(food[['fdc_id', 'description']], on='fdc_id')
merged = merged.rename(columns={'description': 'food_name'})

# Drop rows with too many missing nutrient values
merged = merged.dropna(thresh=5)

# Keep unknown nutrients as NaN — do NOT fill with 0
# (unknown sodium ≠ zero sodium; would silently mark unsafe foods as safe)
# Only truly optional / additive nutrients get a 0 fill
merged[['sugar_g', 'fiber_g']] = merged[['sugar_g', 'fiber_g']].fillna(0)

# Lowercase + strip whitespace
merged['food_name'] = merged['food_name'].str.lower().str.strip()


# ──────────────────────────────────────────────────────────────────────────────
# FOOD NAME PARSER
# Problem: USDA names are verbose, inconsistent, and polluted with:
#   • preparation states  ("cooked, boiled, drained, without salt")
#   • fat/cut metadata    ("separable lean and fat, trimmed to 1/8\" fat")
#   • parenthetical aliases ("waxgourd, (chinese preserving melon)")
#   • brand names         ("applebee's, double crunch shrimp")
#   • age-specific tags   ("babyfood, fruit, pears, strained")
#   • restaurant tags     ("restaurant, chinese, kung pao chicken")
#
# Strategy: decompose each name into structured fields WITHOUT discarding
# any information. canonical_name is used for GI matching and display.
# All metadata is preserved in separate columns for model features.
# ──────────────────────────────────────────────────────────────────────────────

# ── Keyword lists ─────────────────────────────────────────────────────────────

PREP_STATES = [
    'raw', 'cooked', 'roasted', 'boiled', 'baked', 'fried', 'grilled',
    'steamed', 'broiled', 'stewed', 'braised', 'poached', 'smoked',
    'dried', 'dehydrated', 'frozen', 'canned', 'pickled', 'fermented',
    'strained', 'pureed', 'mashed', 'toasted', 'unheated', 'heated',
    'dry mix', 'dry', 'reconstituted', 'unprepared', 'prepared',
]

COOKING_METHODS = [
    'roasted', 'boiled', 'baked', 'fried', 'grilled', 'steamed',
    'broiled', 'stewed', 'braised', 'poached', 'smoked', 'toasted',
    'microwaved', 'deep-fried', 'pan-fried', 'stir-fried',
]

FAT_PATTERNS = [
    r'trimmed to \d+/?\d*["\s]*fat',
    r'separable lean and fat',
    r'separable lean only',
    r'lean and fat',
    r'lean only',
    r'\d+%\s*lean\s*/\s*\d+%\s*fat',
    r'extra lean',
    r'low.?fat',
    r'full.?fat',
    r'reduced.?fat',
    r'fat.?free',
    r'skim',
    r'whole milk',
    r'nonfat',
    r'light',
]

BONE_PATTERNS = [
    r'\bbone.?in\b',
    r'\bboneless\b',
    r'\bbone.?out\b',
]

ENRICHMENT_KEYWORDS = [
    'enriched', 'unenriched', 'fortified', 'unfortified',
    'bleached', 'unbleached',
]

SPECIAL_POP_KEYWORDS = {
    'infant':     ['babyfood', 'baby food', 'infant formula', 'infant'],
    'toddler':    ['toddler'],
    'school_age': ['school age'],
}

CUISINE_KEYWORDS = {
    'chinese':    ['chinese', 'kung pao', 'fried rice', 'chow mein', 'dim sum'],
    'mexican':    ['mexican', 'taco', 'burrito', 'enchilada', 'tamale'],
    'italian':    ['italian', 'pasta', 'pizza', 'risotto', 'lasagna'],
    'indian':     ['indian', 'curry', 'biryani', 'dal', 'tandoor'],
    'japanese':   ['japanese', 'sushi', 'ramen', 'tempura', 'miso'],
    'american':   ['american', 'burger', 'hot dog', 'bbq'],
    'fast_food':  ['mcdonald', 'burger king', 'kfc', 'subway', 'wendy'],
}

RESTAURANT_BRANDS = [
    "mcdonald's", "burger king", "kfc", "subway", "wendy's",
    "applebee's", "olive garden", "chili's", "denny's", "ihop",
    "starbucks", "dunkin", "pizza hut", "domino's", "papa john's",
    "taco bell", "chipotle", "panera", "outback", "red lobster",
]

FOOD_CATEGORIES = {
    'meat':       ['beef', 'pork', 'lamb', 'veal', 'venison', 'bison', 'goat',
                   'rabbit', 'boar', 'mutton'],
    'poultry':    ['chicken', 'turkey', 'duck', 'goose', 'quail', 'pheasant',
                   'ostrich', 'cornish'],
    'seafood':    ['fish', 'salmon', 'tuna', 'cod', 'tilapia', 'trout', 'bass',
                   'halibut', 'sardine', 'mackerel', 'herring', 'anchovy',
                   'shrimp', 'prawn', 'crab', 'lobster', 'clam', 'oyster',
                   'scallop', 'mussel', 'squid', 'octopus'],
    'dairy':      ['milk', 'cheese', 'yogurt', 'butter', 'cream', 'whey',
                   'ghee', 'kefir', 'cottage cheese'],
    'egg':        ['egg'],
    'grain':      ['wheat', 'rice', 'oat', 'barley', 'corn', 'maize', 'rye',
                   'millet', 'sorghum', 'quinoa', 'bread', 'pasta', 'noodle',
                   'cereal', 'flour', 'cracker', 'bagel', 'muffin', 'cake',
                   'cookie', 'pastry', 'tortilla', 'semolina'],
    'legume':     ['bean', 'lentil', 'pea', 'chickpea', 'soybean', 'tofu',
                   'tempeh', 'peanut', 'fava', 'broadbean'],
    'vegetable':  ['spinach', 'broccoli', 'carrot', 'tomato', 'potato',
                   'sweet potato', 'onion', 'garlic', 'celery', 'cabbage',
                   'lettuce', 'kale', 'zucchini', 'squash', 'pepper',
                   'cucumber', 'beet', 'radish', 'turnip', 'asparagus',
                   'artichoke', 'eggplant', 'mushroom', 'pumpkin',
                   'waxgourd', 'okra', 'leek', 'cauliflower'],
    'fruit':      ['apple', 'banana', 'orange', 'grape', 'strawberry',
                   'blueberry', 'raspberry', 'mango', 'pineapple', 'peach',
                   'pear', 'plum', 'cherry', 'watermelon', 'melon', 'kiwi',
                   'lemon', 'lime', 'grapefruit', 'avocado', 'fig', 'date'],
    'nut_seed':   ['almond', 'walnut', 'pecan', 'cashew', 'pistachio',
                   'macadamia', 'hazelnut', 'coconut', 'sunflower seed',
                   'pumpkin seed', 'sesame', 'flaxseed', 'chia'],
    'fat_oil':    ['oil', 'lard', 'shortening', 'margarine', 'tallow'],
    'beverage':   ['juice', 'soda', 'coffee', 'tea', 'wine', 'beer', 'spirit',
                   'smoothie', 'shake', 'drink', 'water', 'broth', 'stock'],
    'condiment':  ['sauce', 'ketchup', 'mustard', 'mayonnaise', 'dressing',
                   'vinegar', 'soy sauce', 'hot sauce', 'relish', 'jam',
                   'jelly', 'syrup', 'honey', 'sugar', 'salt', 'spice'],
    'soup':       ['soup', 'stew', 'chowder', 'broth', 'ramen', 'bisque'],
    'snack':      ['chip', 'crisp', 'popcorn', 'pretzel', 'granola bar',
                   'energy bar', 'trail mix'],
    'dessert':    ['ice cream', 'gelato', 'pudding', 'brownie', 'pie',
                   'cheesecake', 'doughnut', 'candy', 'chocolate'],
    'processed':  ['hot dog', 'sausage', 'bacon', 'salami', 'pepperoni',
                   'ham', 'bologna', 'spam', 'deli meat'],
}


def parse_food_name(raw_name: str) -> dict:
    """
    Decompose a raw USDA food description into structured fields.

    Returns a dict with:
      canonical_name   — clean short name for display and GI matching
      aliases          — parenthetical / alternative names
      food_category    — broad category (meat, grain, vegetable, …)
      preparation_state — raw / cooked / canned / frozen / …
      cooking_method   — roasted / boiled / fried / …
      fat_descriptor   — lean, lean and fat, % lean/fat, trimmed to X fat
      bone_status      — bone_in / boneless / None
      enrichment       — enriched / unenriched / bleached / None
      special_population — infant / toddler / None
      source           — restaurant / home / None
      brand            — brand name if present / None
      cuisine          — chinese / mexican / … / None
      is_composite_dish — True for multi-ingredient dishes (kung pao, biryani)
    """
    name = raw_name.lower().strip()
    result = {
        'canonical_name':    None,
        'aliases':           None,
        'food_category':     'other',
        'preparation_state': None,
        'cooking_method':    None,
        'fat_descriptor':    None,
        'bone_status':       None,
        'enrichment':        None,
        'special_population': None,
        'source':            None,
        'brand':             None,
        'cuisine':           None,
        'is_composite_dish': False,
    }

    working = name

    # ── 1. Extract aliases from parentheses ────────────────────────
    alias_matches = re.findall(r'\(([^)]+)\)', working)
    if alias_matches:
        result['aliases'] = '; '.join(alias_matches)
    working = re.sub(r'\([^)]*\)', '', working).strip()

    # ── 2. Special population ──────────────────────────────────────
    for pop, keywords in SPECIAL_POP_KEYWORDS.items():
        if any(k in working for k in keywords):
            result['special_population'] = pop
            break

    # ── 3. Restaurant / brand ──────────────────────────────────────
    if working.startswith('restaurant,'):
        result['source'] = 'restaurant'
        working = re.sub(r'^restaurant,\s*', '', working).strip()

    for brand in RESTAURANT_BRANDS:
        if brand in working:
            result['brand']  = brand
            result['source'] = 'restaurant'
            working = working.replace(brand, '').strip().strip(',').strip()
            break

    # ── 4. Cuisine ────────────────────────────────────────────────
    for cuisine, keywords in CUISINE_KEYWORDS.items():
        if any(k in working for k in keywords):
            result['cuisine'] = cuisine
            break

    # ── 5. Composite dish detection ───────────────────────────────
    # Short names with no commas are likely composite dishes
    # (e.g. "kung pao chicken", "ramen noodle, chicken flavor")
    COMPOSITE_SIGNALS = [
        'kung pao', 'fried rice', 'chicken soup', 'beef stew', 'biryani',
        'curry', 'lasagna', 'casserole', 'stir.?fry', 'burrito', 'taco',
        'pizza', 'sandwich', 'burger', 'pot pie', 'chowder',
    ]
    if any(re.search(sig, working) for sig in COMPOSITE_SIGNALS):
        result['is_composite_dish'] = True

    # ── 6. Enrichment ─────────────────────────────────────────────
    for kw in ENRICHMENT_KEYWORDS:
        if kw in working:
            result['enrichment'] = kw
            working = working.replace(kw, '').strip().strip(',').strip()
            break

    # ── 7. Bone status ────────────────────────────────────────────
    for pat in BONE_PATTERNS:
        m = re.search(pat, working)
        if m:
            result['bone_status'] = m.group().replace('-', '_').replace(' ', '_')
            working = re.sub(pat, '', working).strip().strip(',').strip()
            break

    # ── 8. Fat descriptor ─────────────────────────────────────────
    for pat in FAT_PATTERNS:
        m = re.search(pat, working)
        if m:
            result['fat_descriptor'] = m.group().strip()
            working = re.sub(pat, '', working).strip().strip(',').strip()
            break

    # ── 9. Cooking method (before prep state) ─────────────────────
    for method in COOKING_METHODS:
        if re.search(r'\b' + method + r'\b', working):
            result['cooking_method'] = method
            working = re.sub(r'\b' + method + r'\b', '', working)
            break

    # ── 10. Preparation state ─────────────────────────────────────
    for state in PREP_STATES:
        if re.search(r'\b' + re.escape(state) + r'\b', working):
            result['preparation_state'] = state
            working = re.sub(r'\b' + re.escape(state) + r'\b', '', working)
            break

    # ── 11. Food category ─────────────────────────────────────────
    for category, keywords in FOOD_CATEGORIES.items():
        if any(kw in working for kw in keywords):
            result['food_category'] = category
            break

    # ── 12. Build canonical name ──────────────────────────────────
    # Strip leftover punctuation, extra commas, whitespace
    canonical = re.sub(r',\s*,', ',', working)        # double commas
    canonical = re.sub(r'\s+', ' ', canonical)         # multiple spaces
    canonical = canonical.strip().strip(',').strip()

    # Take the first meaningful segment (before first comma)
    # for composite dishes keep the whole name
    if not result['is_composite_dish'] and ',' in canonical:
        parts = [p.strip() for p in canonical.split(',') if p.strip()]
        # The first non-empty part is usually the core food
        canonical = parts[0] if parts else canonical

    # Final cleanup
    canonical = re.sub(r'\s+', ' ', canonical).strip()
    result['canonical_name'] = canonical if canonical else raw_name.lower().strip()

    return result


# ──────────────────────────────────────────────────────────────────────────────
# APPLY PARSER
# ──────────────────────────────────────────────────────────────────────────────

print("\nParsing food names...")
parsed = merged['food_name'].apply(lambda n: pd.Series(parse_food_name(n)))
merged = pd.concat([merged, parsed], axis=1)

# ──────────────────────────────────────────────────────────────────────────────
# COLUMN ORDER & SAVE
# ──────────────────────────────────────────────────────────────────────────────

base_cols = [
    'fdc_id',
    'food_name',          # original USDA name — kept for traceability
    'canonical_name',     # clean short name — used for GI matching & display
    'aliases',
    'food_category',
    'preparation_state',
    'cooking_method',
    'fat_descriptor',
    'bone_status',
    'enrichment',
    'special_population',
    'source',
    'brand',
    'cuisine',
    'is_composite_dish',
]

nutrient_cols = [
    'calories', 'protein_g', 'fat_g', 'carbs_g',
    'sugar_g', 'fiber_g', 'sodium_mg', 'potassium_mg', 'phosphorus_mg',
]

# Only keep columns that actually exist after pivot
nutrient_cols = [c for c in nutrient_cols if c in merged.columns]
final_cols    = [c for c in base_cols if c in merged.columns] + nutrient_cols

merged = merged[final_cols]

print(f"\nFinal shape: {merged.shape}")
print(f"Columns: {list(merged.columns)}")
print("\nSample parsed names:")
print(merged[['food_name', 'canonical_name', 'food_category',
              'preparation_state', 'fat_descriptor']].head(20).to_string())

# ── Diagnostics ───────────────────────────────────────────────────────────────
print("\n── Category distribution ────────────────────────────────────")
print(merged['food_category'].value_counts())

print("\n── Preparation state distribution ───────────────────────────")
print(merged['preparation_state'].value_counts().head(15))

print("\n── Special population count ─────────────────────────────────")
print(merged['special_population'].value_counts())

print("\n── Source distribution ──────────────────────────────────────")
print(merged['source'].value_counts())

merged.to_csv("processed/foods_clean.csv", index=False)
print("\n✅ Saved → processed/foods_clean.csv")