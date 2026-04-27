import re 
import json 
import pandas as pd 
import numpy as np
import os

from groq import Groq
from dotenv import load_dotenv
load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

food          = pd.read_csv("raw/food.csv")
food_nutrient = pd.read_csv("raw/food_nutrient.csv")
nutrient      = pd.read_csv("raw/nutrient.csv")

# LLM Function

cache = {}

def extract_food_name_llm(description:str) -> str | None:
    if description in cache:
        return cache[description]
    
    try:
        prompt = f"""
                    Extract a clean, specific food name from this description.
                    Rules:
                    - For restaurant items, skip the restaurant/cuisine type prefix (e.g. "restaurant, mexican,")
                    - Remove cooking/preparation words (frozen, microwaved, dry, etc.)
                    - Remove storage terms
                    - Keep the main food identity
                    - Reorder words naturally if needed
                    - Keep brand only if essential
                    Return JSON:
                    {{"food_name": "..."}}
                    Text: {description}
                """
        
        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'```json|```', '', raw).strip()

        parsed = json.loads(raw)
        name = parsed.get("food_name")

        cache[description] = name
        return name

    except Exception as e:
        print(f"Error processing description: {description} - {e}")
        return None
    

# Nutrient Mapping

NUTRIENT_MAP = {
    1008: 'calories',
    1003: 'protein_g',
    1004: 'fat_g',
    1005: 'carbs_g',
    2000: 'sugar_g',
    1079: 'fiber_g',
    1093: 'sodium_mg',
    1092: 'potassium_mg',    
    1091: 'phosphorus_mg',   
}

# Filtering Nutrients

filtered = food_nutrient[food_nutrient['nutrient_id'].isin(NUTRIENT_MAP.keys())].copy()
filtered['nutrient_name'] = filtered['nutrient_id'].map(NUTRIENT_MAP)

# Pivoting to wide format

pivoted = filtered.pivot_table(
    index='fdc_id',
    columns='nutrient_name',
    values='amount',
    aggfunc='first'
).reset_index()

# Merging with food descriptions

merged = pivoted.merge(food[['fdc_id', 'description']], on='fdc_id')

# CLEANING

# Drop rows with too many missing nutrient values
merged = merged.dropna(thresh=5)

merged[['sugar_g', 'fiber_g']] = merged[['sugar_g', 'fiber_g']].fillna(0)

merged['description'] = merged['description'].str.lower().str.strip()

# Parser

def parse_food_name(raw_name: str) -> dict:
    
    name = raw_name.lower().strip()
    result = {
        'canonical_name':    None,
        'food_category':     'other',
    }

    parts = [p.strip() for p in name.split(',') if p.strip()]
    canonical = parts[0] if parts else name
    
    result['canonical_name'] = canonical
    return result

BAD_NAMES = {'restaurant', 'branded', 'other', 'unknown', 'food'}

def get_canonical_name(row):
    parsed = parse_food_name(row['description'])
    name = parsed['canonical_name']

    # Trigger LLM if name is missing, too short, OR a known bad placeholder
    if not name or len(name) < 3 or name.strip().lower() in BAD_NAMES:
        llm_name = extract_food_name_llm(row['description'])
        if llm_name:
            parsed['canonical_name'] = llm_name

    return pd.Series(parsed)

parsed = merged.apply(get_canonical_name, axis=1)
merged = pd.concat([merged, parsed], axis=1)

merged['food_name'] = merged['canonical_name']
final_cols = [
    'fdc_id', 'food_name', 'description', 'calories', 'protein_g', 'fat_g', 'carbs_g',
    'sugar_g', 'fiber_g', 'sodium_mg', 'potassium_mg', 'phosphorus_mg',
]

final_cols = [c for c in final_cols if c in merged.columns]

merged = merged[final_cols]

print(f"\nFinal shape: {merged.shape}")
print("\nSample:") 
print(merged[['food_name', 'description']].head(10))
merged.to_csv("processed/foods_clean.csv", index=False)
print("\n✅ Saved → processed/foods_clean.csv")