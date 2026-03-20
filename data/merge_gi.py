import pandas as pd
from difflib import SequenceMatcher

foods = pd.read_csv("processed/foods_clean.csv")
gi = pd.read_csv("raw/glycemic_index.csv")

def best_gi_match(food_name, gi_df, threshold=0.5):
    food_name = food_name.lower()
    best_score = 0
    best_gi = 50  # default medium if no match
    best_cat = 'medium'
    for _, row in gi_df.iterrows():
        keyword = row['food_keyword'].lower()
        # Check if keyword appears in food name
        if keyword in food_name:
            return row['gi_value'], row['gi_category']
        # Fuzzy match fallback
        score = SequenceMatcher(None, food_name, keyword).ratio()
        if score > best_score and score > threshold:
            best_score = score
            best_gi = row['gi_value']
            best_cat = row['gi_category']
    return best_gi, best_cat

print("Matching GI values to foods...")
foods[['gi_value', 'gi_category']] = foods['food_name'].apply(
    lambda x: pd.Series(best_gi_match(x, gi))
)

print(foods[['food_name', 'gi_value', 'gi_category']].head(20))
foods.to_csv("processed/foods_with_gi.csv", index=False)
print(f"Saved {len(foods)} foods with GI values")
