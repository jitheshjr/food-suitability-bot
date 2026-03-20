import pandas as pd

print("Loading USDA data...")
food = pd.read_csv("raw/food.csv")
food_nutrient = pd.read_csv("raw/food_nutrient.csv")
nutrient = pd.read_csv("raw/nutrient.csv")

# Nutrient IDs we care about (from USDA SR Legacy)
NUTRIENT_MAP = {
    1008: 'calories',
    1003: 'protein_g',
    1004: 'fat_g',
    1005: 'carbs_g',
    2000: 'sugar_g',
    1079: 'fiber_g',
    1093: 'sodium_mg',
}

print("Filtering nutrients...")
filtered = food_nutrient[food_nutrient['nutrient_id'].isin(NUTRIENT_MAP.keys())].copy()
filtered['nutrient_name'] = filtered['nutrient_id'].map(NUTRIENT_MAP)

# Pivot so each food is one row with nutrient columns
print("Pivoting table...")
pivoted = filtered.pivot_table(
    index='fdc_id',
    columns='nutrient_name',
    values='amount',
    aggfunc='first'
).reset_index()

# Merge with food names
merged = pivoted.merge(food[['fdc_id', 'description']], on='fdc_id')
merged = merged.rename(columns={'description': 'food_name'})

# Drop rows with too many missing values
merged = merged.dropna(thresh=5)

# Fill remaining NaN with 0
merged = merged.fillna(0)

# Clean food names to lowercase
merged['food_name'] = merged['food_name'].str.lower().str.strip()

# Keep only relevant columns in clean order
cols = ['fdc_id', 'food_name', 'calories', 'protein_g', 'fat_g',
        'carbs_g', 'sugar_g', 'fiber_g', 'sodium_mg']
# Only keep cols that exist
cols = [c for c in cols if c in merged.columns]
merged = merged[cols]

print(f"Final shape: {merged.shape}")
print(merged.head())

merged.to_csv("processed/foods_clean.csv", index=False)
print("Saved to processed/foods_clean.csv")
