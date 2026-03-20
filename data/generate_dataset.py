import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

foods = pd.read_csv("processed/foods_with_gi.csv")

CONDITIONS = ['diabetes', 'hypertension', 'kidney_disease', 'healthy']

def compute_label(age, condition, cal, protein, fat, carbs, sugar, fiber, sodium, gi):
    score = 0

    if condition == 'diabetes':
        if gi > 70:        score += 3
        elif gi > 55:      score += 1
        if sugar > 15:     score += 2
        elif sugar > 8:    score += 1
        if fiber > 5:      score -= 1  # fiber is protective

    elif condition == 'hypertension':
        if sodium > 600:   score += 3
        elif sodium > 300: score += 1
        if fat > 20:       score += 1
        if fat > 30:       score += 1

    elif condition == 'kidney_disease':
        if sodium > 400:   score += 2
        if protein > 20:   score += 2
        elif protein > 10: score += 1
        if carbs > 50:     score += 1

    elif condition == 'healthy':
        if cal > 600:      score += 1
        if sodium > 800:   score += 1
        if sugar > 30:     score += 1

    # Age modifier — older patients are more sensitive
    if age > 60:
        score += 1
    elif age > 45:
        score += 0.5

    # Final label
    if score >= 3:   return 'avoid'
    elif score >= 1: return 'moderate'
    else:            return 'safe'

print("Generating synthetic patient-food dataset...")
rows = []

for _ in range(8000):
    # Random patient profile
    age = random.randint(18, 80)
    condition = random.choice(CONDITIONS)

    # Random food from dataset
    food_row = foods.sample(1).iloc[0]

    label = compute_label(
        age=age,
        condition=condition,
        cal=food_row.get('calories', 0),
        protein=food_row.get('protein_g', 0),
        fat=food_row.get('fat_g', 0),
        carbs=food_row.get('carbs_g', 0),
        sugar=food_row.get('sugar_g', 0),
        fiber=food_row.get('fiber_g', 0),
        sodium=food_row.get('sodium_mg', 0),
        gi=food_row.get('gi_value', 50),
    )

    rows.append({
        'age': age,
        'condition': condition,
        'food_name': food_row['food_name'],
        'calories': food_row.get('calories', 0),
        'protein_g': food_row.get('protein_g', 0),
        'fat_g': food_row.get('fat_g', 0),
        'carbs_g': food_row.get('carbs_g', 0),
        'sugar_g': food_row.get('sugar_g', 0),
        'fiber_g': food_row.get('fiber_g', 0),
        'sodium_mg': food_row.get('sodium_mg', 0),
        'gi_value': food_row.get('gi_value', 50),
        'label': label,
    })

df = pd.DataFrame(rows)

# Check label distribution
print("\nLabel distribution:")
print(df['label'].value_counts())
print("\nCondition distribution:")
print(df['condition'].value_counts())
print(f"\nTotal rows: {len(df)}")

df.to_csv("processed/labeled_dataset.csv", index=False)
print("\nSaved to processed/labeled_dataset.csv")
