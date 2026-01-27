import pandas as pd

df = pd.read_csv('data/training/Assessment_AI_Training_Data.csv')

metadata = {'plan_id', 'user_id', 'user_name', 'workspace_id', 
            'workspace_name', 'email_id', 'hired_workers', 'Unique identifier', 
            'address', 'county', 'plan_year', 'country_code', 'locale', 'state', 
            'zipcode', 'longitude', 'latitude', 'crop_name', 'area', 
            'additional_farm_area', 'irrigation', 'Partner', 'sub_partner', 
            'plan_created', 'plan_modified', 'plan_deleted', 'Submitted', 
            'unit_type', 'yield_unit', 'nitrogen_unit', 'essential_percentage',
            'intermediate_percentage', 'advanced_percentage', 'sai_score'}

indicators = [c for c in df.columns if c not in metadata and not c.startswith('Carbon') and 'Reason' not in c]

print("=" * 80)
print("TRAINING DATA QUALITY ANALYSIS")
print("=" * 80)
print(f"Total indicators: {len(indicators)}")
print(f"Total training samples: {len(df)}")
print()

trained = []
low_data = []

for ind in indicators:
    non_null = df[ind].notna().sum()
    unique = df[ind].nunique()
    
    if non_null >= 10 and unique >= 2:
        trained.append(ind)
    else:
        low_data.append((ind, non_null, unique))

print(f"Indicators WITH sufficient training data: {len(trained)}")
print(f"Indicators WITH insufficient data: {len(low_data)}")
print(f"Current Coverage: {len(trained)/len(indicators)*100:.1f}%")
print()

print("=" * 80)
print("WHY ONLY 4/38 PREDICTIONS?")
print("=" * 80)
print(f"Out of {len(indicators)} total indicators:")
print(f"  • Only {len(trained)} have enough data to train reliable models")
print(f"  • {len(low_data)} indicators lack sufficient training data")
print()
print("The model can ONLY predict indicators it was trained on.")
print("If your 38 questions include many indicators with low data,")
print("the model cannot predict them accurately.")
print()

print("=" * 80)
print("INDICATORS NEEDING MORE DATA (Top 30):")
print("=" * 80)
for ind, cnt, uniq in sorted(low_data, key=lambda x: x[1], reverse=True)[:30]:
    print(f"  {ind:<50} {cnt:3d} samples, {uniq:2d} unique")

print()
print("=" * 80)
print("SOLUTION:")
print("=" * 80)
print("1. Add MORE diverse training data (target: 500+ assessments)")
print("2. Ensure data covers ALL question types, not just common ones")
print("3. Import more CSV files with complete assessment data")
print("4. Retrain the model after adding data")
print()
print(f"Current: {len(trained)}/{len(indicators)} = {len(trained)/len(indicators)*100:.1f}% coverage")
print(f"Target:  250/{len(indicators)} = 80%+ coverage")
