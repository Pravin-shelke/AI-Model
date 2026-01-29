import pickle
import pandas as pd
import numpy as np

print("=" * 80)
print("🔍 MODEL COVERAGE ANALYSIS")
print("=" * 80)
print()

# Load models
with open('models/assessment_ai_models.pkl', 'rb') as f:
    models = pickle.load(f)

print(f"Total trained models: {len(models)}")
print()

# Load training data
df = pd.read_csv('data/training/Assessment_AI_Training_Data.csv')

# Get all indicator columns (exclude metadata columns)
metadata_cols = ['plan_id', 'user_id', 'user_name', 'workspace_id', 
                 'workspace_name', 'email_id', 'hired_workers', 'Unique identifier', 
                 'address', 'county', 'plan_year', 'country_code', 'locale', 'state', 
                 'zipcode', 'longitude', 'latitude', 'crop_name', 'area', 
                 'additional_farm_area', 'irrigation', 'Partner', 'sub_partner', 
                 'plan_created', 'plan_modified', 'plan_deleted', 'Submitted', 
                 'unit_type', 'yield_unit', 'nitrogen_unit', 'essential_percentage',
                 'intermediate_percentage', 'advanced_percentage', 'sai_score',
                 'Carbon-score - Total Emissions', 'Carbon-score - Residue Management',
                 'Carbon-score - Fertilizer Production', 'Carbon-score - Applied Fertilizer',
                 'Carbon-score - Crop Protection', 'Carbon-score - Carbon Stock Changes',
                 'Carbon-score - Energy Use (field)', 'Carbon-score - Delivery to point of sale',
                 'Carbon-score - Irrigation', 'Reason for failure of Carbon-score generation']

indicator_cols = [col for col in df.columns if col not in metadata_cols]

print(f" Indicator Analysis:")
print(f"   Total indicator columns: {len(indicator_cols)}")
print(f"   Trained models: {len(models)}")
print(f"   Coverage: {len(models)/len(indicator_cols)*100:.1f}%")
print()

# Analyze data quality per indicator
print(" Data Quality Breakdown:")
print()

indicators_with_data = []
indicators_trained = []
indicators_skipped = []

for col in indicator_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        coverage = non_null / len(df) * 100
        
        has_model = col in models
        
        if non_null >= 10:  # Threshold for training
            indicators_with_data.append((col, non_null, coverage, has_model))
            if has_model:
                indicators_trained.append(col)
            else:
                indicators_skipped.append(col)

print(f" Indicators with sufficient data (≥10 samples): {len(indicators_with_data)}")
print(f" Indicators with trained models: {len(indicators_trained)}")
print(f" Indicators with data but NO model: {len(indicators_skipped)}")
print()

# Show indicators that should be trained but aren't
if indicators_skipped:
    print(f"  PROBLEM: {len(indicators_skipped)} indicators have data but NO trained model!")
    print()
    print("Sample indicators that should be trained:")
    for col in indicators_skipped[:10]:
        non_null = df[col].notna().sum()
        print(f"   - {col}: {non_null} samples")
    if len(indicators_skipped) > 10:
        print(f"   ... and {len(indicators_skipped) - 10} more")
    print()

# Check low coverage indicators
low_coverage = []
for col in indicator_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        coverage = non_null / len(df) * 100
        if coverage < 10:  # Less than 10% coverage
            low_coverage.append((col, non_null, coverage))

print(f"Low Coverage Indicators (<10% data):")
print(f"   Count: {len(low_coverage)}")
print(f"   These indicators have insufficient training data")
print()

# Overall statistics
total_indicators = len(indicator_cols)
usable_indicators = len(indicators_with_data)
trained_pct = len(models) / total_indicators * 100
usable_pct = usable_indicators / total_indicators * 100

print("=" * 80)
print(" SUMMARY:")
print("=" * 80)
print(f"Total Indicators in Data:        {total_indicators}")
print(f"Indicators with Usable Data:     {usable_indicators} ({usable_pct:.1f}%)")
print(f"Indicators with Trained Models:  {len(models)} ({trained_pct:.1f}%)")
print()
print(f"🎯 PREDICTION CAPABILITY:")
print(f"   Your model can predict answers for ~{trained_pct:.0f}% of indicators")
print()

if trained_pct < 50:
    print(" WARNING: Low prediction coverage detected!")
    print("   Possible causes:")
    print("   1. Many indicators have insufficient training data")
    print("   2. Training threshold might be too strict")
    print("   3. Need more diverse training samples")
print()
print("=" * 80)
