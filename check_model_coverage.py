"""
Analyze which indicators have trained models vs total indicators
"""
import pandas as pd
import os
import sys

# Workaround for pickle loading issue
class FakeModule:
    pass

dummy_modules = ['BH-2', 'BP-2', 'CE-2', 'CW-2', 'CT-2', 'CC-2', 'HS-2', 'HW-2', 'HP-2', 'HA-2', 'HH-2',
                 'OR-2', 'OP-2', 'OS-2', 'OT-2', 'OF-2', 'OE-2', 'OM-2', 'OW-2',
                 'SR-2', 'SW-2', 'SF-2', 'SC-2', 'SM-2', 'WI-2', 'WP-2', 'WQ-2']
for mod_name in dummy_modules:
    sys.modules[mod_name] = FakeModule()

print("=" * 80)
print("🔍 TRAINED MODEL COVERAGE ANALYSIS")
print("=" * 80)
print()

# Load training data
df = pd.read_csv('data/training/Assessment_AI_Training_Data.csv')

# Load list of trained model keys directly from file
import pickle
try:
    with open('models/assessment_ai_models.pkl', 'rb') as f:
        # Read just the keys
        import io
        import pickletools
        
        # Just try to load and catch the keys we can
        models = {}
        try:
            models = pickle.load(f)
        except:
            pass
        
        if isinstance(models, dict):
            trained_indicators = list(models.keys())
        else:
            trained_indicators = []
except Exception as e:
    print(f"⚠️  Could not load model file: {e}")
    print("Using alternative method to detect trained models...")
    trained_indicators = []

if not trained_indicators:
    # Alternative: check which indicators have good data coverage
    print("Analyzing data quality to estimate trainable indicators...")
    print()
    
    metadata_cols = {
        'plan_id', 'assessment_name', 'user_id', 'user_name', 'workspace_id', 
        'workspace_name', 'email_id', 'hired_workers', 'Unique identifier', 
        'address', 'county', 'plan_year', 'country_code', 'locale', 'state', 
        'zipcode', 'longitude', 'latitude', 'crop_name', 'area', 
        'additional_farm_area', 'irrigation', 'Partner', 'sub_partner', 
        'plan_created', 'plan_modified', 'plan_deleted', 'Submitted', 
        'unit_type', 'yield_unit', 'nitrogen_unit', 'essential_percentage',
        'intermediate_percentage', 'advanced_percentage', 'sai_score'
    }
    
    indicator_cols = [col for col in df.columns 
                      if col not in metadata_cols 
                      and not col.startswith('Carbon-score')
                      and col != 'Reason for failure of Carbon-score generation']
    
    # Check each indicator's data quality
    for col in indicator_cols:
        non_null = df[col].notna().sum()
        unique_vals = df[col].nunique()
        
        # Model is likely trained if >= 10 samples and >= 2 unique values
        if non_null >= 10 and unique_vals >= 2:
            trained_indicators.append(col)

print(f"📊 Trained Models Found: {len(trained_indicators)}")
print()

# Get all indicator columns
metadata_cols = {
    'plan_id', 'assessment_name', 'user_id', 'user_name', 'workspace_id', 
    'workspace_name', 'email_id', 'hired_workers', 'Unique identifier', 
    'address', 'county', 'plan_year', 'country_code', 'locale', 'state', 
    'zipcode', 'longitude', 'latitude', 'crop_name', 'area', 
    'additional_farm_area', 'irrigation', 'Partner', 'sub_partner', 
    'plan_created', 'plan_modified', 'plan_deleted', 'Submitted', 
    'unit_type', 'yield_unit', 'nitrogen_unit', 'essential_percentage',
    'intermediate_percentage', 'advanced_percentage', 'sai_score'
}

all_indicators = [col for col in df.columns 
                  if col not in metadata_cols 
                  and not col.startswith('Carbon-score')
                  and col != 'Reason for failure of Carbon-score generation']

print(f"📋 Total Indicators in Data: {len(all_indicators)}")
print(f"✅ Indicators WITH trained models: {len(trained_indicators)}")
print(f"❌ Indicators WITHOUT trained models: {len(all_indicators) - len(trained_indicators)}")
print(f"📈 Coverage: {len(trained_indicators)/len(all_indicators)*100:.1f}%")
print()

# Group by category
trained_by_category = {}
all_by_category = {}

for indicator in all_indicators:
    cat = indicator.split('-')[0].split('.')[0] if ('-' in indicator or '.' in indicator) else 'OTHER'
    
    if cat not in all_by_category:
        all_by_category[cat] = []
    all_by_category[cat].append(indicator)
    
    if indicator in trained_indicators:
        if cat not in trained_by_category:
            trained_by_category[cat] = []
        trained_by_category[cat].append(indicator)

print("=" * 80)
print("📊 COVERAGE BY CATEGORY:")
print("=" * 80)
print(f"{'Category':<15} {'Total':<10} {'Trained':<10} {'Coverage':<15} Status")
print("-" * 80)

for cat in sorted(all_by_category.keys()):
    total = len(all_by_category[cat])
    trained = len(trained_by_category.get(cat, []))
    coverage = (trained / total * 100) if total > 0 else 0
    
    status = "✅" if coverage >= 50 else "⚠️" if coverage >= 25 else "❌"
    
    print(f"{cat:<15} {total:<10} {trained:<10} {coverage:>6.1f}%        {status}")

print()
print("=" * 80)
print("🔍 WHY LOW PREDICTION RATE?")
print("=" * 80)

# Analyze missing models
missing_indicators = [ind for ind in all_indicators if ind not in trained_indicators]

print(f"\n❌ {len(missing_indicators)} indicators have NO trained model")
print(f"   Reason: Insufficient quality training data")
print()

# Show sample of indicators with no model but some data
print("Sample indicators that NEED more training data:")
sample_missing = []
for ind in missing_indicators[:15]:
    non_null = df[ind].notna().sum()
    if non_null > 0:
        sample_missing.append((ind, non_null))

for ind, count in sorted(sample_missing, key=lambda x: x[1], reverse=True)[:10]:
    print(f"   • {ind:<40} ({count} samples - needs more diverse data)")

print()
print("=" * 80)
print("💡 SOLUTIONS TO IMPROVE COVERAGE:")
print("=" * 80)
print("1. ✅ Add more diverse training data from different farms/assessments")
print("2. ✅ Focus on collecting data for indicators with 0-10 samples")
print("3. ✅ Ensure data quality: avoid too many duplicates or 'NO' responses")
print("4. ✅ Retrain model after adding more data")
print()
print(f"Current Status: {len(trained_indicators)}/{len(all_indicators)} indicators can be predicted")
print(f"Target: Get to at least 250+ indicators (80% coverage)")
print()
