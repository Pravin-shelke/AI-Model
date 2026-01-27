"""
Convert mock_test_data.csv to proper training format
The mock data has JSON in 'labelsAnswersMap' column that needs to be parsed
"""
import pandas as pd
import json
import sys
from datetime import datetime

print("=" * 80)
print("CONVERTING MOCK TEST DATA TO TRAINING FORMAT")
print("=" * 80)
print()

# Load mock test data
print("📂 Loading mock_test_data.csv...")
df_mock = pd.read_csv('mock_test_data.csv')
print(f"   Loaded {len(df_mock)} records")
print()

# Load existing training data to get column structure
print("📂 Loading existing training data structure...")
df_training = pd.read_csv('data/training/Assessment_AI_Training_Data.csv')
print(f"   Training data has {len(df_training.columns)} columns")
print()

# Create new dataframe with training structure
print("🔄 Converting mock data to training format...")
converted_data = []

for idx, row in df_mock.iterrows():
    # Start with base metadata
    record = {
        'plan_id': f'mock_{idx}',
        'assessment_name': f'Mock Assessment {idx}',
        'country_code': row['country_code'],
        'crop_name': row['crop_name'],
        'Partner': row['Partner'],
        'irrigation': row['irrigation'],
        'hired_workers': 'Yes' if row['hired_workers'] == True or row['hired_workers'] == 'TRUE' else 'No',
        'area': row['area'],
        'plan_year': row.get('planYear', 2025)
    }
    
    # Parse the JSON labels/answers map
    try:
        if pd.notna(row['labelsAnswersMap']):
            labels_map = json.loads(row['labelsAnswersMap'])
            
            # Extract answers from the JSON structure
            for item in labels_map:
                soa_id = item.get('soaId')
                label_name = item.get('labelName')
                answer = item.get('answer')
                
                # If there's an answer, add it to the record
                if answer and soa_id:
                    # Use soaId as column name if available
                    record[soa_id] = answer
                elif answer and label_name:
                    # Otherwise use label name
                    record[label_name] = answer
    except Exception as e:
        print(f"⚠️  Warning: Could not parse JSON for row {idx}: {e}")
    
    converted_data.append(record)
    
    if (idx + 1) % 1000 == 0:
        print(f"   Processed {idx + 1} records...")

# Create DataFrame
df_converted = pd.DataFrame(converted_data)
print(f"\n✅ Converted {len(df_converted)} records")
print(f"   Columns: {len(df_converted.columns)}")
print()

# Show what indicators were extracted
indicator_cols = [col for col in df_converted.columns if '-' in col]
print(f"📊 Extracted {len(indicator_cols)} indicator columns:")
for col in sorted(indicator_cols)[:20]:
    non_null = df_converted[col].notna().sum()
    print(f"   • {col}: {non_null} values")
if len(indicator_cols) > 20:
    print(f"   ... and {len(indicator_cols) - 20} more")
print()

# Merge with existing training data structure
print("🔗 Merging with existing training data...")

# Add missing columns from training data (fill with NaN)
for col in df_training.columns:
    if col not in df_converted.columns:
        df_converted[col] = None

# Reorder columns to match training data
df_converted = df_converted[df_training.columns]

# Combine with existing training data
df_combined = pd.concat([df_training, df_converted], ignore_index=True)

# Save backup
backup_file = f'data/training/Assessment_AI_Training_Data_Backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
print(f"💾 Creating backup: {backup_file}")
df_training.to_csv(backup_file, index=False)

# Save combined data
output_file = 'data/training/Assessment_AI_Training_Data.csv'
print(f"💾 Saving combined training data: {output_file}")
df_combined.to_csv(output_file, index=False)

print()
print("=" * 80)
print("✅ CONVERSION COMPLETE!")
print("=" * 80)
print(f"Previous training data:  {len(df_training)} records")
print(f"New mock data added:     {len(df_converted)} records")
print(f"Total training data:     {len(df_combined)} records")
print()
print("Next step: Retrain the model")
print("  cd /Users/pravinshelke/Documents/AI-Model")
print("  python src/training/trainer.py --retrain")
print()
