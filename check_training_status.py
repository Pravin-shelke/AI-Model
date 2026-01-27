import pickle
import pandas as pd
from datetime import datetime
import os

print("=" * 80)
print("📊 MODEL TRAINING STATUS REPORT")
print("=" * 80)
print()

# Check model file
model_path = 'models/assessment_ai_models.pkl'
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
    print(f"📁 Model File Information:")
    print(f"   File: {model_path}")
    print(f"   Size: {file_size:.2f} MB")
    print(f"   Last Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

# Load training data
try:
    df = pd.read_csv('data/training/Assessment_AI_Training_Data.csv')
    print(f"📈 Training Data Statistics:")
    print(f"   Total samples (rows): {len(df)}")
    print(f"   Total features (columns): {len(df.columns)}")
    print()
    
    # Check for RM-1 column
    if 'RM-1' in df.columns:
        rm1_values = df['RM-1'].dropna()
        print(f"🎯 RM-1 Column Analysis:")
        print(f"   Total non-null values: {len(rm1_values)}")
        print(f"   Unique values: {rm1_values.nunique()}")
        print(f"   Missing values: {df['RM-1'].isna().sum()}")
        if len(rm1_values) > 0:
            print(f"\n   Value distribution (top 5):")
            for val, count in rm1_values.value_counts().head(5).items():
                print(f"      {val}: {count} samples")
    else:
        print(f"⚠️  RM-1 column not found in training data")
        
except Exception as e:
    print(f"❌ Error loading training data: {e}")

print()
print("=" * 80)
print("🔍 TRAINING QUALITY SUMMARY:")
print("=" * 80)

# Training data in both locations
try:
    df1 = pd.read_csv('Assessment_AI_Training_Data.csv')
    print(f"   Root directory data: {len(df1)} samples")
except:
    print(f"   Root directory data: Not found")

try:
    df2 = pd.read_csv('data/training/Assessment_AI_Training_Data.csv')
    print(f"   data/training/ data: {len(df2)} samples")
except:
    print(f"   data/training/ data: Not found")

print(f"\n   Model file size: {file_size:.2f} MB ({'✅ Good' if file_size > 50 else '⚠️  Check'})")
print(f"   Last training: {mod_time.strftime('%b %d, %Y at %H:%M')}")
print()
print("=" * 80)
