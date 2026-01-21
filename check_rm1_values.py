import pandas as pd

# Load training data
df = pd.read_csv('Balaji_Framework_Training_Data.csv')

# Check RM-1 column
if 'RM-1' in df.columns:
    print("\n" + "="*70)
    print("RM-1 Values in Training Data:")
    print("="*70)
    print(f"\nTotal records: {len(df)}")
    print(f"\nValue counts:")
    print(df['RM-1'].value_counts(dropna=False))
    print(f"\nUnique values:")
    for val in df['RM-1'].unique():
        print(f"  - '{val}'")
else:
    print("RM-1 column not found!")
    print(f"\nAvailable columns with 'RM': {[col for col in df.columns if 'RM' in col]}")
