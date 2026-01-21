import pandas as pd

df = pd.read_csv('Balaji_Framework_Training_Data.csv')
print(f'Rows: {len(df)}')
print(f'Columns: {len(df.columns)}')

# Check SAI indicators
sai_cols = [c for c in df.columns if any(c.strip().startswith(p) for p in ['BH-', 'BP-', 'CE-', 'CW-'])]
print(f'\nFound {len(sai_cols)} SAI columns')

if sai_cols:
    col = sai_cols[0]
    print(f'\nFirst SAI column: {col}')
    print(df[col].value_counts())
    print(f'\nNull count: {df[col].isna().sum()}')
    print(f'Unique values: {df[col].nunique()}')
