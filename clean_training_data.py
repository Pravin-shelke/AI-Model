import pandas as pd
import numpy as np

def clean_training_data():
    """Remove invalid values like 'Unknown', 'NO TEXT', empty strings from training data"""
    
    print("\n🧹 Cleaning Training Data...")
    print("="*70)
    
    # Load training data
    df = pd.read_csv('Balaji_Framework_Training_Data.csv')
    print(f"✓ Loaded {len(df)} records")
    
    # Backup original
    backup_file = f"Balaji_Framework_Training_Data_BEFORE_CLEAN.csv"
    df.to_csv(backup_file, index=False)
    print(f"✓ Backed up to: {backup_file}")
    
    # Get all indicator columns (skip the essential features)
    essential_features = ['country_code', 'crop_name', 'Partner', 'irrigation', 'hired_workers', 'area']
    indicator_columns = [col for col in df.columns if col not in essential_features and 
                        not col.startswith('plan_') and 
                        not col.startswith('assessment_') and
                        not col.startswith('user_') and
                        not col.startswith('workspace_') and
                        not col.startswith('email_') and
                        not col in ['Unique identifier', 'address', 'county', 'locale', 
                                   'state', 'zipcode', 'longitude', 'latitude', 
                                   'additional_farm_area', 'sub_partner', 'plan_year',
                                   'plan_created', 'plan_modified', 'plan_deleted', 
                                   'Submitted', 'unit_type', 'yield_unit', 'nitrogen_unit',
                                   'essential_percentage', 'intermediate_percentage', 
                                   'advanced_percentage', 'sai_score',
                                   'Carbon-score - Total Emissions',
                                   'Carbon-score - Residue Management',
                                   'Carbon-score - Fertilizer Production',
                                   'Carbon-score - Applied Fertilizer',
                                   'Carbon-score - Crop Protection',
                                   'Carbon-score - Carbon Stock Changes',
                                   'Carbon-score - Energy Use (field)',
                                   'Carbon-score - Delivery to point of sale',
                                   'Carbon-score - Irrigation',
                                   'Reason for failure of Carbon-score generation']]
    
    print(f"✓ Found {len(indicator_columns)} indicator columns to clean")
    
    # Invalid values to replace with NaN
    invalid_values = ['Unknown', 'NO TEXT', 'unknown', 'no text', '', ' ', 'N/A', 'NA', 'null']
    
    # Clean each indicator column
    total_cleaned = 0
    for col in indicator_columns:
        if col in df.columns:
            # Count invalid values before cleaning
            before = df[col].isin(invalid_values).sum() + df[col].isna().sum()
            
            # Replace invalid values with NaN
            df[col] = df[col].replace(invalid_values, np.nan)
            
            # Also replace values that are just whitespace
            df[col] = df[col].apply(lambda x: np.nan if isinstance(x, str) and x.strip() == '' else x)
            
            after = df[col].isna().sum()
            if after > before:
                total_cleaned += (after - before)
    
    print(f"✓ Cleaned {total_cleaned} invalid values across all indicators")
    
    # Calculate data quality metrics
    total_cells = len(df) * len(indicator_columns)
    valid_cells = 0
    for col in indicator_columns:
        if col in df.columns:
            valid_cells += df[col].notna().sum()
    
    fill_percentage = (valid_cells / total_cells) * 100
    print(f"✓ Data completeness: {fill_percentage:.1f}% ({valid_cells:,}/{total_cells:,} cells)")
    
    # Save cleaned data
    df.to_csv('Balaji_Framework_Training_Data.csv', index=False)
    print(f"✓ Saved cleaned data: {len(df)} records")
    
    print("\n✅ Cleaning Complete!")
    print("="*70)
    print("\n⚡ Next step: Retrain the model")
    print("   python retrain_model.py --retrain")
    print()
    
    return df

if __name__ == "__main__":
    clean_training_data()
