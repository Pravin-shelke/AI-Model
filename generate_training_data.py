"""
Generate Synthetic Training Data for Balaji Framework
Creates realistic assessment data based on existing patterns
"""

import pandas as pd
import numpy as np
import random

def generate_synthetic_balaji_data(input_csv, output_csv, num_samples=50):
    """
    Generate synthetic training data based on existing Balaji Framework records
    
    Parameters:
    -----------
    input_csv : str
        Path to existing Balaji Framework CSV
    output_csv : str
        Path to save generated data
    num_samples : int
        Number of synthetic samples to generate
    """
    print(f"\n🔄 Loading existing data from {input_csv}...")
    df_original = pd.read_csv(input_csv, encoding='utf-8')
    print(f"✓ Loaded {len(df_original)} original records")
    
    # Define variation options
    countries = ['IN', 'US', 'BR', 'CN', 'MX']
    crops = ['Potato', 'Corn', 'Soybean', 'Wheat', 'Rice', 'Cotton', 'Tomato']
    partners = ['Balaji ', 'Balaji-East', 'Balaji-West', 'Syngenta India', 'Syngenta USA']
    irrigations = ['irrigated', 'rainfed', 'drip irrigation', 'sprinkler']
    hired_workers_options = ['Yes', 'No']
    
    # Common answer variations for assessment indicators
    yes_no_answers = ['YES', 'NO', 'null']
    answer_weights_yes = [0.6, 0.3, 0.1]  # 60% Yes, 30% No, 10% null
    answer_weights_no = [0.3, 0.6, 0.1]   # 30% Yes, 60% No, 10% null
    
    text_answers = [
        'NO TEXT',
        'Safety training',
        'Training records kept',
        'Grassed waterways in place',
        'Buffer or Filter Strips in place',
        'Crop rotation provides conservation benefit',
        'Legumes (e.g., clover, triticale)',
        'Global GAP',
        'USDA Organic Program',
        'Ground water',
        'Flood/Furrow',
        'Economic optimum nutrient rate',
        '4R Nutrient Stewardship',
        ''
    ]
    
    print(f"\n🤖 Generating {num_samples} synthetic assessment records...")
    
    # Create synthetic rows
    synthetic_rows = []
    
    for i in range(num_samples):
        # Start with a copy of a random original row
        base_row = df_original.iloc[random.randint(0, len(df_original)-1)].copy()
        
        # Modify key input features
        base_row['country_code'] = random.choice(countries)
        base_row['crop_name'] = random.choice(crops)
        base_row['Partner'] = random.choice(partners)
        base_row['irrigation'] = random.choice(irrigations)
        base_row['hired_workers'] = random.choice(hired_workers_options)
        base_row['area'] = round(random.uniform(0.5, 50), 1)
        base_row['additional_farm_area'] = round(random.uniform(0, 20), 1)
        
        # Vary assessment indicators based on patterns
        for col in base_row.index:
            col_stripped = col.strip()
            
            # Skip essential features
            if col in ['country_code', 'crop_name', 'Partner', 'irrigation', 'hired_workers', 'area']:
                continue
            
            # Skip ID and metadata fields
            if col in ['plan_id', 'user_id', 'workspace_id', 'email_id', 'plan_created', 'plan_modified']:
                continue
            
            # Check if this is an SAI assessment indicator
            sai_prefixes = ['BH-', 'BP-', 'CE-', 'CW-', 'CT-', 'CC-', 'HS-', 'HW-', 'HP-', 'HA-', 'HH-',
                           'OR-', 'OP-', 'OS-', 'OT-', 'OF-', 'OE-', 'OM-', 'OW-',
                           'SR-', 'SW-', 'SF-', 'SC-', 'SM-',
                           'WI-', 'WP-', 'WQ-',
                           'CM-', 'CO-', 'FM-', 'IO-', 'IM-', 'LM-', 'NM-', 'PM-', 'RM-', 'ST-']
            
            is_sai_indicator = any(col_stripped.startswith(prefix) for prefix in sai_prefixes)
            
            if is_sai_indicator:
                current_val = str(base_row[col])
                
                # If current value is YES or NO, vary it
                if current_val in ['YES', 'NO']:
                    if current_val == 'YES':
                        base_row[col] = np.random.choice(yes_no_answers, p=answer_weights_yes)
                    else:
                        base_row[col] = np.random.choice(yes_no_answers, p=answer_weights_no)
                
                # If current value is text, occasionally change it
                elif current_val not in ['null', 'nan', '']:
                    if random.random() < 0.3:  # 30% chance to change
                        base_row[col] = random.choice(text_answers)
                
                # If current value is null, sometimes add data
                elif current_val in ['null', 'nan', ''] or pd.isna(current_val):
                    if random.random() < 0.4:  # 40% chance to fill in
                        if random.random() < 0.6:
                            base_row[col] = np.random.choice(yes_no_answers, p=[0.5, 0.4, 0.1])
                        else:
                            base_row[col] = random.choice(text_answers)
        
        synthetic_rows.append(base_row)
        
        if (i + 1) % 10 == 0:
            print(f"  ✓ Generated {i + 1}/{num_samples} records...")
    
    # Combine original and synthetic data
    df_combined = pd.concat([df_original, pd.DataFrame(synthetic_rows)], ignore_index=True)
    
    print(f"\n💾 Saving combined data to {output_csv}...")
    df_combined.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"\n✅ Data Generation Complete!")
    print(f"   • Original records: {len(df_original)}")
    print(f"   • Synthetic records: {num_samples}")
    print(f"   • Total records: {len(df_combined)}")
    print(f"   • Saved to: {output_csv}")
    
    return df_combined


if __name__ == "__main__":
    print("=" * 70)
    print("  SYNTHETIC DATA GENERATOR FOR BALAJI FRAMEWORK")
    print("=" * 70)
    
    # Generate 50 synthetic samples
    generate_synthetic_balaji_data(
        input_csv='Balaji  Framework 2025-2026-01-20.csv',
        output_csv='Assessment_AI_Training_Data.csv',
        num_samples=50
    )
    
    print("\n" + "=" * 70)
    print("  🎯 Next Step: Train XGBoost Model")
    print("=" * 70)
    print("\n  Run this command:")
    print("  python assessment_ai_predictor.py")
    print("\n" + "=" * 70)
