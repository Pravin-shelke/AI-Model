"""
Add Your Real Assessment Data to Training
This script helps you prepare and add your existing user assessment data
"""

import pandas as pd
import os
from datetime import datetime

def check_csv_format(csv_file):
    """
    Check if your CSV file has the correct format
    """
    print("\n🔍 Checking CSV Format...")
    print("=" * 70)
    
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        
        print(f"✓ File loaded successfully")
        print(f"✓ Total records: {len(df)}")
        print(f"✓ Total columns: {len(df.columns)}")
        
        # Check required columns
        required_columns = ['country_code', 'crop_name', 'Partner', 'irrigation', 'hired_workers', 'area']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"\n❌ Missing required columns: {missing_columns}")
            print("\nYour CSV must have these columns:")
            for col in required_columns:
                print(f"  - {col}")
            return False
        else:
            print(f"✓ All required columns present")
        
        # Show column preview
        print(f"\n📋 First few columns:")
        print(df.columns[:10].tolist())
        
        # Show sample data
        print(f"\n📊 Sample data (first 2 rows):")
        print(df[required_columns].head(2))
        
        # Check for SAI indicators
        sai_prefixes = ['BH-', 'BP-', 'CE-', 'CW-', 'CT-', 'CC-', 'HS-', 'HW-', 'HP-', 'HA-', 'HH-',
                       'OR-', 'OP-', 'OS-', 'OT-', 'OF-', 'OE-', 'OM-', 'OW-',
                       'SR-', 'SW-', 'SF-', 'SC-', 'SM-',
                       'WI-', 'WP-', 'WQ-',
                       'CM-', 'CO-', 'FM-', 'IO-', 'IM-', 'LM-', 'NM-', 'PM-', 'RM-', 'ST-']
        
        sai_columns = [col for col in df.columns if any(col.strip().startswith(prefix) for prefix in sai_prefixes)]
        print(f"\n✓ Found {len(sai_columns)} SAI assessment indicators")
        
        # Check data completeness
        null_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        print(f"✓ Data completeness: {100 - null_percentage:.1f}% filled")
        
        print("\n✅ CSV Format is CORRECT! Ready to add to training data.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error reading CSV: {e}")
        return False

def map_your_columns_to_balaji_format(your_csv, output_csv):
    """
    If your CSV has different column names, map them here
    """
    print("\n🔄 Mapping Your Data to Balaji Format...")
    print("=" * 70)
    
    df = pd.read_csv(your_csv, encoding='utf-8')
    
    # Example mapping - CUSTOMIZE THIS based on your column names
    column_mapping = {
        # Your column name → Balaji column name
        # 'your_country': 'country_code',
        # 'your_crop': 'crop_name',
        # 'your_partner': 'Partner',
        # 'your_irrigation': 'irrigation',
        # 'workers': 'hired_workers',
        # 'farm_size': 'area',
    }
    
    # Apply mapping if you have different column names
    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"✓ Mapped {len(column_mapping)} columns")
    
    # Save mapped data
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✓ Saved mapped data to: {output_csv}")
    print("=" * 70)
    
    return output_csv

def add_to_training_data(new_data_csv):
    """
    Add your data to the training dataset
    """
    print("\n📥 Adding Your Data to Training Dataset...")
    print("=" * 70)
    
    # Load your new data
    df_new = pd.read_csv(new_data_csv, encoding='utf-8')
    print(f"✓ Your data: {len(df_new)} records")
    
    # Load existing training data
    training_file = 'Assessment_AI_Training_Data.csv'
    
    if os.path.exists(training_file):
        df_existing = pd.read_csv(training_file, encoding='utf-8')
        print(f"✓ Existing training data: {len(df_existing)} records")
        
        # Backup existing data
        backup_file = f'Training_Data_Backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df_existing.to_csv(backup_file, index=False)
        print(f"✓ Backed up to: {backup_file}")
        
        # Combine data
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        print("✓ Creating new training file")
        df_combined = df_new
    
    # Remove duplicates
    before_dedup = len(df_combined)
    df_combined = df_combined.drop_duplicates()
    after_dedup = len(df_combined)
    
    if before_dedup > after_dedup:
        print(f"✓ Removed {before_dedup - after_dedup} duplicate records")
    
    # Save combined data
    df_combined.to_csv(training_file, index=False, encoding='utf-8')
    
    print(f"\n✅ Training Data Updated!")
    print(f"   Total records now: {len(df_combined)}")
    print(f"   New records added: {len(df_new)}")
    print("=" * 70)
    
    return len(df_combined)

def main():
    print("=" * 70)
    print("  ADD REAL ASSESSMENT DATA TO TRAINING")
    print("=" * 70)
    
    print("\n📝 Instructions:")
    print("1. Export your user assessments to CSV format")
    print("2. Make sure CSV has same columns as Balaji Framework")
    print("3. Place the CSV file in this folder")
    print("4. Run this script with your filename")
    
    print("\n" + "=" * 70)
    
    # Ask user for their CSV file
    csv_file = input("\n📁 Enter your CSV filename (or press Enter for example): ").strip()
    
    if not csv_file:
        print("\n💡 Example Usage:")
        print("   1. Put your file in this folder: user_assessments.csv")
        print("   2. Run: python add_real_data.py")
        print("   3. Enter filename: user_assessments.csv")
        print("\nOr use command line:")
        print("   python add_real_data.py user_assessments.csv")
        return
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"\n❌ File not found: {csv_file}")
        print(f"   Please make sure the file is in: {os.getcwd()}")
        return
    
    # Check format
    if not check_csv_format(csv_file):
        print("\n❌ Please fix the CSV format and try again")
        return
    
    # Ask if mapping is needed
    print("\n❓ Do your column names match Balaji format exactly?")
    print("   (country_code, crop_name, Partner, irrigation, hired_workers, area)")
    needs_mapping = input("   Enter 'n' if you need to map different column names (y/n): ").strip().lower()
    
    if needs_mapping == 'n':
        print("\n📝 Edit this file (add_real_data.py) and update the column_mapping dictionary")
        print("   Then run this script again")
        return
    
    # Add to training data
    total_records = add_to_training_data(csv_file)
    
    # Ask if user wants to retrain now
    print("\n❓ Do you want to retrain the model now?")
    retrain = input("   This will take a few minutes (y/n): ").strip().lower()
    
    if retrain == 'y':
        print("\n🚀 Starting Model Retraining...")
        print("=" * 70)
        
        from assessment_ai_predictor import AssessmentAIPredictor
        
        # Backup old models
        if os.path.exists('assessment_ai_models.pkl'):
            backup_model = f'Models_Backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
            os.rename('assessment_ai_models.pkl', backup_model)
            print(f"✓ Backed up old models to: {backup_model}")
        
        # Train new models
        predictor = AssessmentAIPredictor()
        predictor.load_data('Assessment_AI_Training_Data.csv')
        predictor.prepare_training_data()
        trained_count = predictor.train_models()
        predictor.save_models('assessment_ai_models.pkl')
        
        print("\n✅ RETRAINING COMPLETE!")
        print(f"   • Trained {trained_count} models")
        print(f"   • Using {total_records} assessment records")
        print("\n⚠️  RESTART the Flask server to use new models:")
        print("   ./start_ai_server.sh")
    else:
        print("\n💡 To retrain later, run:")
        print("   python retrain_model.py --retrain")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage
        csv_file = sys.argv[1]
        if os.path.exists(csv_file):
            if check_csv_format(csv_file):
                add_to_training_data(csv_file)
                print("\n✅ Data added! Now retrain:")
                print("   python retrain_model.py --retrain")
        else:
            print(f"❌ File not found: {csv_file}")
    else:
        # Interactive mode
        main()
