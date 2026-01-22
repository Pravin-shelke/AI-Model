#!/usr/bin/env python3
"""
Add Real Assessment Data to Training
Quick script to add your real user assessment data to the training dataset
"""

import sys
import os
import pandas as pd
from datetime import datetime
from src.training.data_loader import check_csv_format, add_to_training_data

def main():
    print("=" * 80)
    print("  ADD REAL ASSESSMENT DATA TO TRAINING")
    print("=" * 80)
    
    # Get CSV filename from command line or ask user
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        print("\n📝 Instructions:")
        print("1. Export your user assessments to CSV format")
        print("2. Make sure CSV has same columns as SAI Framework")
        print("3. Place the CSV file in this folder")
        print("4. Run: python add_real_data.py your_file.csv")
        
        csv_file = input("\n📁 Enter your CSV filename: ").strip()
        
        if not csv_file:
            print("\n💡 Example Usage:")
            print('   python add_real_data.py "United States Framework 2025-2026-01-21.csv"')
            return
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"\n❌ File not found: {csv_file}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Please make sure the file is in the current directory")
        return
    
    print(f"\n📂 Processing file: {csv_file}")
    
    # Check CSV format
    if not check_csv_format(csv_file):
        print("\n❌ CSV format check failed. Please fix the issues above and try again.")
        return
    
    # Ask for confirmation before adding data
    print("\n" + "=" * 80)
    confirm = input("✅ CSV format is correct. Add this data to training? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("❌ Operation cancelled")
        return
    
    # Add to training data
    try:
        total_records = add_to_training_data(csv_file)
        
        print("\n" + "=" * 80)
        print("✅ SUCCESS! Data added to training dataset")
        print("=" * 80)
        
        print("\n📊 Next Steps:")
        print("1. Retrain the model:")
        print("   cd src/training")
        print("   python trainer.py --retrain")
        print("\n2. Restart the API server:")
        print("   python src/api/server.py")
        print("\n3. Test the predictions with your new data!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error adding data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
