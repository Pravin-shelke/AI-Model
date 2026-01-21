"""
Retrain XGBoost Model with New Data
Adds new assessment data and retrains the model for better accuracy
"""

import pandas as pd
import numpy as np
from assessment_ai_predictor import AssessmentAIPredictor
import os
from datetime import datetime

def check_existing_data():
    """Check what training data currently exists"""
    print("\n🔍 Checking Existing Training Data...")
    print("=" * 70)
    
    files = {
        'Original Data': 'Balaji  Framework 2025-2026-01-20.csv',
        'Training Data': 'Assessment_AI_Training_Data.csv',
        'Current Models': 'assessment_ai_models.pkl'
    }
    
    for name, filename in files.items():
        if os.path.exists(filename):
            if filename.endswith('.csv'):
                df = pd.read_csv(filename)
                print(f"✅ {name}: {len(df)} records, {len(df.columns)} columns")
            else:
                size = os.path.getsize(filename) / (1024 * 1024)
                print(f"✅ {name}: {size:.1f} MB")
        else:
            print(f"❌ {name}: Not found")
    print("=" * 70)

def add_new_assessments(new_csv_file):
    """
    Add new assessment data to training dataset
    
    Parameters:
    -----------
    new_csv_file : str
        Path to CSV file with new assessments (must have same format as Balaji Framework)
    """
    print(f"\n📥 Adding New Assessments from: {new_csv_file}")
    print("=" * 70)
    
    # Check if file exists
    if not os.path.exists(new_csv_file):
        print(f"❌ Error: File not found: {new_csv_file}")
        return False
    
    # Load new data
    df_new = pd.read_csv(new_csv_file, encoding='utf-8')
    print(f"✓ Loaded {len(df_new)} new records")
    
    # Load existing training data
    training_file = 'Assessment_AI_Training_Data.csv'
    if os.path.exists(training_file):
        df_existing = pd.read_csv(training_file, encoding='utf-8')
        print(f"✓ Existing training data: {len(df_existing)} records")
        
        # Combine
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        print(f"✓ Combined: {len(df_combined)} total records")
    else:
        df_combined = df_new
        print(f"✓ Creating new training file with {len(df_new)} records")
    
    # Backup old training data
    if os.path.exists(training_file):
        backup_file = f'Assessment_AI_Training_Data_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df_existing.to_csv(backup_file, index=False)
        print(f"✓ Backed up old data to: {backup_file}")
    
    # Save combined data
    df_combined.to_csv(training_file, index=False, encoding='utf-8')
    print(f"✅ Saved updated training data: {len(df_combined)} records")
    print("=" * 70)
    
    return True

def retrain_model():
    """Retrain the XGBoost model with updated data"""
    print("\n🚀 Retraining XGBoost Models...")
    print("=" * 70)
    
    # Backup old models
    if os.path.exists('assessment_ai_models.pkl'):
        backup_file = f'assessment_ai_models_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        os.rename('assessment_ai_models.pkl', backup_file)
        print(f"✓ Backed up old models to: {backup_file}")
    
    # Initialize predictor
    predictor = AssessmentAIPredictor()
    
    # Load training data
    predictor.load_data('Assessment_AI_Training_Data.csv')
    
    # Prepare data
    predictor.prepare_training_data()
    
    # Train models
    trained_count = predictor.train_models()
    
    # Save new models
    predictor.save_models('assessment_ai_models.pkl')
    
    print("\n✅ Retraining Complete!")
    print("=" * 70)
    
    return predictor, trained_count

def test_model_improvement(predictor):
    """Test the retrained model with sample predictions"""
    print("\n🧪 Testing Retrained Model...")
    print("=" * 70)
    
    # Test scenarios
    test_cases = [
        {
            'name': 'Indian Potato Farmer',
            'country': 'IN',
            'crop': 'Potato',
            'partner': 'Balaji ',
            'irrigation': 'irrigated',
            'hired_workers': 'Yes',
            'area': 10.0
        },
        {
            'name': 'US Corn Farmer',
            'country': 'US',
            'crop': 'Corn',
            'partner': 'Syngenta USA',
            'irrigation': 'rainfed',
            'hired_workers': 'No',
            'area': 50.0
        }
    ]
    
    for test_case in test_cases:
        name = test_case.pop('name')
        print(f"\nTest: {name}")
        print("-" * 70)
        
        predictions = predictor.predict_assessment(**test_case)
        
        # Calculate stats
        high_conf = sum(1 for p in predictions.values() if p['confidence'] >= 80)
        avg_conf = sum(p['confidence'] for p in predictions.values()) / len(predictions)
        
        print(f"  Total predictions: {len(predictions)}")
        print(f"  High confidence (≥80%): {high_conf}")
        print(f"  Average confidence: {avg_conf:.1f}%")
    
    print("\n" + "=" * 70)

def main():
    """Main retraining workflow"""
    print("=" * 70)
    print("  XGBOOST MODEL RETRAINING TOOL")
    print("=" * 70)
    
    # Check current state
    check_existing_data()
    
    # Ask user for new data
    print("\n📝 How to add more training data:")
    print("=" * 70)
    print("1. Export new assessments from your app to CSV format")
    print("2. Ensure CSV has the same columns as 'Balaji  Framework 2025-2026-01-20.csv'")
    print("3. Save the file in this folder")
    print("4. Run this script with the new file name")
    print("\nExample:")
    print("  python retrain_model.py --add new_assessments.csv")
    print("\nOr for automatic retraining with current data:")
    print("  python retrain_model.py --retrain")
    print("=" * 70)
    
    # Check for command line arguments
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--add' and len(sys.argv) > 2:
            # Add new data and retrain
            new_file = sys.argv[2]
            if add_new_assessments(new_file):
                predictor, trained_count = retrain_model()
                test_model_improvement(predictor)
                
                print("\n✅ Model Updated Successfully!")
                print(f"   • Trained {trained_count} models")
                print(f"   • Ready to use in your app")
                print("\n⚠️  Remember to restart the Flask API server:")
                print("   ./start_ai_server.sh")
        
        elif sys.argv[1] == '--retrain':
            # Just retrain with existing data
            predictor, trained_count = retrain_model()
            test_model_improvement(predictor)
            
            print("\n✅ Model Retrained Successfully!")
            print(f"   • Trained {trained_count} models")
            print("\n⚠️  Remember to restart the Flask API server:")
            print("   ./start_ai_server.sh")
        
        elif sys.argv[1] == '--generate' and len(sys.argv) > 2:
            # Generate synthetic data
            num_samples = int(sys.argv[2])
            print(f"\n🤖 Generating {num_samples} synthetic samples...")
            
            from generate_training_data import generate_synthetic_balaji_data
            generate_synthetic_balaji_data(
                input_csv='Balaji  Framework 2025-2026-01-20.csv',
                output_csv='Assessment_AI_Training_Data.csv',
                num_samples=num_samples
            )
            
            predictor, trained_count = retrain_model()
            test_model_improvement(predictor)
            
            print("\n✅ Synthetic Data Generated and Model Trained!")
        
        else:
            print("\n❌ Invalid command. Use:")
            print("   python retrain_model.py --add <new_data.csv>")
            print("   python retrain_model.py --retrain")
            print("   python retrain_model.py --generate <num_samples>")
    
    else:
        # Interactive mode
        print("\n💡 Quick Actions:")
        print("1. Generate more synthetic data (for testing)")
        print("2. Retrain with current data")
        print("3. Exit")
        
        choice = input("\nYour choice (1-3): ").strip()
        
        if choice == '1':
            num_samples = input("How many synthetic samples to generate? (default: 50): ").strip()
            num_samples = int(num_samples) if num_samples else 50
            
            from generate_training_data import generate_synthetic_balaji_data
            generate_synthetic_balaji_data(
                input_csv='Balaji  Framework 2025-2026-01-20.csv',
                output_csv='Assessment_AI_Training_Data.csv',
                num_samples=num_samples
            )
            
            predictor, trained_count = retrain_model()
            test_model_improvement(predictor)
            
            print("\n✅ Done! Restart Flask server: ./start_ai_server.sh")
        
        elif choice == '2':
            predictor, trained_count = retrain_model()
            test_model_improvement(predictor)
            
            print("\n✅ Done! Restart Flask server: ./start_ai_server.sh")
        
        else:
            print("\n👋 Exiting...")

if __name__ == "__main__":
    main()
