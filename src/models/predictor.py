import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import json
import warnings
warnings.filterwarnings('ignore')


class AssessmentAIPredictor:
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.target_columns = []
        self.df = None
        self.feature_modes = {}
        self.min_training_samples = 20
        self.questions_metadata = {}
        self.suggest_thresholds = {
            'radio_binary': 80.0,
            'radio_multiclass': 70.0
        }

    def load_questions_metadata(self, json_file):
        """Load question metadata for type-aware prediction behavior"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.questions_metadata = {
            q.get('indicatorCode'): q for q in data.get('questions', [])
        }
        print(f"Loaded metadata for {len(self.questions_metadata)} questions")
        
    def load_data(self, csv_file):
        print(f"\nLoading data from {csv_file}...")
        self.df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"Loaded {len(self.df)} records")
        
        self.feature_columns = [
            'country_code',
            'crop_name',
            'Partner',
            'irrigation',
            'hired_workers',
            'area'
        ]
        
        sai_prefixes = ['BH-', 'BP-', 'CE-', 'CW-', 'CT-', 'CC-', 'HS-', 'HW-', 'HP-', 'HA-', 'HH-',
                       'OR-', 'OP-', 'OS-', 'OT-', 'OF-', 'OE-', 'OM-', 'OW-',
                       'SR-', 'SW-', 'SF-', 'SC-', 'SM-',
                       'WI-', 'WP-', 'WQ-',
                       'CM-', 'CO-', 'FM-', 'IO-', 'IM-', 'LM-', 'NM-', 'PM-', 'RM-', 'ST-']
        
        self.target_columns = []
        for col in self.df.columns:
            if any(col.strip().startswith(p) for p in sai_prefixes):
                if col not in self.feature_columns:
                    self.target_columns.append(col)
        
        print(f"Found {len(self.target_columns)} indicators")
        return self.df
    
    def prepare_training_data(self):
        print("\nPreparing data...")
        
        # Normalize hired_workers column to handle boolean and string values
        if 'hired_workers' in self.df.columns:
            # Convert booleans and variations to 'Yes'/'No'
            self.df['hired_workers'] = self.df['hired_workers'].apply(lambda x: 
                'Yes' if x in [True, 'True', 'TRUE', 'yes', 'YES', 'Y', 1, '1'] 
                else 'No' if x in [False, 'False', 'FALSE', 'no', 'NO', 'N', 0, '0']
                else str(x) if pd.notna(x) else 'Unknown'
            )
        
        for col in self.feature_columns:
            if self.df[col].dtype == 'object' or col == 'hired_workers':
                le = LabelEncoder()
                # Convert all values to strings first
                values = self.df[col].astype(str).fillna('Unknown')
                classes = values.unique().tolist()
                if 'Unknown' not in classes:
                    classes.append('Unknown')
                le.fit(classes)
                self.df[col] = values
                self.df[col + '_encoded'] = le.transform(values)
                self.label_encoders[col] = le
                self.feature_modes[col] = values.mode(dropna=True)[0]
        
        if 'area' in self.feature_columns:
            self.df['area'] = pd.to_numeric(self.df['area'], errors='coerce').fillna(0)
        
        print("Data ready")
    
    def train_models(self):
        print("\nTraining models...")
        
        X_cols = []
        for col in self.feature_columns:
            if col + '_encoded' in self.df.columns:
                X_cols.append(col + '_encoded')
            else:
                X_cols.append(col)
        
        X = self.df[X_cols].values
        
        trained_count = 0
        skipped_count = 0
        
        # Train a model for each target column
        for idx, target_col in enumerate(self.target_columns, 1):
            try:
                # Skip targets that are not suitable for ML prediction
                meta = self.questions_metadata.get(target_col, {})
                q_type = meta.get('type')
                if q_type in {'text', 'checkbox'}:
                    skipped_count += 1
                    continue

                # Skip if all values are null or same
                if self.df[target_col].isna().all():
                    skipped_count += 1
                    continue
                
                # Skip if only one unique value (no variation to learn)
                unique_vals = self.df[target_col].dropna().nunique()
                if unique_vals <= 1:
                    skipped_count += 1
                    continue
                
                # Skip if less than 30% data available (too many nulls)
                null_percentage = (self.df[target_col].isna().sum() / len(self.df)) * 100
                if null_percentage > 70:
                    skipped_count += 1
                    continue
                
                # Prepare target - only use rows with valid data
                valid_indices = self.df[target_col].notna()
                y = self.df.loc[valid_indices, target_col]
                X_valid = X[valid_indices]
                
                # Skip if less than 10 training samples
                if len(y) < 10:
                    skipped_count += 1
                    continue

                # Skip if severely imbalanced (model will always guess majority)
                value_counts = y.value_counts(normalize=True)
                if value_counts.max() > 0.85:
                    skipped_count += 1
                    continue
                
                # Create label encoder for target
                le_target = LabelEncoder()
                y_encoded = le_target.fit_transform(y)
                num_classes = y_encoded.nunique()
                if num_classes == 2:
                    objective = 'binary:logistic'
                else:
                    objective = 'multi:softprob'
                
                # Train XGBoost classifier
                model = xgb.XGBClassifier(
                    max_depth=3,
                    n_estimators=50,
                    learning_rate=0.1,
                    objective=objective,
                    random_state=42,
                    verbosity=0
                )
                
                model.fit(X_valid, y_encoded)
                
                # Store model and encoder
                self.models[target_col] = {
                    'model': model,
                    'encoder': le_target,
                    'feature_cols': X_cols,
                    'objective': objective,
                    'training_samples': len(y)
                }
                
                trained_count += 1
                
                # Progress indicator
                if trained_count % 20 == 0:
                    print(f"  ✓ Trained {trained_count} models...")
                    
            except Exception as e:
                skipped_count += 1
                continue
        
        print(f"\n✅ Training Complete!")
        print(f"   • Successfully trained: {trained_count} models")
        print(f"   • Skipped (insufficient data): {skipped_count} indicators")
        
        return trained_count
    
    def predict_assessment(self, country, crop, partner, irrigation, hired_workers, area, existing_answers=None):
        """
        Predict all assessment indicators based on 6 inputs
        
        Parameters:
        -----------
        country : str
            Country code (e.g., 'IN', 'US', 'BR')
        crop : str
            Crop name (e.g., 'Potato', 'Corn', 'Soybean')
        partner : str
            Partner name (e.g., 'Balaji ')
        irrigation : str
            Irrigation type (e.g., 'irrigated', 'rainfed')
        hired_workers : str
            Hired workers ('Yes' or 'No')
        area : float
            Farm size in acres
        
        Returns:
        --------
        dict : Predicted values for all assessment indicators with confidence scores
        """
        print(f"\n🔮 Predicting assessment for:")
        print(f"   Country: {country}")
        print(f"   Crop: {crop}")
        print(f"   Partner: {partner}")
        print(f"   Irrigation: {irrigation}")
        print(f"   Hired Workers: {hired_workers}")
        print(f"   Area: {area} acres")
        
        # Prepare input features
        input_data = {
            'country_code': country,
            'crop_name': crop,
            'Partner': partner,
            'irrigation': irrigation,
            'hired_workers': hired_workers,
            'area': area
        }
        
        # Encode categorical features
        X_input = []
        for col in self.feature_columns:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                try:
                    encoded_val = le.transform([input_data[col]])[0]
                except ValueError:
                    # Handle unseen category - prefer 'Unknown', else most common value
                    fallback = 'Unknown' if 'Unknown' in le.classes_ else self.feature_modes.get(col, le.classes_[0])
                    encoded_val = le.transform([fallback])[0]
                X_input.append(encoded_val)
            else:
                X_input.append(input_data[col])
        
        X_input = np.array(X_input).reshape(1, -1)
        
        # Predict all assessment indicators
        predictions = {}
        high_confidence_count = 0
        
        existing_answers = existing_answers or {}

        for target_col, model_data in self.models.items():
            try:
                meta = self.questions_metadata.get(target_col, {})
                q_type = meta.get('type')
                options = meta.get('options', [])

                # Conditional gating (only predict if condition met)
                if meta.get('conditional'):
                    show_if = meta.get('showIf', {})
                    parent_indicator = show_if.get('indicator')
                    parent_value = show_if.get('value')
                    if parent_indicator and parent_value:
                        parent_pred = existing_answers.get(parent_indicator)
                        if parent_pred is None:
                            parent_pred = predictions.get(parent_indicator, {}).get('value')
                        if parent_pred != parent_value:
                            predictions[target_col] = {
                                'value': 'Unknown',
                                'confidence': 0.0,
                                'method': 'skipped_conditional',
                                'questionType': q_type,
                                'suggestOnly': True
                            }
                            continue

                # Skip non-ML types at prediction time as safety net
                if q_type == 'text':
                    predictions[target_col] = {
                        'value': 'Unknown',
                        'confidence': 0.0,
                        'method': 'skipped_text',
                        'questionType': q_type,
                        'suggestOnly': True
                    }
                    continue
                if q_type == 'checkbox':
                    predictions[target_col] = {
                        'suggestedOptions': [],
                        'confidence': 0.0,
                        'method': 'skipped_checkbox',
                        'questionType': q_type,
                        'suggestOnly': True
                    }
                    continue

                if model_data.get('training_samples', 0) < self.min_training_samples:
                    predictions[target_col] = {
                        'value': 'Unknown',
                        'confidence': 0.0,
                        'method': 'insufficient_data',
                        'questionType': q_type,
                        'suggestOnly': True
                    }
                    continue

                model = model_data['model']
                encoder = model_data['encoder']
                
                # Get prediction
                y_pred = model.predict(X_input)[0]
                
                # Get probability scores
                y_proba = model.predict_proba(X_input)[0]
                if model_data.get('objective') == 'binary:logistic' and len(y_proba) >= 2:
                    proba_yes = float(y_proba[1])
                    confidence = abs(proba_yes - 0.5) * 200
                else:
                    confidence = float(np.max(y_proba) * 100)
                
                # Decode prediction
                predicted_value = encoder.inverse_transform([int(y_pred)])[0]
                
                predictions[target_col] = {
                    'value': predicted_value,
                    'confidence': confidence,
                    'method': 'xgboost',
                    'questionType': q_type
                }

                # Suggest-only gating based on question type and confidence
                is_binary_radio = q_type == 'radio' and len(options) == 2
                if is_binary_radio:
                    threshold = self.suggest_thresholds['radio_binary']
                else:
                    threshold = self.suggest_thresholds['radio_multiclass']

                if confidence >= threshold:
                    predictions[target_col]['suggestOnly'] = True
                else:
                    predictions[target_col]['value'] = 'Unknown'
                    predictions[target_col]['confidence'] = 0.0
                    predictions[target_col]['suggestOnly'] = True
                
                if predictions[target_col].get('confidence', 0) >= 80:
                    high_confidence_count += 1
                    
            except Exception as e:
                predictions[target_col] = {
                    'value': 'Unknown',
                    'confidence': 0.0,
                    'method': 'error',
                    'questionType': q_type,
                    'suggestOnly': True
                }
        
        print(f"\n✅ Predictions Complete!")
        print(f"   • Total predictions: {len(predictions)}")
        print(f"   • High confidence (≥80%): {high_confidence_count}")
        print(f"   • Average confidence: {np.mean([p['confidence'] for p in predictions.values()]):.1f}%")
        
        return predictions
    
    def save_models(self, filename='xgboost_balaji_models.pkl'):
        """Save all trained models"""
        print(f"\n💾 Saving models to {filename}...")
        
        model_package = {
            'models': self.models,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns
        }
        
        joblib.dump(model_package, filename)
        
        import os
        file_size = os.path.getsize(filename) / (1024 * 1024)
        print(f"✅ Models saved! File size: {file_size:.1f} MB")
    
    def load_models(self, filename='xgboost_balaji_models.pkl'):
        """Load pre-trained models"""
        print(f"\n📂 Loading models from {filename}...")
        
        model_package = joblib.load(filename)
        
        self.models = model_package['models']
        self.label_encoders = model_package['label_encoders']
        self.feature_columns = model_package['feature_columns']
        self.target_columns = model_package['target_columns']
        
        print(f"✅ Loaded {len(self.models)} trained models")
    
    def export_predictions_to_csv(self, predictions, output_file='predicted_assessment.csv'):
        """Export predictions to CSV format"""
        print(f"\n📄 Exporting predictions to {output_file}...")
        
        # Create DataFrame from predictions
        data = []
        for indicator, pred_data in predictions.items():
            data.append({
                'Indicator': indicator,
                'Predicted_Value': pred_data['value'],
                'Confidence_%': f"{pred_data['confidence']:.1f}%"
            })
        
        df_export = pd.DataFrame(data)
        df_export.to_csv(output_file, index=False)
        
        print(f"✅ Predictions exported to {output_file}")
        print(f"   Total indicators: {len(data)}")


# ============================================================================
# DEMO / TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  XGBOOST BALAJI FRAMEWORK AI - QUESTIONNAIRE REDUCER")
    print("  Reduces 15-20 minute assessment to 2-3 minutes")
    print("=" * 70)
    
    # Initialize predictor
    predictor = AssessmentAIPredictor()
    
    # Load data (using training data with 52 records)
    predictor.load_data('Balaji_Framework_Training_Data.csv')
    # Load question metadata for type-aware behavior
    predictor.load_questions_metadata('config/questions_metadata.json')
    
    # Prepare training data
    predictor.prepare_training_data()
    
    # Train models
    trained_count = predictor.train_models()
    
    # Save models
    predictor.save_models('xgboost_balaji_models.pkl')
    
    print("\n" + "=" * 70)
    print("  DEMO: PREDICT ASSESSMENT FOR NEW FARMER")
    print("=" * 70)
    
    # Demo prediction for a new farmer
    predictions = predictor.predict_assessment(
        country='IN',
        crop='Potato',
        partner='Balaji ',
        irrigation='irrigated',
        hired_workers='Yes',
        area=10.0
    )
    
    # Show sample predictions
    print("\n📊 Sample Predictions (First 10 indicators):")
    print("-" * 70)
    count = 0
    for indicator, pred_data in predictions.items():
        if count < 10:
            print(f"  {indicator}: {pred_data['value']}")
            print(f"    └─ Confidence: {pred_data['confidence']:.1f}%")
            count += 1
    
    # Export predictions
    predictor.export_predictions_to_csv(predictions, 'farmer_assessment_predictions.csv')
    
    print("\n" + "=" * 70)
    print("  ✅ XGBoost Balaji AI Model Ready!")
    print("=" * 70)
    print("\n📈 Time Savings:")
    print("   • Original time: 15-20 minutes")
    print("   • With AI: 2-3 minutes")
    print("   • Time saved: 85-90%")
    print("\n🎯 Next Steps:")
    print("   1. Collect more assessment data (target: 50-100 records)")
    print("   2. Retrain model with more data")
    print("   3. Integrate into your app")
    print("   4. Let farmers complete assessments faster!")
    print("=" * 70)
