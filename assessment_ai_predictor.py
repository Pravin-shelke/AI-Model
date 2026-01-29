"""
Assessment AI - XGBoost Predictor for SAI Framework
Reduces 15-20 minute questionnaire to 2-3 minutes

User provides 6 key inputs:
1. Country
2. Crop
3. Partner Name
4. Irrigation Type
5. Hired Workers
6. Farm Size (Area)

AI predicts remaining 266+ SAI assessment indicators
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
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
        self.binary_rule_min_samples = 5
        self.binary_rule_confidence_cap = 90.0
        self.binary_rule_filter_levels = [
            ['country_code', 'crop_name', 'irrigation', 'area_bucket'],
            ['country_code', 'crop_name', 'irrigation'],
            ['country_code', 'crop_name'],
            ['country_code'],
            []
        ]

    @staticmethod
    def _normalize_yes_no(value):
        if pd.isna(value):
            return None
        val = str(value).strip().lower()
        if val in {'yes', 'y', '1', 'true'}:
            return 'Yes'
        if val in {'no', 'n', '0', 'false'}:
            return 'No'
        return None

    def _is_binary_yes_no(self, series):
        normalized = series.dropna().map(self._normalize_yes_no).dropna().unique().tolist()
        return len(normalized) > 0 and set(normalized).issubset({'Yes', 'No'})

    def _predict_yes_no_rule(self, target_col, filters):
        # Progressive filtering: strict -> relaxed until enough samples
        for level in self.binary_rule_filter_levels:
            subset = self.df.copy()
            for col in level:
                if col in subset.columns and col in filters:
                    subset = subset[subset[col] == filters[col]]
            if len(subset) >= self.binary_rule_min_samples:
                normalized = subset[target_col].dropna().map(self._normalize_yes_no).dropna()
                if not normalized.empty:
                    prob_yes = (normalized == 'Yes').mean()
                    value = 'Yes' if prob_yes >= 0.5 else 'No'
                    confidence = max(prob_yes, 1 - prob_yes) * 100
                    confidence = min(confidence, self.binary_rule_confidence_cap)
                    return value, confidence

        # Smarter fallback using global prevalence
        global_vals = self.df[target_col].dropna().map(self._normalize_yes_no).dropna()
        if global_vals.empty:
            return 'Unknown', 0.0

        prob_yes = (global_vals == 'Yes').mean()
        if prob_yes < 0.2:
            return 'No', 65.0
        if prob_yes > 0.8:
            return 'Yes', 65.0
        return 'Unknown', 40.0
        
    def load_data(self, csv_file):
        """Load SAI Framework CSV assessment data"""
        print("\n🔄 Loading Balaji Framework data...")
        self.df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"✓ Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        
        # Define the 6 essential input features
        self.feature_columns = [
            'country_code',      # Country
            'crop_name',         # Crop
            'Partner',           # Partner Name
            'irrigation',        # Irrigation Type
            'hired_workers',     # Hired Workers
            'area'              # Farm Size
        ]
        
        # Get all SAI assessment indicator columns (266 columns starting with BH-, BP-, etc.)
        sai_prefixes = ['BH-', 'BP-', 'CE-', 'CW-', 'CT-', 'CC-', 'HS-', 'HW-', 'HP-', 'HA-', 'HH-',
                       'OR-', 'OP-', 'OS-', 'OT-', 'OF-', 'OE-', 'OM-', 'OW-',
                       'SR-', 'SW-', 'SF-', 'SC-', 'SM-',
                       'WI-', 'WP-', 'WQ-',
                       'CM-', 'CO-', 'FM-', 'IO-', 'IM-', 'LM-', 'NM-', 'PM-', 'RM-', 'ST-']
        
        # Find all target columns (SAI indicators)
        self.target_columns = []
        for col in self.df.columns:
            col_stripped = col.strip()
            if any(col_stripped.startswith(prefix) for prefix in sai_prefixes):
                if col_stripped not in self.feature_columns:
                    self.target_columns.append(col)
        
        print(f"✓ Identified {len(self.feature_columns)} input features")
        print(f"✓ Identified {len(self.target_columns)} assessment indicators to predict")
        
        return self.df
    
    def prepare_training_data(self):
        """Prepare data for training"""
        print("\n🔄 Preparing training data...")
        
        # Create label encoders for categorical features
        for col in self.feature_columns:
            if self.df[col].dtype == 'object':
                le = LabelEncoder()
                # Handle missing values
                values = self.df[col].fillna('Unknown').astype(str)
                # Ensure 'Unknown' exists in training categories for robust inference
                classes = values.unique().tolist()
                if 'Unknown' not in classes:
                    classes.append('Unknown')
                le.fit(classes)
                self.df[col] = values
                self.df[col + '_encoded'] = le.transform(values)
                self.label_encoders[col] = le
                # Store most frequent value for safe fallback if needed
                self.feature_modes[col] = values.mode(dropna=True)[0]
        
        # Convert area to numeric
        if 'area' in self.feature_columns:
            self.df['area'] = pd.to_numeric(self.df['area'], errors='coerce').fillna(0)
            self.df['area_bucket'] = pd.cut(
                self.df['area'],
                bins=[0, 2, 5, 10, 20, 1000],
                labels=['XS', 'S', 'M', 'L', 'XL'],
                include_lowest=True
            ).astype(str)

        # Normalize Yes/No targets once to avoid repeated normalization
        for col in self.target_columns:
            if col in self.df.columns:
                normalized = self.df[col].map(self._normalize_yes_no)
                if normalized.notna().any():
                    self.df[col] = normalized
        
        print("✓ Data preparation complete")
    
    def train_models(self):
        """Train XGBoost models for each assessment indicator"""
        print("\n🚀 Training XGBoost models...")
        print("This will create one model for each of the 266+ assessment indicators")
        
        # Prepare feature matrix
        X_cols = []
        for col in self.feature_columns:
            if col + '_encoded' in self.df.columns:
                X_cols.append(col + '_encoded')
            else:
                X_cols.append(col)
        
        X = self.df[X_cols].values
        
        trained_count = 0
        rule_count = 0
        skipped_count = 0
        
        # Train a model for each target column
        for idx, target_col in enumerate(self.target_columns, 1):
            try:
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
                
                # Use rule-based predictor for binary Yes/No indicators
                if self._is_binary_yes_no(y):
                    self.models[target_col] = {
                        'type': 'rule_based'
                    }
                    rule_count += 1
                    continue

                # Create label encoder for target
                le_target = LabelEncoder()
                y_encoded = le_target.fit_transform(y)
                
                # Train XGBoost classifier
                model = xgb.XGBClassifier(
                    max_depth=3,
                    n_estimators=50,
                    learning_rate=0.1,
                    objective='multi:softprob',
                    random_state=42,
                    verbosity=0
                )
                
                model.fit(X_valid, y_encoded)
                
                # Store model and encoder
                self.models[target_col] = {
                    'type': 'xgboost',
                    'model': model,
                    'encoder': le_target,
                    'feature_cols': X_cols
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
        print(f"   • Rule-based (binary Yes/No): {rule_count} indicators")
        print(f"   • Skipped (insufficient data): {skipped_count} indicators")
        
        return trained_count
    
    def predict_assessment(self, country, crop, partner, irrigation, hired_workers, area):
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
                    # Handle unseen category - use 'Unknown' if available, else use mode
                    fallback = 'Unknown' if 'Unknown' in le.classes_ else self.feature_modes.get(col, le.classes_[0])
                    encoded_val = le.transform([fallback])[0]
                X_input.append(encoded_val)
            else:
                X_input.append(input_data[col])
        
        X_input = np.array(X_input).reshape(1, -1)
        
        # Predict all assessment indicators
        predictions = {}
        high_confidence_count = 0
        
        for target_col, model_data in self.models.items():
            try:
                if model_data.get('type') == 'rule_based':
                    area_bucket = pd.cut(
                        pd.Series([area]),
                        bins=[0, 2, 5, 10, 20, 1000],
                        labels=['XS', 'S', 'M', 'L', 'XL'],
                        include_lowest=True
                    ).astype(str).iloc[0]
                    value, confidence = self._predict_yes_no_rule(
                        target_col,
                        {
                            'country_code': country,
                            'crop_name': crop,
                            'irrigation': irrigation,
                            'area_bucket': area_bucket
                        }
                    )
                    predictions[target_col] = {
                        'value': value,
                        'confidence': confidence,
                        'method': 'rule_based'
                    }
                    if confidence >= 80:
                        high_confidence_count += 1
                    continue

                model = model_data['model']
                encoder = model_data['encoder']
                
                # Get prediction
                y_pred = model.predict(X_input)[0]
                
                # Get probability scores
                y_proba = model.predict_proba(X_input)[0]
                confidence = float(np.max(y_proba) * 100)
                
                # Decode prediction
                predicted_value = encoder.inverse_transform([int(y_pred)])[0]
                
                predictions[target_col] = {
                    'value': predicted_value,
                    'confidence': confidence,
                    'method': 'xgboost'
                }
                
                if confidence >= 80:
                    high_confidence_count += 1
                    
            except Exception as e:
                predictions[target_col] = {
                    'value': 'Unknown',
                    'confidence': 0.0
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
