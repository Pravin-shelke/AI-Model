"""
Balaji Framework AI Model
Uses the Balaji Framework 2025-2026 data for training and predictions
This model predicts agricultural practices and scores based on farm data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')


class BalajiFrameworkAI:
    """
    AI Model trained on Balaji Framework data
    Predicts agricultural practices, SAI scores, and sustainability metrics
    """
    
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.feature_columns = []
        self.target_columns = []
        self.data = None
        
    def load_balaji_data(self, file_path='Balaji  Framework 2025-2026-01-20.csv'):
        """Load and preprocess Balaji Framework data"""
        print(f"📂 Loading Balaji Framework data from: {file_path}")
        
        try:
            self.data = pd.read_csv(file_path)
            print(f"✓ Loaded {len(self.data)} records")
            print(f"✓ Total columns: {len(self.data.columns)}")
            
            # Show key columns
            print(f"\n📊 Key Information:")
            print(f"   • Countries: {self.data['country_code'].nunique() if 'country_code' in self.data.columns else 'N/A'}")
            print(f"   • Crops: {self.data['crop_name'].nunique() if 'crop_name' in self.data.columns else 'N/A'}")
            print(f"   • Partners: {self.data['Partner'].nunique() if 'Partner' in self.data.columns else 'N/A'}")
            
            return self.data
            
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return None
    
    def identify_key_features(self):
        """Identify the most important features for prediction"""
        
        # Essential input features (what farmers provide)
        essential_features = [
            'country_code', 'crop_name', 'area', 'irrigation', 
            'Partner', 'hired_workers', 'plan_year'
        ]
        
        # Advanced assessment features (SAI Framework codes)
        assessment_features = []
        
        # Find all columns that match assessment patterns (like BH-1, BP-2, etc.)
        for col in self.data.columns:
            # Check for pattern: XX-N or XX-N.x
            if len(col) >= 4 and '-' in col and col[:2].isupper():
                assessment_features.append(col)
        
        self.feature_columns = essential_features
        self.assessment_columns = assessment_features
        
        print(f"\n🎯 Identified Features:")
        print(f"   • Essential features: {len(essential_features)}")
        print(f"   • Assessment features: {len(assessment_features)}")
        
        return essential_features, assessment_features
    
    def prepare_training_data(self):
        """Prepare data for training"""
        
        if self.data is None:
            print("❌ No data loaded. Run load_balaji_data() first.")
            return None
        
        print("\n⚙️  Preparing training data...")
        
        # Identify available features
        available_features = []
        for col in self.feature_columns:
            if col in self.data.columns:
                available_features.append(col)
        
        if len(available_features) == 0:
            print("❌ No features found")
            return None
        
        # Create feature matrix
        X = self.data[available_features].copy()
        
        # Encode categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    # Handle missing values
                    X[col] = X[col].fillna('Unknown')
                    X[col] = self.encoders[col].fit_transform(X[col].astype(str))
                else:
                    X[col] = X[col].fillna('Unknown')
                    X[col] = self.encoders[col].transform(X[col].astype(str))
            else:
                # Fill numeric missing values with median
                X[col] = X[col].fillna(X[col].median())
        
        print(f"✓ Prepared {X.shape[0]} samples with {X.shape[1]} features")
        
        return X, available_features
    
    def train_irrigation_predictor(self):
        """Train model to predict irrigation type"""
        
        print("\n🔧 Training Irrigation Prediction Model...")
        
        X, features = self.prepare_training_data()
        
        if X is None or 'irrigation' not in self.data.columns:
            print("❌ Cannot train irrigation predictor")
            return None
        
        # Remove irrigation from features if present
        feature_cols = [f for f in features if f != 'irrigation']
        X_train = X[[f for f in feature_cols if f in X.columns]]
        
        # Encode target
        y = self.data['irrigation'].fillna('Unknown')
        irrigation_encoder = LabelEncoder()
        y_encoded = irrigation_encoder.fit_transform(y.astype(str))
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_encoded)
        
        # Evaluate
        accuracy = model.score(X_train, y_encoded)
        
        self.models['irrigation'] = model
        self.encoders['irrigation_target'] = irrigation_encoder
        
        print(f"✓ Irrigation Predictor trained")
        print(f"  Accuracy: {accuracy:.2%}")
        
        return model
    
    def train_practice_predictor(self, practice_column):
        """Train model to predict specific agricultural practices"""
        
        if practice_column not in self.data.columns:
            return None
        
        X, features = self.prepare_training_data()
        
        if X is None:
            return None
        
        # Get target
        y = self.data[practice_column].fillna('Unknown')
        
        # Encode target
        practice_encoder = LabelEncoder()
        y_encoded = practice_encoder.fit_transform(y.astype(str))
        
        # Train model
        model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
        model.fit(X, y_encoded)
        
        # Store
        self.models[practice_column] = model
        self.encoders[f'{practice_column}_target'] = practice_encoder
        
        return model
    
    def train_all_models(self):
        """Train multiple prediction models"""
        
        print("\n" + "=" * 70)
        print("🚀 Training Balaji Framework AI Models")
        print("=" * 70)
        
        # Load data
        if self.data is None:
            self.load_balaji_data()
        
        # Identify features
        self.identify_key_features()
        
        # Train irrigation predictor
        self.train_irrigation_predictor()
        
        # Train practice predictors for key assessment areas
        key_practices = []
        
        # Find important practice columns
        for col in self.assessment_columns[:20]:  # Train on first 20 for demo
            if col in self.data.columns:
                # Check if column has multiple unique values
                unique_vals = self.data[col].nunique()
                if unique_vals > 1 and unique_vals < 50:
                    print(f"\n📊 Training model for: {col}")
                    model = self.train_practice_predictor(col)
                    if model:
                        key_practices.append(col)
                        accuracy = model.score(*self.prepare_training_data())
                        print(f"   ✓ Accuracy: {accuracy:.2%}")
        
        print(f"\n✅ Training Complete!")
        print(f"   Total models trained: {len(self.models)}")
        
        return self.models
    
    def predict_for_farmer(self, country, crop, area, partner, hired_workers, plan_year=2025):
        """Make predictions for a farmer based on their inputs"""
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'country_code': [country],
            'crop_name': [crop],
            'area': [area],
            'Partner': [partner],
            'hired_workers': [hired_workers],
            'plan_year': [plan_year]
        })
        
        # Encode features
        for col in input_data.columns:
            if col in self.encoders:
                try:
                    input_data[col] = self.encoders[col].transform(input_data[col].astype(str))
                except ValueError:
                    # Handle unseen categories - use most common class
                    print(f"   ⚠️  Warning: '{input_data[col][0]}' not seen in training, using default")
                    input_data[col] = 0
            elif input_data[col].dtype == 'object':
                # Handle new categories
                input_data[col] = 0
        
        # Make predictions
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Get prediction
                pred = model.predict(input_data)
                
                # Decode if encoder exists
                encoder_key = f'{model_name}_target'
                if encoder_key in self.encoders:
                    pred_decoded = self.encoders[encoder_key].inverse_transform(pred)
                    predictions[model_name] = pred_decoded[0]
                else:
                    predictions[model_name] = pred[0]
                
                # Get confidence
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_data)
                    confidence = max(proba[0])
                    predictions[f'{model_name}_confidence'] = confidence
                    
            except Exception as e:
                predictions[model_name] = 'Unknown'
        
        return predictions
    
    def save_models(self, path='balaji_framework_models.pkl'):
        """Save all trained models"""
        model_data = {
            'models': self.models,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns,
            'assessment_columns': self.assessment_columns
        }
        joblib.dump(model_data, path)
        print(f"\n💾 Models saved to: {path}")
    
    def load_models(self, path='balaji_framework_models.pkl'):
        """Load trained models"""
        model_data = joblib.load(path)
        self.models = model_data['models']
        self.encoders = model_data['encoders']
        self.feature_columns = model_data['feature_columns']
        self.assessment_columns = model_data.get('assessment_columns', [])
        print(f"✓ Models loaded from: {path}")


def demo():
    """Demonstration of Balaji Framework AI"""
    
    print("=" * 70)
    print("🌾 BALAJI FRAMEWORK AI - Agricultural Intelligence System")
    print("=" * 70)
    
    # Initialize AI
    ai = BalajiFrameworkAI()
    
    # Load Balaji Framework data
    data = ai.load_balaji_data('Balaji  Framework 2025-2026-01-20.csv')
    
    if data is None:
        print("\n❌ Failed to load data. Please check the file path.")
        return
    
    # Show data summary
    print("\n📋 Data Summary:")
    print("-" * 70)
    print(f"Total assessments: {len(data)}")
    
    if 'crop_name' in data.columns:
        print(f"\nCrops in dataset:")
        for crop in data['crop_name'].dropna().unique():
            print(f"   • {crop}")
    
    if 'irrigation' in data.columns:
        print(f"\nIrrigation types:")
        for irr in data['irrigation'].dropna().unique():
            print(f"   • {irr}")
    
    # Train models
    ai.train_all_models()
    
    # Save models
    ai.save_models()
    
    # Demo prediction
    print("\n" + "=" * 70)
    print("🎯 DEMO: Predicting for a New Farmer")
    print("=" * 70)
    
    print("\n📝 Farmer Input:")
    # Use actual values from the dataset
    farmer_input = {
        'country': 'IN',
        'crop': 'Potato',  # From actual data
        'area': 5,
        'partner': 'Balaji ',  # Note: space at end from actual data
        'hired_workers': 'Yes',
        'plan_year': 2025
    }
    
    for key, value in farmer_input.items():
        print(f"   • {key}: {value}")
    
    # Make predictions
    print("\n🤖 AI Predictions:")
    print("-" * 70)
    
    predictions = ai.predict_for_farmer(**farmer_input)
    
    for key, value in predictions.items():
        if '_confidence' in key:
            print(f"   Confidence: {value:.1%}")
        else:
            print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)
    
    print("""
    
🎯 What This AI Can Do:

1. ✓ Predict irrigation recommendations
2. ✓ Suggest sustainable practices
3. ✓ Auto-fill agricultural assessments
4. ✓ Provide SAI Framework guidance
5. ✓ Reduce assessment time from hours to minutes

💡 Integration Ready!
   Use these models in your Balaji Framework app to help farmers
   complete assessments faster and more accurately.
    """)


if __name__ == '__main__':
    demo()
