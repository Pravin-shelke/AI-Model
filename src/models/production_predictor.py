"""
PRODUCTION-READY PREDICTOR
Includes confidence thresholds, fallback mechanisms, and monitoring
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import logging


class ProductionPredictor:
    """
    Production-ready predictor with:
    - Confidence thresholds
    - Fallback mechanisms  
    - Prediction monitoring
    - Quality assurance
    """
    
    def __init__(self, models_path, validation_path, min_confidence=70):
        self.models = {}
        self.validation_results = {}
        self.min_confidence = min_confidence
        self.prediction_log = []
        
        # Load models
        try:
            with open(models_path, 'rb') as f:
                self.models = pickle.load(f)
            print(f"✓ Loaded {len(self.models)} production models")
        except FileNotFoundError:
            print(f"❌ Models file not found: {models_path}")
            raise
        
        # Load validation results
        try:
            with open(validation_path, 'r') as f:
                self.validation_results = json.load(f)
            print(f"✓ Loaded validation results for {len(self.validation_results)} indicators")
        except FileNotFoundError:
            print(f"⚠️  Validation results not found, using default confidence")
            self.validation_results = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('prediction_monitoring.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ProductionPredictor')
    
    def predict_with_confidence(self, input_data):
        """
        Make predictions with confidence-based decisions
        
        Returns:
        - high_confidence: predictions to show to user
        - low_confidence: questions user should answer manually
        - no_model: indicators without trained models
        """
        results = {
            'high_confidence': [],
            'low_confidence': [],
            'no_model': [],
            'metadata': {
                'prediction_id': datetime.now().isoformat(),
                'total_indicators': len(self.models),
                'input_data': input_data
            }
        }
        
        # Prepare features
        X = self._prepare_features(input_data)
        
        for indicator, model in self.models.items():
            try:
                # Get prediction
                prediction = model.predict(X)[0]
                prediction_proba = model.predict_proba(X)[0]
                max_proba = max(prediction_proba)
                
                # Get model's validation confidence
                model_confidence = self.validation_results.get(indicator, {}).get('confidence', 0)
                
                # Combined confidence score
                # Weight both model quality and prediction certainty
                combined_confidence = (model_confidence * 0.5) + (max_proba * 100 * 0.5)
                
                prediction_data = {
                    'indicator': indicator,
                    'predicted_value': prediction,
                    'prediction_confidence': max_proba * 100,
                    'model_confidence': model_confidence,
                    'combined_confidence': combined_confidence,
                    'test_accuracy': self.validation_results.get(indicator, {}).get('test_accuracy', 0),
                    'is_overfitting': self.validation_results.get(indicator, {}).get('is_overfitting', False)
                }
                
                # Decision logic based on combined confidence
                if combined_confidence >= self.min_confidence and not prediction_data['is_overfitting']:
                    results['high_confidence'].append(prediction_data)
                    decision = 'accept'
                else:
                    results['low_confidence'].append(prediction_data)
                    decision = 'manual'
                
                # Log prediction
                self._log_prediction(indicator, prediction, combined_confidence, decision)
                
            except Exception as e:
                self.logger.error(f"Prediction failed for {indicator}: {str(e)}")
                results['no_model'].append({
                    'indicator': indicator,
                    'error': str(e)
                })
        
        # Add all indicators without models
        all_indicators = set(self.validation_results.keys())
        trained_indicators = set(self.models.keys())
        untrained = all_indicators - trained_indicators
        
        for indicator in untrained:
            results['no_model'].append({
                'indicator': indicator,
                'reason': 'not_trained'
            })
        
        # Update metadata
        results['metadata']['high_confidence_count'] = len(results['high_confidence'])
        results['metadata']['low_confidence_count'] = len(results['low_confidence'])
        results['metadata']['no_model_count'] = len(results['no_model'])
        results['metadata']['coverage'] = len(results['high_confidence']) / (len(self.models) + len(untrained)) * 100
        
        return results
    
    def _prepare_features(self, input_data):
        """Prepare input features for prediction"""
        # This should match the training feature preparation
        feature_columns = ['country_code', 'crop_name', 'Partner', 'irrigation', 'hired_workers', 'area']
        
        # Create feature vector
        X = pd.DataFrame([input_data])[feature_columns]
        
        # TODO: Apply same encoding as training (load encoders)
        
        return X
    
    def _log_prediction(self, indicator, prediction, confidence, decision):
        """Log predictions for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'indicator': indicator,
            'prediction': prediction,
            'confidence': confidence,
            'decision': decision
        }
        
        self.prediction_log.append(log_entry)
        
        # Log to file
        self.logger.info(f"Prediction: {indicator} -> {prediction} (conf: {confidence:.1f}%, decision: {decision})")
    
    def get_user_feedback(self, prediction_id, indicator, predicted_value, actual_value):
        """
        Record user feedback for monitoring model performance
        
        This allows tracking:
        - Prediction accuracy in production
        - Model drift over time
        - Which indicators need retraining
        """
        feedback = {
            'prediction_id': prediction_id,
            'timestamp': datetime.now().isoformat(),
            'indicator': indicator,
            'predicted_value': predicted_value,
            'actual_value': actual_value,
            'was_correct': predicted_value == actual_value
        }
        
        # Log feedback
        self.logger.info(f"Feedback: {indicator} - Predicted: {predicted_value}, Actual: {actual_value}, Correct: {feedback['was_correct']}")
        
        # Save to feedback file for retraining
        feedback_file = 'models/prediction_feedback.jsonl'
        with open(feedback_file, 'a') as f:
            f.write(json.dumps(feedback) + '\n')
        
        return feedback
    
    def generate_monitoring_report(self):
        """Generate monitoring report from prediction logs"""
        if not self.prediction_log:
            return {'message': 'No predictions logged yet'}
        
        df = pd.DataFrame(self.prediction_log)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_predictions': len(df),
            'average_confidence': df['confidence'].mean(),
            'high_confidence_rate': (df['decision'] == 'accept').mean(),
            'by_indicator': df.groupby('indicator').agg({
                'confidence': ['mean', 'std', 'count'],
                'decision': lambda x: (x == 'accept').mean()
            }).to_dict()
        }
        
        return report
    
    def check_model_drift(self, feedback_file='models/prediction_feedback.jsonl'):
        """
        Check for model drift by analyzing prediction feedback
        
        Returns indicators that may need retraining
        """
        try:
            # Load feedback
            feedback_data = []
            with open(feedback_file, 'r') as f:
                for line in f:
                    feedback_data.append(json.loads(line))
            
            df = pd.DataFrame(feedback_data)
            
            # Calculate accuracy by indicator
            indicator_accuracy = df.groupby('indicator').agg({
                'was_correct': ['mean', 'count']
            })
            
            # Flag indicators with low accuracy
            drift_threshold = 0.7  # <70% accuracy indicates drift
            min_samples = 10
            
            drifted_indicators = []
            for indicator, row in indicator_accuracy.iterrows():
                accuracy = row[('was_correct', 'mean')]
                count = row[('was_correct', 'count')]
                
                if count >= min_samples and accuracy < drift_threshold:
                    drifted_indicators.append({
                        'indicator': indicator,
                        'production_accuracy': accuracy,
                        'sample_count': count,
                        'action': 'needs_retraining'
                    })
            
            return {
                'total_feedback': len(df),
                'indicators_analyzed': len(indicator_accuracy),
                'drifted_indicators': drifted_indicators,
                'recommendation': 'Retrain models for drifted indicators' if drifted_indicators else 'No drift detected'
            }
            
        except FileNotFoundError:
            return {'message': 'No feedback data available yet'}


def main():
    """Demo production predictor"""
    import sys
    import os
    
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    models_path = os.path.join(base_dir, 'models', 'production_models.pkl')
    validation_path = os.path.join(base_dir, 'models', 'validation_results.json')
    
    # Initialize predictor
    predictor = ProductionPredictor(
        models_path=models_path,
        validation_path=validation_path,
        min_confidence=70
    )
    
    # Test prediction
    test_input = {
        'country_code': 'US',
        'crop_name': 'Potato',
        'Partner': 'Test Partner',
        'irrigation': 'irrigated',
        'hired_workers': 'Yes',
        'area': 100.0
    }
    
    print("\n" + "=" * 70)
    print("🧪 Testing Production Predictor")
    print("=" * 70)
    
    results = predictor.predict_with_confidence(test_input)
    
    print(f"\n📊 Prediction Results:")
    print(f"  ✅ High confidence predictions: {results['metadata']['high_confidence_count']}")
    print(f"  ⚠️  Low confidence (manual): {results['metadata']['low_confidence_count']}")
    print(f"  ❌ No model available: {results['metadata']['no_model_count']}")
    print(f"  📈 Coverage: {results['metadata']['coverage']:.1f}%")
    
    if results['high_confidence']:
        print(f"\n  Top 5 High Confidence Predictions:")
        for pred in sorted(results['high_confidence'], 
                          key=lambda x: x['combined_confidence'], 
                          reverse=True)[:5]:
            print(f"    • {pred['indicator']}: {pred['predicted_value']} "
                  f"(confidence: {pred['combined_confidence']:.1f}%)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
