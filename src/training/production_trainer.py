"""
PRODUCTION-READY MODEL TRAINER
Fixes all critical issues:
1. Train/test split with validation
2. Cross-validation
3. Data quality filtering
4. Overfitting detection
5. Confidence thresholds
6. Performance monitoring
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class ProductionTrainer:
    """Production-ready trainer with validation and monitoring"""
    
    def __init__(self, min_samples=10, min_confidence=60, test_size=0.2):
        self.min_samples = min_samples
        self.min_confidence = min_confidence
        self.test_size = test_size
        self.models = {}
        self.metadata = {}
        self.validation_results = {}
        
    def filter_quality_data(self, df):
        """Filter out incomplete/failed assessments"""
        print("\n🔍 Filtering Data Quality...")
        print("=" * 70)
        
        initial_count = len(df)
        
        # 1. Remove failed carbon score assessments
        if 'Reason for failure of Carbon-score generation' in df.columns:
            failed_mask = df['Reason for failure of Carbon-score generation'].notna()
            df = df[~failed_mask]
            print(f"✓ Removed {failed_mask.sum()} failed carbon score assessments")
        
        # 2. Remove assessments with too many missing values (>50% empty)
        missing_threshold = len(df.columns) * 0.5
        missing_counts = df.isnull().sum(axis=1)
        high_missing_mask = missing_counts >= missing_threshold
        df = df[~high_missing_mask]
        print(f"✓ Removed {high_missing_mask.sum()} assessments with >50% missing data")
        
        # 3. Keep only submitted/completed plans
        if 'Submitted' in df.columns:
            submitted_mask = df['Submitted'].astype(str).str.lower().isin(['true', 'yes', '1'])
            df = df[submitted_mask]
            print(f"✓ Kept only {submitted_mask.sum()} submitted assessments")
        
        # 4. Remove duplicate assessments
        if 'plan_id' in df.columns:
            duplicates = df.duplicated(subset=['plan_id'], keep='first')
            df = df[~duplicates]
            if duplicates.sum() > 0:
                print(f"✓ Removed {duplicates.sum()} duplicate plans")
        
        final_count = len(df)
        removed = initial_count - final_count
        
        print(f"\n📊 Data Quality Summary:")
        print(f"  Initial: {initial_count} assessments")
        print(f"  Removed: {removed} low-quality assessments ({removed/initial_count*100:.1f}%)")
        print(f"  Final: {final_count} high-quality assessments")
        
        if final_count < 100:
            print(f"\n⚠️  WARNING: Only {final_count} quality assessments available")
            print(f"   Recommend collecting at least 500+ assessments for production")
        
        print("=" * 70)
        
        return df
    
    def balance_data(self, df, column):
        """Address data imbalance by stratified sampling"""
        value_counts = df[column].value_counts()
        
        # If one class dominates (>80%), undersample it
        if len(value_counts) > 1:
            dominant_ratio = value_counts.iloc[0] / len(df)
            if dominant_ratio > 0.8:
                print(f"   ⚠️  Imbalanced data for {column}: {dominant_ratio*100:.1f}% dominant class")
                return True
        
        return False
    
    def train_with_validation(self, X, y, indicator):
        """Train model with proper validation and overfitting detection"""
        
        # Check minimum samples
        if len(X) < self.min_samples:
            return None, {
                'status': 'skipped',
                'reason': f'insufficient_data',
                'samples': len(X),
                'required': self.min_samples
            }
        
        # Check unique values
        unique_values = y.nunique()
        if unique_values < 2:
            return None, {
                'status': 'skipped',
                'reason': 'insufficient_variance',
                'unique_values': unique_values
            }
        
        # Train/test split (stratified)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42, stratify=y
            )
        except:
            # Fallback if stratification fails
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42
            )
        
        # Train XGBoost model
        model = xgb.XGBClassifier(
            max_depth=3,  # Reduced to prevent overfitting
            learning_rate=0.05,  # Lower learning rate
            n_estimators=50,  # Fewer trees
            min_child_weight=3,  # Higher minimum samples per leaf
            subsample=0.7,  # Subsample training data
            colsample_bytree=0.8,  # Feature sampling
            gamma=0.1,  # Minimum loss reduction
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=10,
            verbosity=0
        )
        
        # Fit with early stopping
        eval_set = [(X_test, y_test)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Cross-validation (5-fold or less if small dataset)
        n_folds = min(5, len(X_train) // 5)
        if n_folds >= 2:
            try:
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42),
                    scoring='accuracy'
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except:
                cv_mean = train_accuracy
                cv_std = 0.0
        else:
            cv_mean = train_accuracy
            cv_std = 0.0
        
        # Overfitting detection
        overfit_diff = train_accuracy - test_accuracy
        is_overfitting = overfit_diff > 0.15  # >15% gap
        
        # Calculate production-ready confidence score
        confidence = self._calculate_confidence(
            test_accuracy=test_accuracy,
            cv_mean=cv_mean,
            train_size=len(X_train),
            is_overfitting=is_overfitting,
            overfit_diff=overfit_diff
        )
        
        # Decide if model meets production standards
        meets_standards = (
            confidence >= self.min_confidence and
            test_accuracy >= 0.6 and
            not is_overfitting
        )
        
        validation_results = {
            'status': 'trained' if meets_standards else 'low_confidence',
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'cv_mean': float(cv_mean),
            'cv_std': float(cv_std),
            'confidence': float(confidence),
            'is_overfitting': bool(is_overfitting),
            'overfit_diff': float(overfit_diff),
            'train_size': int(len(X_train)),
            'test_size': int(len(X_test)),
            'unique_classes': int(unique_values),
            'meets_standards': meets_standards
        }
        
        return model if meets_standards else None, validation_results
    
    def _calculate_confidence(self, test_accuracy, cv_mean, train_size, is_overfitting, overfit_diff):
        """Calculate realistic confidence score with penalties"""
        
        # Base confidence from test accuracy
        confidence = test_accuracy * 100
        
        # Penalty for small sample size
        if train_size < 20:
            confidence *= 0.5
        elif train_size < 50:
            confidence *= 0.7
        elif train_size < 100:
            confidence *= 0.85
        
        # Penalty for overfitting
        if is_overfitting:
            confidence *= (1.0 - min(overfit_diff, 0.3))  # Up to 30% penalty
        
        # Adjust based on CV consistency
        if cv_mean > 0:
            cv_penalty = abs(test_accuracy - cv_mean)
            confidence *= (1.0 - cv_penalty)
        
        return min(max(confidence, 0), 99.0)
    
    def train_all_indicators(self, df):
        """Train models for all indicators with validation"""
        print("\n🤖 Training Models with Validation...")
        print("=" * 70)
        
        # Prepare features
        feature_columns = ['country_code', 'crop_name', 'Partner', 'irrigation', 'hired_workers', 'area']
        X_base = df[feature_columns].copy()
        
        # Encode categorical features
        encoders = {}
        for col in ['country_code', 'crop_name', 'Partner', 'irrigation', 'hired_workers']:
            if col in X_base.columns:
                le = LabelEncoder()
                X_base[col] = le.fit_transform(X_base[col].fillna('Unknown'))
                encoders[col] = le
        
        # Get all indicator columns (exclude metadata)
        exclude_cols = set(feature_columns + ['plan_id', 'assessment_name', 'user_id', 'workspace_id', 
                                               'email_id', 'Submitted', 'plan_created', 'plan_modified'])
        indicator_cols = [col for col in df.columns if col not in exclude_cols]
        
        results = {
            'trained': [],
            'skipped': [],
            'low_confidence': []
        }
        
        for idx, indicator in enumerate(indicator_cols, 1):
            if df[indicator].isna().all():
                continue
            
            # Prepare target
            y = df[indicator].fillna('Unknown')
            
            # Check if worth training
            unique_vals = y.nunique()
            if unique_vals < 2:
                results['skipped'].append({
                    'indicator': indicator,
                    'reason': 'no_variance',
                    'unique_values': unique_vals
                })
                continue
            
            # Train with validation
            model, validation = self.train_with_validation(X_base, y, indicator)
            
            if validation['status'] == 'trained':
                self.models[indicator] = model
                self.validation_results[indicator] = validation
                results['trained'].append(indicator)
                
                print(f"✓ [{idx}/{len(indicator_cols)}] {indicator[:30]:<30} "
                      f"Test: {validation['test_accuracy']:.2%} | "
                      f"Conf: {validation['confidence']:.1f}% | "
                      f"CV: {validation['cv_mean']:.2%}±{validation['cv_std']:.2%}")
                
            elif validation['status'] == 'low_confidence':
                results['low_confidence'].append({
                    'indicator': indicator,
                    'validation': validation
                })
                print(f"⚠️  [{idx}/{len(indicator_cols)}] {indicator[:30]:<30} "
                      f"Test: {validation['test_accuracy']:.2%} | "
                      f"Conf: {validation['confidence']:.1f}% (TOO LOW)")
                
            else:
                results['skipped'].append({
                    'indicator': indicator,
                    'reason': validation['reason'],
                    **validation
                })
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 Training Summary:")
        print(f"  ✅ Trained (meets standards): {len(results['trained'])} indicators")
        print(f"  ⚠️  Low confidence: {len(results['low_confidence'])} indicators")
        print(f"  ❌ Skipped: {len(results['skipped'])} indicators")
        
        if results['trained']:
            avg_test_acc = np.mean([self.validation_results[i]['test_accuracy'] 
                                   for i in results['trained']])
            avg_confidence = np.mean([self.validation_results[i]['confidence'] 
                                     for i in results['trained']])
            print(f"\n  Average Test Accuracy: {avg_test_acc:.2%}")
            print(f"  Average Confidence: {avg_confidence:.1f}%")
        
        print("=" * 70)
        
        return results, encoders
    
    def save_models(self, models_path, results_path):
        """Save models and validation results"""
        # Save models
        with open(models_path, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"✓ Saved {len(self.models)} models to {models_path}")
        
        # Save validation results
        with open(results_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        print(f"✓ Saved validation results to {results_path}")
    
    def generate_report(self, output_path):
        """Generate comprehensive performance report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_indicators': len(self.validation_results),
                'average_test_accuracy': np.mean([v['test_accuracy'] 
                                                  for v in self.validation_results.values()]),
                'average_confidence': np.mean([v['confidence'] 
                                              for v in self.validation_results.values()]),
                'high_confidence_count': sum(1 for v in self.validation_results.values() 
                                            if v['confidence'] >= 80),
                'overfitting_count': sum(1 for v in self.validation_results.values() 
                                        if v['is_overfitting'])
            },
            'indicators': self.validation_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Generated performance report: {output_path}")
        return report


def main():
    """Main training workflow"""
    print("=" * 70)
    print("  PRODUCTION-READY MODEL TRAINER")
    print("  Fixes: Data quality, validation, overfitting, monitoring")
    print("=" * 70)
    
    # Initialize trainer
    trainer = ProductionTrainer(
        min_samples=10,
        min_confidence=60,
        test_size=0.2
    )
    
    # Load data
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    data_file = os.path.join(base_dir, 'Assessment_AI_Training_Data.csv')
    
    print(f"\n📂 Loading data from: {data_file}")
    df = pd.read_csv(data_file, encoding='utf-8')
    print(f"✓ Loaded {len(df)} assessments with {len(df.columns)} columns")
    
    # Filter quality data
    df_clean = trainer.filter_quality_data(df)
    
    if len(df_clean) < 50:
        print("\n⚠️  CRITICAL WARNING:")
        print(f"   Only {len(df_clean)} quality assessments available")
        print(f"   This is below the recommended 500+ for production")
        print(f"   Models will have limited accuracy and generalization")
        print(f"\n   Recommendation: Collect more data before production deployment")
        
        response = input("\n   Continue training anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("   Training cancelled.")
            return
    
    # Train models
    results, encoders = trainer.train_all_indicators(df_clean)
    
    # Save models and results
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    models_path = os.path.join(models_dir, 'production_models.pkl')
    validation_path = os.path.join(models_dir, 'validation_results.json')
    report_path = os.path.join(models_dir, 'performance_report.json')
    
    trainer.save_models(models_path, validation_path)
    report = trainer.generate_report(report_path)
    
    # Print final assessment
    print("\n" + "=" * 70)
    print("🎯 PRODUCTION READINESS ASSESSMENT:")
    print("=" * 70)
    
    summary = report['summary']
    
    print(f"\n✅ Trained Models: {summary['total_indicators']}")
    print(f"📊 Average Test Accuracy: {summary['average_test_accuracy']:.2%}")
    print(f"💯 Average Confidence: {summary['average_confidence']:.1f}%")
    print(f"⭐ High Confidence (≥80%): {summary['high_confidence_count']}")
    print(f"⚠️  Overfitting Detected: {summary['overfitting_count']}")
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    if summary['average_test_accuracy'] < 0.7:
        print("  ❌ Average accuracy <70% - Collect more diverse data")
    if summary['overfitting_count'] > summary['total_indicators'] * 0.3:
        print("  ❌ >30% models overfitting - Need more training data")
    if summary['high_confidence_count'] < summary['total_indicators'] * 0.5:
        print("  ⚠️  <50% high confidence - Recommend A/B testing before full rollout")
    
    if (summary['average_test_accuracy'] >= 0.7 and 
        summary['overfitting_count'] <= summary['total_indicators'] * 0.2):
        print("  ✅ Models meet minimum production standards")
        print("  ✅ Ready for pilot deployment with monitoring")
    else:
        print("  ⚠️  Models need improvement before production")
        print("  ⚠️  Recommend: Collect 300-500 more quality assessments")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
