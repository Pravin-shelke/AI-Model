"""
Enhanced Assessment AI Predictor with Question Type Support
Handles CheckBox, Radio, TextField, and Conditional questions
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pickle
import json
import os


class EnhancedAssessmentAI:
    def __init__(self):
        self.models = {}
        self.question_metadata = {}
        self.conditional_rules = {}
        
    def load_question_metadata(self, metadata_file):
        """Load question types, options, and conditional rules"""
        with open(metadata_file, 'r') as f:
            data = json.load(f)
            
        for q in data['questions']:
            self.question_metadata[q['indicatorCode']] = q
            
            # Build conditional rules map
            if q.get('conditional') and q.get('showIf'):
                parent = q['showIf']['indicator']
                if parent not in self.conditional_rules:
                    self.conditional_rules[parent] = []
                self.conditional_rules[parent].append({
                    'child': q['indicatorCode'],
                    'showIfValue': q['showIf']['value']
                })
    
    def predict_with_context(self, inputs):
        """
        Predict answers with question context awareness
        
        Returns predictions grouped by type:
        - radio: Single selection
        - checkbox: Multiple selections
        - text: AI suggests, user refines
        - conditional: Only if parent condition met
        """
        base_predictions = self._get_base_predictions(inputs)
        
        results = {
            'radio': [],
            'checkbox': [],
            'text': [],
            'conditional': [],
            'skipped': []
        }
        
        # Process each prediction with context
        for indicator, prediction in base_predictions.items():
            if indicator not in self.question_metadata:
                continue
                
            q_meta = self.question_metadata[indicator]
            q_type = q_meta['type']
            
            # Check if conditional question should be shown
            if q_meta.get('conditional'):
                parent_indicator = q_meta['showIf']['indicator']
                required_value = q_meta['showIf']['value']
                
                # Only include if parent has matching value
                if parent_indicator in base_predictions:
                    parent_value = base_predictions[parent_indicator]['value']
                    if parent_value != required_value:
                        results['skipped'].append({
                            'indicator': indicator,
                            'reason': f'Condition not met: {parent_indicator} != {required_value}'
                        })
                        continue
            
            # Validate prediction against allowed options
            if q_type in ['radio', 'checkbox']:
                if prediction['value'] not in q_meta.get('options', []):
                    # Invalid prediction, skip it
                    results['skipped'].append({
                        'indicator': indicator,
                        'reason': f'Predicted value not in allowed options'
                    })
                    continue
            
            # Add to appropriate category
            results[q_type].append({
                'indicator': indicator,
                'value': prediction['value'],
                'confidence': prediction['confidence'],
                'displayId': q_meta['displayId'],
                'outcomeId': q_meta['outcomeId'],
                'outcomeName': q_meta['outcomeName'],
                'description': q_meta['description'],
                'options': q_meta.get('options', []),
                'required': q_meta.get('required', False)
            })
        
        return results
    
    def _get_base_predictions(self, inputs):
        """Get raw predictions from trained models"""
        # This calls your existing prediction logic
        # Return format: {'BH-1': {'value': 'Yes', 'confidence': 85}, ...}
        pass
    
    def get_conditional_questions(self, answered_questions):
        """
        Determine which conditional questions should be shown
        based on already answered questions
        """
        show_questions = []
        
        for parent, rules in self.conditional_rules.items():
            if parent in answered_questions:
                parent_value = answered_questions[parent]
                
                for rule in rules:
                    if parent_value == rule['showIfValue']:
                        show_questions.append(rule['child'])
        
        return show_questions
    
    def validate_prediction(self, indicator, predicted_value):
        """Validate if predicted value is allowed for this question"""
        if indicator not in self.question_metadata:
            return False, "Question not found"
        
        q_meta = self.question_metadata[indicator]
        q_type = q_meta['type']
        
        if q_type in ['radio', 'checkbox']:
            if predicted_value not in q_meta.get('options', []):
                return False, f"Value '{predicted_value}' not in allowed options"
        
        if q_type == 'text':
            max_len = q_meta.get('maxLength', 1000)
            if len(str(predicted_value)) > max_len:
                return False, f"Text exceeds max length of {max_len}"
        
        return True, "Valid"
    
    def group_by_outcome(self, predictions):
        """Group predictions by outcome/category for UI display"""
        grouped = {}
        
        for q_type, questions in predictions.items():
            for q in questions:
                outcome_id = q['outcomeId']
                if outcome_id not in grouped:
                    grouped[outcome_id] = {
                        'outcomeId': outcome_id,
                        'outcomeName': q['outcomeName'],
                        'displayId': q['displayId'],
                        'questions': []
                    }
                grouped[outcome_id]['questions'].append(q)
        
        return list(grouped.values())


# Example usage
if __name__ == "__main__":
    ai = EnhancedAssessmentAI()
    
    # Load question metadata
    ai.load_question_metadata('../../config/questions_metadata.json')
    
    # Make predictions with context
    inputs = {
        'country': 'US',
        'crop': 'Potato',
        'partner': 'Kellanova',
        'irrigation': 'irrigated',
        'hired_workers': 'No',
        'area': 44
    }
    
    predictions = ai.predict_with_context(inputs)
    
    print(f"Radio questions: {len(predictions['radio'])}")
    print(f"Checkbox questions: {len(predictions['checkbox'])}")
    print(f"Text questions: {len(predictions['text'])}")
    print(f"Conditional questions: {len(predictions['conditional'])}")
    print(f"Skipped (invalid/conditional): {len(predictions['skipped'])}")
