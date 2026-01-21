"""
Flask API Server for XGBoost Balaji Framework AI
Serves predictions to React Native app
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
from models.predictor import AssessmentAIPredictor
import logging
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React Native

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')

# Load AI model at startup
predictor = AssessmentAIPredictor()
logger.info("Loading Assessment AI models...")
predictor.load_models(os.path.join(MODELS_DIR, 'assessment_ai_models.pkl'))
logger.info(" Models loaded successfully!")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Assessment AI - SAI Framework Predictor',
        'models_loaded': len(predictor.models),
        'version': '2.0.0'
    })


@app.route('/api/v1/predict', methods=['POST'])
def predict_assessment():
    """
    Predict assessment indicators based on 6 inputs
    
    Request Body:
    {
        "country": "IN",
        "crop": "Potato",
        "partner": "Balaji ",
        "irrigation": "irrigated",
        "hired_workers": "Yes",
        "area": 10.0
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['country', 'crop', 'partner', 'irrigation', 'hired_workers', 'area']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing': missing_fields
            }), 400
        
        # Extract inputs
        country = data['country']
        crop = data['crop']
        partner = data['partner']
        irrigation = data['irrigation']
        hired_workers = data['hired_workers']
        area = float(data['area'])
        
        logger.info(f"Predicting for: {country}, {crop}, {partner}, {irrigation}, {hired_workers}, {area}")
        
        # Get predictions
        predictions = predictor.predict_assessment(
            country=country,
            crop=crop,
            partner=partner,
            irrigation=irrigation,
            hired_workers=hired_workers,
            area=area
        )
        
        # Calculate statistics
        high_confidence = sum(1 for p in predictions.values() if p['confidence'] >= 80)
        medium_confidence = sum(1 for p in predictions.values() if 60 <= p['confidence'] < 80)
        low_confidence = sum(1 for p in predictions.values() if p['confidence'] < 60)
        avg_confidence = sum(p['confidence'] for p in predictions.values()) / len(predictions)
        
        # Format response
        response = {
            'success': True,
            'predictions': predictions,
            'statistics': {
                'total_indicators': len(predictions),
                'high_confidence': high_confidence,
                'medium_confidence': medium_confidence,
                'low_confidence': low_confidence,
                'average_confidence': round(avg_confidence, 2)
            },
            'metadata': {
                'country': country,
                'crop': crop,
                'partner': partner,
                'irrigation': irrigation,
                'hired_workers': hired_workers,
                'area': area
            }
        }
        
        logger.info(f"✅ Predictions complete: {len(predictions)} indicators")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict assessments for multiple farmers
    
    Request Body:
    {
        "assessments": [
            {
                "country": "IN",
                "crop": "Potato",
                "partner": "Balaji ",
                "irrigation": "irrigated",
                "hired_workers": "Yes",
                "area": 10.0
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if 'assessments' not in data or not isinstance(data['assessments'], list):
            return jsonify({
                'error': 'Invalid request format. Expected "assessments" array'
            }), 400
        
        results = []
        
        for idx, assessment in enumerate(data['assessments']):
            try:
                predictions = predictor.predict_assessment(
                    country=assessment['country'],
                    crop=assessment['crop'],
                    partner=assessment['partner'],
                    irrigation=assessment['irrigation'],
                    hired_workers=assessment['hired_workers'],
                    area=float(assessment['area'])
                )
                
                results.append({
                    'index': idx,
                    'success': True,
                    'predictions': predictions
                })
            except Exception as e:
                results.append({
                    'index': idx,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total': len(data['assessments']),
            'results': results
        }), 200
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/indicators', methods=['GET'])
def get_indicators():
    """Get list of all assessment indicators that can be predicted"""
    try:
        indicators = list(predictor.models.keys())
        return jsonify({
            'success': True,
            'total': len(indicators),
            'indicators': indicators
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/questions', methods=['GET'])
def get_questions():
    """
    Get all available questions with metadata
    Loads from questions_config.json instead of hardcoding
    
    Response:
    {
        "success": true,
        "total_questions": 266,
        "total_categories": 30,
        "questions": [...],
        "categories": [...]
    }
    """
    try:
        # Load questions config
        config_file = os.path.join(CONFIG_DIR, 'questions_config.json')
        if not os.path.exists(config_file):
            return jsonify({
                'success': False,
                'error': 'Questions config file not found'
            }), 404
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Flatten questions for easy access
        all_questions = []
        for category in config['categories']:
            for question in category['questions']:
                question['categoryId'] = category['categoryId']
                question['categoryName'] = category['categoryName']
                question['displayId'] = category['displayId']
                
                # Add AI availability info
                question['aiAvailable'] = question['indicatorCode'] in predictor.models
                if question['aiAvailable']:
                    model_info = predictor.models[question['indicatorCode']]
                    question['aiTrainingSamples'] = len(model_info['encoder'].classes_)
                
                all_questions.append(question)
        
        return jsonify({
            'success': True,
            'total_questions': len(all_questions),
            'total_categories': len(config['categories']),
            'version': config.get('version', '1.0'),
            'lastUpdated': config.get('lastUpdated'),
            'questions': all_questions,
            'categories': config['categories']
        }), 200
        
    except Exception as e:
        logger.error(f"Error loading questions: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/questions/<indicator_code>', methods=['GET'])
def get_question_details(indicator_code):
    """
    Get details for a specific indicator/question
    
    Response includes AI training statistics if available
    """
    try:
        # Load questions config
        with open(os.path.join(CONFIG_DIR, 'questions_config.json'), 'r') as f:
            config = json.load(f)
        
        # Find the question
        question = None
        for category in config['categories']:
            for q in category['questions']:
                if q['indicatorCode'] == indicator_code:
                    question = q.copy()
                    question['categoryId'] = category['categoryId']
                    question['categoryName'] = category['categoryName']
                    question['displayId'] = category['displayId']
                    break
            if question:
                break
        
        if not question:
            return jsonify({
                'success': False,
                'error': f'Question {indicator_code} not found'
            }), 404
        
        # Add AI model info if available
        question['aiAvailable'] = indicator_code in predictor.models
        if question['aiAvailable']:
            model_info = predictor.models[indicator_code]
            question['aiInfo'] = {
                'trained': True,
                'possibleValues': model_info['encoder'].classes_.tolist(),
                'trainingSamples': len(model_info['encoder'].classes_)
            }
        else:
            question['aiInfo'] = {
                'trained': False,
                'reason': 'Insufficient training data (needs >30% data availability and 10+ samples)'
            }
        
        return jsonify({
            'success': True,
            'question': question
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting question details: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/indicators', methods=['GET'])
def get_trained_indicators():
    """
    Get list of all indicators that have trained AI models
    
    Response:
    {
        "success": true,
        "total_trained": 120,
        "total_possible": 266,
        "indicators": ["BH-1", "BH-2", ...]
    }
    """
    try:
        trained_indicators = list(predictor.models.keys())
        
        return jsonify({
            'success': True,
            'total_trained': len(trained_indicators),
            'indicators': sorted(trained_indicators)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    PORT = 5001  # Using 5001 to avoid macOS AirPlay on port 5000
    
    print("=" * 70)
    print("  🚀 Starting Assessment AI Server")
    print("=" * 70)
    print(f"\n  📍 Server will run on: http://localhost:{PORT}")
    print(f"  🔗 Health Check: http://localhost:{PORT}/health")
    print(f"  🤖 Prediction API: http://localhost:{PORT}/api/v1/predict")
    print(f"  📦 Batch API: http://localhost:{PORT}/api/v1/predict/batch")
    print(f"  📋 Questions API: http://localhost:{PORT}/api/v1/questions")
    print(f"  🔍 Indicator Details: http://localhost:{PORT}/api/v1/questions/<code>")
    print(f"  📊 Trained Models: http://localhost:{PORT}/api/v1/indicators")
    print("\n" + "=" * 70 + "\n")
    
    # Run Flask server
    app.run(host='0.0.0.0', port=PORT, debug=False)
