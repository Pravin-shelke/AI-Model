"""
Flask API Server for XGBoost Balaji Framework AI
Serves predictions to React Native app
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from xgboost_balaji_predictor import XGBoostBalajiPredictor
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React Native

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load AI model at startup
predictor = XGBoostBalajiPredictor()
logger.info("Loading XGBoost Balaji AI models...")
predictor.load_models('xgboost_balaji_models.pkl')
logger.info(" Models loaded successfully!")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Balaji Framework XGBoost AI',
        'models_loaded': len(predictor.models),
        'version': '1.0.0'
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


if __name__ == '__main__':
    PORT = 5001  # Using 5001 to avoid macOS AirPlay on port 5000
    
    print("=" * 70)
    print("  🚀 Starting Balaji Framework XGBoost AI Server")
    print("=" * 70)
    print(f"\n  📍 Server will run on: http://localhost:{PORT}")
    print(f"  🔗 Health Check: http://localhost:{PORT}/health")
    print(f"  🤖 Prediction API: http://localhost:{PORT}/api/v1/predict")
    print(f"  📦 Batch API: http://localhost:{PORT}/api/v1/predict/batch")
    print("\n" + "=" * 70 + "\n")
    
    # Run Flask server
    app.run(host='0.0.0.0', port=PORT, debug=False)
