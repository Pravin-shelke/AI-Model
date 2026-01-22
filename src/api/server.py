import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
from models.predictor import AssessmentAIPredictor
import logging
import json

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')

predictor = AssessmentAIPredictor()
predictor.load_models(os.path.join(MODELS_DIR, 'assessment_ai_models.pkl'))


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Assessment AI Predictor',
        'models_loaded': len(predictor.models)
    })


@app.route('/api/v1/predict', methods=['POST'])
def predict_assessment():
    try:
        data = request.get_json()
        required = ['country', 'crop', 'partner', 'irrigation', 'hired_workers', 'area']
        missing = [f for f in required if f not in data]
        
        if missing:
            return jsonify({'error': 'Missing fields', 'missing': missing}), 400
        
        predictions = predictor.predict_assessment(
            country=data['country'],
            crop=data['crop'],
            partner=data['partner'],
            irrigation=data['irrigation'],
            hired_workers=data['hired_workers'],
            area=float(data['area'])
        )
        
        confidences = [p['confidence'] for p in predictions.values()]
        stats = {
            'total_indicators': len(predictions),
            'high_confidence': sum(1 for c in confidences if c >= 80),
            'medium_confidence': sum(1 for c in confidences if 60 <= c < 80),
            'low_confidence': sum(1 for c in confidences if c < 60),
            'average_confidence': round(sum(confidences) / len(confidences), 2)
        }
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'statistics': stats,
            'metadata': data
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/v1/predict/batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        
        if 'assessments' not in data:
            return jsonify({'error': 'Expected "assessments" array'}), 400
        
        results = []
        for idx, item in enumerate(data['assessments']):
            try:
                preds = predictor.predict_assessment(
                    country=item['country'],
                    crop=item['crop'],
                    partner=item['partner'],
                    irrigation=item['irrigation'],
                    hired_workers=item['hired_workers'],
                    area=float(item['area'])
                )
                results.append({'index': idx, 'success': True, 'predictions': preds})
            except Exception as e:
                results.append({'index': idx, 'success': False, 'error': str(e)})
        
        return jsonify({'success': True, 'total': len(data['assessments']), 'results': results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/v1/indicators', methods=['GET'])
def get_indicators():
    try:
        indicators = sorted(list(predictor.models.keys()))
        return jsonify({
            'success': True,
            'total': len(indicators),
            'indicators': indicators
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/v1/questions', methods=['GET'])
def get_questions():
    try:
        config_file = os.path.join(CONFIG_DIR, 'questions_config.json')
        if not os.path.exists(config_file):
            return jsonify({'success': False, 'error': 'Config file not found'}), 404
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        all_questions = []
        for category in config['categories']:
            for question in category['questions']:
                question['categoryId'] = category['categoryId']
                question['categoryName'] = category['categoryName']
                question['displayId'] = category['displayId']
                question['aiAvailable'] = question['indicatorCode'] in predictor.models
                
                if question['aiAvailable']:
                    model = predictor.models[question['indicatorCode']]
                    question['aiTrainingSamples'] = len(model['encoder'].classes_)
                
                all_questions.append(question)
        
        return jsonify({
            'success': True,
            'total_questions': len(all_questions),
            'total_categories': len(config['categories']),
            'version': config.get('version', '1.0'),
            'lastUpdated': config.get('lastUpdated'),
            'questions': all_questions,
            'categories': config['categories']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/v1/questions/<indicator_code>', methods=['GET'])
def get_question_details(indicator_code):
    try:
        with open(os.path.join(CONFIG_DIR, 'questions_config.json'), 'r') as f:
            config = json.load(f)
        
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
            return jsonify({'success': False, 'error': 'Question not found'}), 404
        
        question['aiAvailable'] = indicator_code in predictor.models
        if question['aiAvailable']:
            model = predictor.models[indicator_code]
            question['aiInfo'] = {
                'trained': True,
                'possibleValues': model['encoder'].classes_.tolist(),
                'trainingSamples': len(model['encoder'].classes_)
            }
        else:
            question['aiInfo'] = {'trained': False}
        
        return jsonify({'success': True, 'question': question})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/v1/indicators', methods=['GET'])
def get_trained_indicators():
    try:
        trained = sorted(list(predictor.models.keys()))
        return jsonify({
            'success': True,
            'total_trained': len(trained),
            'indicators': trained
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    PORT = 5001
    print(f"\nStarting server on http://localhost:{PORT}")
    print(f"Endpoints:")
    print(f"  - GET  /health")
    print(f"  - POST /api/v1/predict")
    print(f"  - POST /api/v1/predict/batch")
    print(f"  - GET  /api/v1/questions")
    print(f"  - GET  /api/v1/indicators\n")
    
    app.run(host='0.0.0.0', port=PORT, debug=False)
