# AI Model Improvement Plan

## 1. Accuracy Improvements

### A. More Training Data (Most Important!)
**Current:** 237 records (154 real + 83 synthetic)
**Target:** 500+ real assessments

**Actions:**
- Export 300+ completed assessments from production
- Ensure data quality (no "Unknown", "NO TEXT")
- Add weekly: `python add_real_data.py "weekly_export.csv"`
- Retrain monthly as data grows

**Expected Impact:** 65% → 85%+ confidence

### B. Better Features (Add More Context)
**Current:** 6 inputs (country, crop, partner, irrigation, workers, area)
**Add:**
- Farm certification (Organic, GlobalGAP, etc.)
- Previous SAI score (if available)
- Region/state (more granular than country)
- Crop category (grain, vegetable, fruit)
- Farm size category (small/medium/large)
- Season/planting date
- Soil type
- Previous year's yield

**Implementation:**
```python
# In xgboost_balaji_predictor.py, add more essential_features:
essential_features = [
    'country_code', 'crop_name', 'Partner', 'irrigation', 
    'hired_workers', 'area',
    'certification_type',  # NEW
    'region',               # NEW
    'crop_category',        # NEW
    'soil_type'            # NEW
]
```

### C. Model Tuning (Optimize XGBoost)
**Current:** Basic XGBoost settings
**Improve:**
```python
model = xgb.XGBClassifier(
    max_depth=5,              # Was 3, try 4-6
    n_estimators=100,         # Was 50, more trees = better
    learning_rate=0.05,       # Was 0.1, slower = more accurate
    subsample=0.8,            # Use 80% data per tree
    colsample_bytree=0.8,     # Use 80% features per tree
    min_child_weight=3,       # Prevent overfitting
    gamma=0.1,                # Regularization
    objective='multi:softmax',
    random_state=42,
    verbosity=0
)
```

### D. Ensemble Models (Combine Multiple Models)
Train RandomForest + XGBoost + LightGBM, take majority vote:
- Higher accuracy
- More robust predictions
- Better confidence scores

### E. Question Dependency Learning
Some questions depend on others:
- If BH-1 = "No" → BH-2, BH-3, BH-4 likely "No"
- If irrigation = "rainfed" → Water questions different

**Implementation:** Add previous question answers as features

### F. Confidence Threshold Optimization
**Current:** Fixed 80% threshold for "high confidence"
**Improve:** Dynamic thresholds per indicator based on:
- Training data quality
- Question complexity
- Historical accuracy

## 2. Code Quality Improvements

### A. Dynamic Question Loading (Remove Hardcoding)

**Current Problem:** Questions hardcoded in React Native
**Solution:** Load from API

**Create new API endpoint:**

```python
# In flask_api_server.py, add:

@app.route('/api/v1/questions', methods=['GET'])
def get_questions():
    """Get all available questions with metadata"""
    try:
        # Load from database or JSON config file
        questions = load_questions_config()
        
        return jsonify({
            'success': True,
            'total_questions': len(questions),
            'questions': questions,
            'categories': group_by_category(questions)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def load_questions_config():
    """Load questions from config file instead of hardcoding"""
    import json
    with open('questions_config.json', 'r') as f:
        return json.load(f)
```

**Create questions config file:**

```json
// questions_config.json
{
  "categories": [
    {
      "categoryId": "CM",
      "categoryName": "Community Management",
      "displayId": "7.1.1",
      "questions": [
        {
          "indicatorCode": "CM-1.a",
          "description": "Safety training",
          "type": "radio",
          "required": true,
          "options": ["Yes", "No", "Partial"]
        },
        {
          "indicatorCode": "CM-1.b",
          "description": "Training records kept",
          "type": "radio",
          "required": true,
          "options": ["Yes", "No"]
        }
      ]
    }
  ]
}
```

### B. Better Architecture

```
AI-Model/
├── src/
│   ├── models/
│   │   ├── xgboost_predictor.py
│   │   ├── random_forest_predictor.py
│   │   └── ensemble_predictor.py
│   ├── api/
│   │   ├── routes/
│   │   │   ├── prediction.py
│   │   │   ├── questions.py
│   │   │   └── training.py
│   │   └── server.py
│   ├── data/
│   │   ├── loader.py
│   │   ├── validator.py
│   │   └── preprocessor.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── optimizer.py
│   └── config/
│       ├── questions_config.json
│       ├── model_config.yaml
│       └── api_config.yaml
├── tests/
│   ├── test_models.py
│   ├── test_api.py
│   └── test_data.py
└── requirements.txt
```

### C. Add Question Metadata API

```python
@app.route('/api/v1/indicators/<indicator_code>', methods=['GET'])
def get_indicator_details(indicator_code):
    """Get details for specific indicator"""
    indicator = questions_db.get(indicator_code)
    
    # Include AI training stats
    model_stats = get_model_stats(indicator_code)
    
    return jsonify({
        'indicatorCode': indicator_code,
        'displayId': indicator['displayId'],
        'description': indicator['description'],
        'category': indicator['category'],
        'options': indicator['options'],
        'aiAvailable': indicator_code in trained_models,
        'aiConfidence': model_stats['avg_confidence'],
        'trainingSamples': model_stats['sample_count']
    })
```

### D. Batch Question Loading

```typescript
// BalajiAIService.ts - Improve
class BalajiAIService {
  private questionsCache: Map<string, Question> | null = null;
  
  async loadQuestions(forceRefresh = false): Promise<Question[]> {
    if (this.questionsCache && !forceRefresh) {
      return Array.from(this.questionsCache.values());
    }
    
    const response = await axios.get(`${this.baseURL}/api/v1/questions`);
    this.questionsCache = new Map(
      response.data.questions.map((q: Question) => [q.indicatorCode, q])
    );
    
    return response.data.questions;
  }
  
  async predictWithQuestionMetadata(inputs: FarmerInputs) {
    const [predictions, questions] = await Promise.all([
      this.predictAssessment(inputs),
      this.loadQuestions()
    ]);
    
    // Merge predictions with question metadata
    return predictions.map(pred => ({
      ...pred,
      questionMetadata: this.questionsCache?.get(pred.indicator)
    }));
  }
}
```

### E. Add Caching Layer

```python
from functools import lru_cache
from datetime import datetime, timedelta

# Cache predictions for same inputs (5 min)
@lru_cache(maxsize=1000)
def cached_predict(country, crop, partner, irrigation, workers, area, timestamp):
    return predictor.predict_assessment(
        country, crop, partner, irrigation, workers, area
    )

@app.route('/api/v1/predict', methods=['POST'])
def predict_assessment():
    data = request.json
    
    # Round timestamp to 5-min intervals for cache hits
    cache_key = datetime.now().replace(second=0, microsecond=0)
    cache_key = cache_key - timedelta(minutes=cache_key.minute % 5)
    
    predictions = cached_predict(
        data['country'],
        data['crop'],
        data['partner'],
        data['irrigation'],
        data['hired_workers'],
        data['area'],
        cache_key.timestamp()
    )
    
    return jsonify(predictions)
```

### F. Add Monitoring & Analytics

```python
# Track prediction usage
@app.route('/api/v1/predict', methods=['POST'])
def predict_assessment():
    start_time = time.time()
    
    try:
        predictions = predictor.predict_assessment(...)
        
        # Log analytics
        log_prediction_event({
            'timestamp': datetime.now(),
            'inputs': data,
            'predictions_count': len(predictions),
            'high_confidence_count': sum(1 for p in predictions if p['confidence'] >= 80),
            'avg_confidence': np.mean([p['confidence'] for p in predictions]),
            'response_time_ms': (time.time() - start_time) * 1000
        })
        
        return jsonify(predictions)
    except Exception as e:
        log_error_event({'error': str(e), 'inputs': data})
        raise
```

## 3. Quick Wins (Implement These First)

### Week 1:
1. ✅ Export 100+ more real assessments
2. ✅ Add to training: `python add_real_data.py "export.csv"`
3. ✅ Retrain with more data

### Week 2:
1. Create `questions_config.json`
2. Add `/api/v1/questions` endpoint
3. Update React Native to load questions dynamically

### Week 3:
1. Add caching to API
2. Improve XGBoost hyperparameters
3. Add monitoring/logging

### Week 4:
1. Add more features (certification, region)
2. Implement ensemble models
3. Add A/B testing framework

## Expected Results

**After Week 1 (More Data):**
- Confidence: 65% → 75%
- Trained indicators: 120 → 180+

**After Week 2 (Dynamic Questions):**
- Code maintainability: ⭐⭐ → ⭐⭐⭐⭐
- Question updates: Manual → Automatic

**After Week 3 (Optimization):**
- API response time: ~500ms → ~100ms
- Confidence: 75% → 80%

**After Week 4 (Advanced Features):**
- Confidence: 80% → 85%+
- Trained indicators: 180 → 220+
- Production ready ✅
