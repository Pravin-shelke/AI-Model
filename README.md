# Assessment AI - SAI Framework Predictor

AI-powered assessment tool that reduces 266-question SAI Framework assessments from 15-20 minutes to 2-3 minutes.

## 📁 Project Structure

```
AI-Model/
├── src/                    # Source code
│   ├── api/               # Flask REST API
│   │   └── server.py      # Main API server
│   ├── models/            # AI prediction models
│   │   └── predictor.py   # XGBoost predictor
│   ├── training/          # Training & data management
│   │   ├── trainer.py     # Model retraining
│   │   ├── data_loader.py # Add new training data
│   │   └── data_generator.py # Synthetic data generation
│   └── utils/             # Helper utilities
│
├── data/                  # Data files
│   ├── training/         # Training datasets
│   ├── original/         # Original source data
│   └── exports/          # User assessment exports
│
├── models/               # Saved model files
│   └── assessment_ai_models.pkl
│
├── config/              # Configuration files
│   └── questions_config.json
│
├── tests/               # Test scripts
│   └── test_api.py
│
├── scripts/             # Utility scripts
│   ├── start_server.sh  # Start API server
│   └── demo.py          # Demo script
│
├── client/              # React Native integration
│   └── AssessmentAIService.ts
│
├── docs/                # Documentation
│   ├── README.md
│   ├── IMPROVEMENT_PLAN.md
│   └── ...
│
├── .gitignore
├── requirements.txt
└── README.md (this file)
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start API Server
```bash
python src/api/server.py
```

Server runs on http://localhost:5001

### 3. Test API
```bash
curl http://localhost:5001/health
```

## 📊 Current Status

- **Training Data:** 237 assessments
- **Trained Models:** 176 indicators (66% coverage)
- **Average Confidence:** 77.9%
- **High Confidence (≥80%):** 100 predictions
- **Time Savings:** 85% reduction (15-20 min → 2-3 min)

## 🔧 Common Tasks

### Retrain Model
```bash
cd src/training
python trainer.py --retrain
```

### Add New Training Data
```bash
cd src/training
python data_loader.py "path/to/export.csv"
python trainer.py --retrain
```

### Run Tests
```bash
cd tests
python test_api.py
```

### Run Demo
```bash
cd scripts
python demo.py
```

## 📦 API Endpoints

- `GET /health` - Health check
- `POST /api/v1/predict` - Predict single assessment
- `POST /api/v1/predict/batch` - Batch predictions
- `GET /api/v1/questions` - Get all questions (dynamic loading)
- `GET /api/v1/questions/<code>` - Get specific question details
- `GET /api/v1/indicators` - List trained indicators

## 🔗 React Native Integration

Copy the TypeScript service to your React Native app:
```bash
cp client/AssessmentAIService.ts ../mobile-app/src/services/ML/
```

## 📚 Documentation

See `docs/` folder for:
- Integration guides
- Training guides
- Improvement plans
- API documentation

## 🎯 How It Works

1. **User inputs 6 fields**: country, crop, partner, irrigation, workers, area
2. **AI predicts 176 answers** based on patterns from 237 real assessments
3. **App auto-fills** high-confidence predictions
4. **User reviews** and answers remaining 90 questions
5. **Result:** 85% faster completion time

## 🔄 Continuous Improvement

With more assessment data:
- Current: 237 assessments → 176 predictions (77.9% confidence)
- Goal: 500+ assessments → 220+ predictions (85%+ confidence)

Export completed assessments monthly and retrain for better accuracy!
