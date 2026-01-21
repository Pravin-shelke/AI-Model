# Balaji Framework XGBoost AI Model

AI-powered assessment predictor that reduces 266-question SAI Framework assessments from 15-20 minutes to 2-3 minutes by predicting answers from just 6 inputs.

## Quick Start

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train Model
```bash
# First time setup - trains with original 2 records + synthetic data
python xgboost_balaji_predictor.py

# Or retrain with current data
python retrain_model.py --retrain
```

### 3. Start API Server
```bash
./start_ai_server.sh
```

Server runs on http://localhost:5001

## Core Files (Commit These)

✅ **Python Scripts:**
- `xgboost_balaji_predictor.py` - Main AI predictor
- `flask_api_server.py` - REST API server
- `retrain_model.py` - Retraining tool
- `add_real_data.py` - Add new assessment data
- `generate_training_data.py` - Synthetic data generator

✅ **Integration:**
- `BalajiAIService.ts` - React Native TypeScript service
- `start_ai_server.sh` - Server startup script
- `requirements.txt` - Python dependencies

✅ **Original Data:**
- `Balaji Framework 2025-2026-01-20.csv` - Original 2 assessments

✅ **Documentation:**
- `README.md` - This file
- `INTEGRATION_SUMMARY.md` - Integration guide
- `REACT_NATIVE_INTEGRATION.md` - React Native setup
- `TRAINING_GUIDE.md` - How to improve model
- `ADD_DATA_GUIDE.md` - Adding real data

## Large Files (NOT in Git)

❌ **Don't Commit:**
- `xgboost_balaji_models.pkl` (40+ MB) - Regenerate with training
- `Balaji_Framework_Training_Data.csv` - Generated from original + new data
- `*_backup_*.pkl` - Backup files
- Other CSV exports

## How to Use on Another Machine

1. Clone the repo
2. Run setup: `pip install -r requirements.txt`
3. Generate synthetic data: `python generate_training_data.py`
4. Train model: `python xgboost_balaji_predictor.py`
5. Start server: `./start_ai_server.sh`

## API Endpoints

- `GET /health` - Health check
- `POST /api/v1/predict` - Predict single assessment
- `POST /api/v1/predict/batch` - Batch predictions
- `GET /api/v1/indicators` - List trained indicators

## Current Status

- **Trained Models:** 120 indicators (skipped 147 with insufficient data)
- **Training Data:** 237 records (2 real + 235 synthetic/real)
- **Average Confidence:** 65%+
- **High Confidence (≥80%):** 36+ predictions

## Adding More Training Data

```bash
# Add real assessment CSV and retrain
python add_real_data.py "your_export.csv"
python retrain_model.py --retrain
./start_ai_server.sh
```

## Model Quality Improvements

With 154 real Demo Partner assessments added:
- Training records: 52 → 237
- Only predicts indicators with >30% data availability
- Requires minimum 10 samples per indicator
- Skips indicators with poor quality data
- No more "Unknown" predictions
