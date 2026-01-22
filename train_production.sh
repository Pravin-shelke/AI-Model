#!/bin/bash

echo "=============================================="
echo "  PRODUCTION-READY MODEL TRAINING"
echo "  Fixing all critical issues..."
echo "=============================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  No virtual environment found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade required packages
echo ""
echo "📦 Installing required packages..."
pip install --quiet --upgrade pip
pip install --quiet pandas numpy scikit-learn xgboost scipy

echo ""
echo "✅ Environment ready!"
echo ""

# Run production trainer
echo "=============================================="
echo "  STEP 1: Training with Validation"
echo "=============================================="
echo ""

python src/training/production_trainer.py

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "  ✅ TRAINING COMPLETE"
    echo "=============================================="
    echo ""
    echo "📁 Generated files:"
    echo "   • models/production_models.pkl"
    echo "   • models/validation_results.json"
    echo "   • models/performance_report.json"
    echo ""
    echo "📊 Next steps:"
    echo "   1. Review: models/performance_report.json"
    echo "   2. Test predictions: python src/models/production_predictor.py"
    echo "   3. Setup A/B testing: python tests/ab_testing.py"
    echo ""
    echo "⚠️  IMPORTANT: Collect 500+ quality assessments for production!"
    echo ""
else
    echo ""
    echo "=============================================="
    echo "  ❌ TRAINING FAILED"
    echo "=============================================="
    echo ""
    echo "Common issues:"
    echo "   • Check if Assessment_AI_Training_Data.csv exists"
    echo "   • Verify CSV format and encoding"
    echo "   • Check for sufficient data (need at least 50+ rows)"
    echo ""
    exit 1
fi
