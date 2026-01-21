# 📚 XGBoost Model Training Guide
## How to Improve Your AI Model with More Data

---

## 🎯 Current Model Status

Your model is currently trained on:
- **52 assessment records** (2 real + 50 synthetic)
- **266 XGBoost classifiers**
- **Average confidence: 60-70%**

To improve accuracy, you need **more real assessment data**!

---

## 📈 Training Strategy

### Phase 1: Current (Testing Phase)
- ✅ 52 records (synthetic data)
- ✅ 60-70% average confidence
- ✅ Good for development/testing
- ⚠️ NOT recommended for production

### Phase 2: Production Ready
- 🎯 **Target: 100+ real assessments**
- 📊 Expected: 75-85% average confidence
- ✅ Safe for production use

### Phase 3: High Accuracy
- 🎯 **Target: 500+ real assessments**
- 📊 Expected: 85-95% average confidence
- ✅ Excellent for production

### Phase 4: Expert System
- 🎯 **Target: 1000+ real assessments**
- 📊 Expected: 90-95% average confidence
- ✅ Best possible accuracy

---

## 🔄 How to Add More Training Data

### Method 1: Export from Your App (Recommended)

#### Step 1: Export Completed Assessments

Add export functionality to your React Native app:

```typescript
// In your assessment completion screen
import { exportAssessmentToCSV } from '@/utils/exportUtils';

const handleExportForTraining = async () => {
  const csvData = await exportAssessmentToCSV(completedAssessment);
  
  // Save to device or send to server
  await RNFS.writeFile(
    `${RNFS.DocumentDirectoryPath}/assessment_${Date.now()}.csv`,
    csvData,
    'utf8'
  );
  
  Alert.alert('Exported!', 'Assessment data saved for AI training');
};
```

#### Step 2: Collect CSV Files

- Ask farmers to complete assessments normally
- Export completed assessments to CSV
- Collect CSV files (via email, cloud storage, etc.)
- Combine multiple exports into one file

#### Step 3: Add to Training Data

```bash
# Copy your new assessments CSV to AI-Model folder
cp ~/Downloads/new_assessments.csv /Users/pravinshelke/Documents/AI-Model/

# Add to training data and retrain
cd /Users/pravinshelke/Documents/AI-Model
python retrain_model.py --add new_assessments.csv
```

---

### Method 2: Manual CSV Creation

If you have assessments in other formats (Excel, database, etc.):

#### Step 1: Export to CSV

Your CSV must have these columns (same as Balaji Framework):
- `country_code` (e.g., 'IN', 'US', 'BR')
- `crop_name` (e.g., 'Potato', 'Corn', 'Soybean')
- `Partner` (e.g., 'Balaji ', 'Balaji-East')
- `irrigation` (e.g., 'irrigated', 'rainfed')
- `hired_workers` ('Yes' or 'No')
- `area` (numeric, farm size)
- Plus all 266 SAI assessment indicators (BH-1, BP-2, etc.)

#### Step 2: Format Requirements

```csv
country_code,crop_name,Partner,irrigation,hired_workers,area,BH-2,BH-3,...
IN,Potato,Balaji ,irrigated,Yes,10.0,NO,NO,...
US,Corn,Syngenta USA,rainfed,No,50.0,YES,NO,...
```

#### Step 3: Validate Format

```bash
# Check if your CSV has correct format
python -c "
import pandas as pd
df = pd.read_csv('your_new_data.csv')
print(f'Rows: {len(df)}')
print(f'Columns: {len(df.columns)}')
print(f'Required columns present: {all(c in df.columns for c in [\"country_code\", \"crop_name\", \"Partner\", \"irrigation\", \"hired_workers\", \"area\"])}')
"
```

---

### Method 3: Generate More Synthetic Data (Testing Only)

For development/testing when you don't have real data yet:

```bash
cd /Users/pravinshelke/Documents/AI-Model

# Generate 100 synthetic samples
python retrain_model.py --generate 100

# Or generate 200 samples
python retrain_model.py --generate 200
```

⚠️ **Note:** Synthetic data is good for testing but NOT as accurate as real data!

---

## 🚀 Retraining the Model

### Option 1: Add New Data and Retrain (Recommended)

```bash
cd /Users/pravinshelke/Documents/AI-Model

# Add new CSV file and automatically retrain
python retrain_model.py --add new_assessments.csv
```

This will:
1. ✅ Load your new assessments
2. ✅ Combine with existing training data
3. ✅ Backup old models
4. ✅ Train new models
5. ✅ Test model accuracy
6. ✅ Save updated models

### Option 2: Retrain with Existing Data

If you just want to retrain with current data:

```bash
python retrain_model.py --retrain
```

### Option 3: Interactive Mode

```bash
python retrain_model.py
```

Then follow the prompts:
1. Generate synthetic data
2. Retrain with current data
3. Exit

---

## 📊 Monitoring Model Performance

### After Retraining, Check:

```bash
# Run demo to see predictions
python demo_xgboost.py

# Check model statistics
python -c "
import joblib
models = joblib.load('xgboost_balaji_models.pkl')
print(f'Total models trained: {len(models[\"models\"])}')
print(f'Feature columns: {models[\"feature_columns\"]}')
print(f'Target indicators: {len(models[\"target_columns\"])}')
"
```

### Key Metrics to Track:

1. **Average Confidence** - Should improve with more data
   - Current: 60-70%
   - Target: 80%+

2. **High Confidence Predictions** (≥80%)
   - Current: 50-70 indicators
   - Target: 150+ indicators

3. **Prediction Accuracy** - Test against known assessments
   - Calculate: % of predictions matching actual values

---

## 🔄 Recommended Training Schedule

### Development Phase (Now)
- Train with synthetic data
- Test with developers
- Collect first 10-20 real assessments

### Beta Phase (Month 1-2)
- Collect 50-100 real assessments
- Retrain weekly
- Monitor accuracy

### Production Phase (Month 3+)
- Collect 100+ real assessments
- Retrain monthly
- Track farmer feedback

### Continuous Improvement
- Add new assessments continuously
- Retrain quarterly
- Version control your models

---

## 🎯 Best Practices

### 1. Data Quality

✅ **DO:**
- Use complete assessments (all 266 fields filled)
- Include diverse crops (Potato, Corn, Soybean, etc.)
- Include diverse regions (IN, US, BR, etc.)
- Include diverse partners
- Verify data accuracy before adding

❌ **DON'T:**
- Add incomplete assessments (many null values)
- Add duplicate assessments
- Add fake/test data to production training set

### 2. Data Diversity

Try to collect assessments with variety in:
- **Countries:** IN, US, BR, CN, MX
- **Crops:** Potato, Corn, Soybean, Wheat, Rice, Cotton
- **Irrigation:** irrigated, rainfed, drip, sprinkler
- **Farm Sizes:** Small (1-5 acres), Medium (5-20), Large (20+)
- **Partners:** Different regions and organizations

### 3. Version Control

Keep track of your models:

```bash
# Before retraining, note current performance
echo "$(date): Retraining with $(wc -l < Balaji_Framework_Training_Data.csv) records" >> training_log.txt

# After retraining, test and log results
python demo_xgboost.py >> training_log.txt

# Backups are automatically created with timestamps
ls -lh xgboost_balaji_models_backup_*.pkl
```

### 4. A/B Testing

When deploying a new model:
1. Keep old model as fallback
2. Test new model with 10% of users
3. Compare feedback and accuracy
4. Roll out to 100% if better

---

## 🛠️ Advanced: Custom Training

### Adjust Model Parameters

Edit `xgboost_balaji_predictor.py`:

```python
# Line ~170 - XGBoost parameters
model = xgb.XGBClassifier(
    max_depth=5,           # Increase for more complex patterns (default: 3)
    n_estimators=100,      # More trees = better accuracy (default: 50)
    learning_rate=0.1,     # Learning speed (default: 0.1)
    min_child_weight=2,    # Minimum samples per leaf (default: 1)
    subsample=0.8,         # Use 80% of data per tree
    colsample_bytree=0.8,  # Use 80% of features per tree
    random_state=42,
    verbosity=0
)
```

**More data → Use higher values for better accuracy**

### Train on Specific Indicators

If some indicators are more important:

```python
# In xgboost_balaji_predictor.py, add priority training
priority_indicators = ['BH-2', 'BP-2', 'CE-2', 'WI-2', 'NM-5']

for target_col in self.target_columns:
    if target_col in priority_indicators:
        # Use more trees for important indicators
        model = xgb.XGBClassifier(n_estimators=200, max_depth=7)
    else:
        model = xgb.XGBClassifier(n_estimators=50, max_depth=3)
```

---

## 📝 Training Checklist

Before retraining:
- [ ] Have new assessment data (CSV format)
- [ ] Data has correct column format
- [ ] Data is validated and complete
- [ ] Backed up current models (automatic)
- [ ] Noted current model performance

After retraining:
- [ ] Test with demo: `python demo_xgboost.py`
- [ ] Check average confidence improved
- [ ] Restart Flask server: `./start_ai_server.sh`
- [ ] Test in React Native app
- [ ] Monitor farmer feedback

---

## 🚨 Troubleshooting

### Problem: "No improvement after retraining"

**Solution:** Check data quality
```bash
# Check for duplicate data
python -c "
import pandas as pd
df = pd.read_csv('Balaji_Framework_Training_Data.csv')
print(f'Total records: {len(df)}')
print(f'Unique records: {df.drop_duplicates().shape[0]}')
print(f'Duplicates: {len(df) - df.drop_duplicates().shape[0]}')
"
```

### Problem: "Training takes too long"

**Solution:** Reduce number of trees
```python
# In xgboost_balaji_predictor.py
model = xgb.XGBClassifier(n_estimators=30)  # Reduce from 50
```

### Problem: "Low confidence on specific crops"

**Solution:** Need more data for that crop
```bash
# Check data distribution
python -c "
import pandas as pd
df = pd.read_csv('Balaji_Framework_Training_Data.csv')
print(df['crop_name'].value_counts())
"
```

Add more assessments for crops with low counts.

---

## 📞 Summary

**To train with more data:**

1. **Collect real assessments** from your app
2. **Export to CSV** (same format as Balaji Framework)
3. **Add and retrain:**
   ```bash
   python retrain_model.py --add new_data.csv
   ```
4. **Restart server:**
   ```bash
   ./start_ai_server.sh
   ```
5. **Test improvements** in your app

**Target:** 100+ real assessments for production
**Schedule:** Retrain monthly as you collect more data
**Result:** 80-90% confidence predictions

---

## 📁 Quick Reference

```bash
# Check current training data
python retrain_model.py

# Add new data and retrain
python retrain_model.py --add new_assessments.csv

# Just retrain with current data
python retrain_model.py --retrain

# Generate synthetic data (testing only)
python retrain_model.py --generate 100

# Test retrained model
python demo_xgboost.py

# Check model file size
ls -lh xgboost_balaji_models.pkl
```

---

**Ready to improve your AI! 🚀**

*The more real assessment data you add, the better your predictions become!*
