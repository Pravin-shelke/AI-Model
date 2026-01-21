import pickle

with open('xgboost_balaji_models.pkl', 'rb') as f:
    models = pickle.load(f)

print(f'Total trained models: {len(models)}')
print(f'\nRM-1 trained: {"RM-1" in models}')

if 'RM-1' in models:
    print('  ✅ RM-1 has a trained model')
else:
    print('  ❌ RM-1 was SKIPPED (insufficient quality data)')

print(f'\nSample trained indicators (first 30):')
for idx, key in enumerate(list(models.keys())[:30]):
    print(f'  {idx+1}. {key}')
