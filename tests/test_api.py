"""
Test Flask API Server
"""
import requests
import json

API_URL = "http://localhost:5001"

print("=" * 70)
print("  Testing Balaji Framework XGBoost AI Server")
print("=" * 70)

# Test 1: Health Check
print("\n1️⃣  Testing Health Check...")
try:
    response = requests.get(f"{API_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Single Prediction
print("\n2️⃣  Testing Single Prediction...")
test_input = {
    "country": "IN",
    "crop": "Potato",
    "partner": "Balaji ",
    "irrigation": "irrigated",
    "hired_workers": "Yes",
    "area": 10.0
}

try:
    response = requests.post(
        f"{API_URL}/api/v1/predict",
        json=test_input,
        headers={"Content-Type": "application/json"}
    )
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Success: {data['statistics']['total_indicators']} indicators predicted")
        print(f"   High Confidence: {data['statistics']['high_confidence']}")
        print(f"   Average Confidence: {data['statistics']['average_confidence']}%")
        
        # Show first 3 predictions
        print("\n   Sample Predictions:")
        for i, (indicator, pred) in enumerate(list(data['predictions'].items())[:3]):
            print(f"     • {indicator}: {pred['value']} ({pred['confidence']:.1f}%)")
    else:
        print(f"   ❌ Error: {response.text}")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Get Indicators
print("\n3️⃣  Testing Get Indicators...")
try:
    response = requests.get(f"{API_URL}/api/v1/indicators")
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Success: {data['total']} indicators available")
        print(f"   First 5: {', '.join(data['indicators'][:5])}")
    else:
        print(f"   ❌ Error: {response.text}")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("  ✅ API Testing Complete!")
print("=" * 70)
