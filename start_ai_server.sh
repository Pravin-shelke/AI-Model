#!/bin/bash

# Start Balaji Framework XGBoost AI Server
# This script starts the Flask API server for your React Native app

echo "🚀 Starting Balaji Framework AI Server..."

cd /Users/pravinshelke/Documents/AI-Model

# Activate virtual environment
source venv/bin/activate

# Start Flask server
python flask_api_server.py

