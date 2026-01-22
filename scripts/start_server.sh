#!/bin/bash

echo "Starting Assessment AI Server..."
cd "$(dirname "$0")/.."
python -m src.api.server
