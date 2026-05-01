#!/usr/bin/env bash
set -e

echo "Starting FieldWeave..."

if [ -d "venv" ]; then
    echo "Virtual environment found. Activating..."
    source venv/bin/activate
else
    echo "Virtual environment not found. Creating..."
    if ! python3 -m venv venv; then
        echo "ERROR: Failed to create virtual environment. Is Python installed and on PATH?"
        exit 1
    fi
    source venv/bin/activate
    echo "Installing dependencies..."
    if ! pip install -r requirements.txt; then
        echo "ERROR: Failed to install requirements."
        exit 1
    fi
fi

echo "Launching FieldWeave..."
python3 main.py