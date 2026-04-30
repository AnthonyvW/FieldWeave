#!/usr/bin/env bash
set -e

echo "Updating FieldWeave..."

echo "Pulling latest changes from git..."
if ! git pull; then
    echo "ERROR: git pull failed. Check your connection or repository status."
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating..."
    if ! python3 -m venv venv; then
        echo "ERROR: Failed to create virtual environment. Is Python installed and on PATH?"
        exit 1
    fi
fi

echo "Installing/updating dependencies..."
if ! venv/bin/pip install -r requirements.txt; then
    echo "ERROR: Failed to install requirements."
    exit 1
fi

echo "FieldWeave updated successfully."