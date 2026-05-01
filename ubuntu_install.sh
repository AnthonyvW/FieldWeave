#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(dirname "$0")"

echo "Setting up FieldWeave..."

chmod +x "$SCRIPT_DIR/update.sh"
chmod +x "$SCRIPT_DIR/start.sh"

echo "Running initial update..."
bash "$SCRIPT_DIR/update.sh"

echo "Setup complete. You can now run FieldWeave with ./start.sh"