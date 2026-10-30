#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(dirname "$0")"

echo "Setting up FieldWeave..."

chmod +x "$SCRIPT_DIR/ubuntu_update.sh"
chmod +x "$SCRIPT_DIR/ubuntu_start.sh"

echo "Running initial update..."
bash "$SCRIPT_DIR/ubuntu_update.sh"

echo "Setup complete. You can now run FieldWeave with ./ubuntu_start.sh"