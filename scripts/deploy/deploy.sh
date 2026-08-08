#!/usr/bin/env bash
# AI Foundry Enterprise — Production Bootstrapper
set -e

echo "======================================================="
echo "   AI FOUNDRY ENTERPRISE (AFE) PRODUCTION DEPLOYMENT   "
echo "======================================================="

echo "[1/4] Verifying Python runtime and virtual environment..."
python --version

echo "[2/4] Applying database migrations and seeding RBAC roles..."
alembic upgrade head
python -c "from app import create_app; app = create_app(); print('Database verified.')"

echo "[3/4] Ensuring storage directories..."
python -c "from core.storage.paths import storage_paths; storage_paths.init_all()"

echo "[4/4] Starting high-performance AFE Web & Inference Server..."
exec python app.py
