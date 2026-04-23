#!/bin/bash
# ── Synapta Demo Server Launcher ──
# Starts the FastAPI backend with Nemotron-30B + adapters + router.
# Server runs on port 8765 by default.
# Open http://localhost:8765 in your browser after model loads.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "╔══════════════════════════════════════════════════════╗"
echo "║          🧬 SYNAPTA DEMO SERVER                     ║"
echo "║  Nemotron-30B + 3 LoRA Adapters + Trained Router    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "📍 Project: $PROJECT_ROOT"
echo "🌐 URL:     http://localhost:${PORT:-8765}"
echo ""
echo "⏳ Model loading takes ~2-3 minutes on first start..."
echo ""

cd "$PROJECT_ROOT"

# Check if required packages are available
python3 -c "import uvicorn, fastapi, torch, peft, bitsandbytes" 2>/dev/null || {
    echo "❌ Missing packages. Installing..."
    pip3 install uvicorn fastapi[standard] websockets
}

# Launch server
exec python3 demo/server.py --port "${PORT:-8765}" --host "0.0.0.0"
