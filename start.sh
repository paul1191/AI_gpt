#!/bin/bash
# ── RegAnalyst Quick Start ────────────────────────────────────────────────────
echo ""
echo "⚖️  RegAnalyst — On-Premise Multi-Agent Regulatory AI"
echo "════════════════════════════════════════════════════"
echo ""

# Check Ollama
if ! command -v ollama &> /dev/null; then
  echo "❌ Ollama not found. Install from: https://ollama.com/download"
  exit 1
fi

# Check if llama3 is pulled
if ! ollama list | grep -q "llama3"; then
  echo "⬇️  Pulling llama3 model (one-time ~4.7GB download)..."
  ollama pull llama3
fi

echo "✅ Ollama ready"

# Start backend
echo ""
echo "🚀 Starting backend on http://localhost:8000"
cd backend
if [ ! -d "venv" ]; then
  echo "  Creating virtualenv..."
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt -q
else
  source venv/bin/activate
fi
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!

# Start frontend
echo "🚀 Starting frontend on http://localhost:5173"
cd ../frontend
if [ ! -d "node_modules" ]; then
  echo "  Installing npm packages..."
  npm install -q
fi
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ App running!"
echo "   Frontend: http://localhost:5173"
echo "   Backend:  http://localhost:8000"
echo "   API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait and cleanup
trap "kill $BACKEND_PID $FRONTEND_PID; echo 'Stopped.'" EXIT
wait
