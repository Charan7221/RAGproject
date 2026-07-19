#!/bin/bash
# Quick start script for FastAPI + React application

echo "🚀 Starting RAG Document QA - Full Stack"
echo "========================================"
echo ""

# Check if PostgreSQL is running
if ! docker compose ps postgres 2>/dev/null | grep -q "running\|Up"; then
    if ! docker-compose ps postgres 2>/dev/null | grep -q "running\|Up"; then
        echo "⚠️  PostgreSQL not running. Starting..."
        docker compose up -d postgres 2>/dev/null || docker-compose up -d postgres
        echo "⏳ Waiting for PostgreSQL to be ready..."
        for i in {1..30}; do
            if docker compose exec -T postgres pg_isready -U rag_user -d rag_db >/dev/null 2>&1; then
                break
            fi
            sleep 1
        done
        echo "✅ PostgreSQL started"
    fi
else
    echo "✅ PostgreSQL is running"
fi

# Check for API key
if [ -z "$GOOGLE_API_KEY" ] && [ -z "$GEMINI_API_KEY" ] && [ ! -f .env ]; then
    echo "⚠️  No GOOGLE_API_KEY found. Copy .env.example to .env and add your key."
fi

echo ""
echo "📋 Starting services..."
echo ""

# Start backend in background
echo "🔧 Starting FastAPI backend on port 8000..."
cd backend
if [ -f ../venv/bin/python ]; then
    ../venv/bin/python api.py > ../backend.log 2>&1 &
else
    python3 api.py > ../backend.log 2>&1 &
fi
BACKEND_PID=$!
cd ..
echo "   Backend PID: $BACKEND_PID"
echo "   Logs: backend.log"

# Wait for backend to start
sleep 3

# Start frontend in background
echo "🎨 Starting React frontend on port 3000..."
cd frontend
npm start > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "   Frontend PID: $FRONTEND_PID"
echo "   Logs: frontend.log"

echo ""
echo "✅ Services started successfully!"
echo ""
echo "🌐 Access the application:"
echo "   Frontend:  http://localhost:3000"
echo "   Backend:   http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo ""
echo "📝 To stop services:"
echo "   ./stop_fullstack.sh"
echo ""
echo "💡 Tip: Check logs with 'tail -f backend.log' or 'tail -f frontend.log'"
echo ""

# Save PIDs to file for cleanup
echo "$BACKEND_PID" > .pids
echo "$FRONTEND_PID" >> .pids

# Keep script running
wait
