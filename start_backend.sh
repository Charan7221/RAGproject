#!/bin/bash
# Start the RAG Backend API

cd "$(dirname "$0")/backend"

echo "🚀 Starting RAG Backend API..."
echo "================================"
echo ""

# Check if PostgreSQL is running
if ! pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo "⚠️  PostgreSQL is not running!"
    echo "Starting PostgreSQL..."
    brew services start postgresql@14
    echo "Waiting for PostgreSQL to start..."
    sleep 3
fi

echo "✅ PostgreSQL is running"
echo ""
echo "Starting backend API server..."
echo "API will be available at: http://localhost:8000"
echo ""

../venv/bin/python api.py
