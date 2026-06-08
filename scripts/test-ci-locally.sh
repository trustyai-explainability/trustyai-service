#!/bin/bash
# Test CI steps locally before pushing
# This mimics what GitHub Actions will run

set -e  # Exit on error

echo "🧪 Testing CI steps locally..."
echo ""

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "📦 Installing dependencies..."
uv sync --all-groups --extra mariadb
echo "✅ Dependencies installed"
echo ""

echo "🔍 Running linters..."
echo "  - Ruff check..."
uv run ruff check src tests
echo "  - Ruff format check..."
uv run ruff format --check src tests
echo "✅ Linting passed"
echo ""

echo "🔬 Running type checker..."
uv run pyrefly check src/
echo "✅ Type checking passed"
echo ""

echo "🔒 Running security scan..."
uv run bandit -c pyproject.toml -r src/
echo "✅ Security scan passed"
echo ""

echo "🧪 Running tests..."
# Check if MariaDB is running
if command -v mariadb &> /dev/null && mariadb -e "SELECT 1" &> /dev/null; then
    echo "  - Full test suite (with MariaDB)..."
    uv run pytest tests/ -v --tb=short
else
    echo "  ⚠️  MariaDB not available, some integration tests may fail"
    echo "  - Running test suite..."
    uv run pytest tests/ -v --tb=short
fi
echo "✅ Tests passed"
echo ""

echo "✨ All CI checks passed locally! Safe to push."
