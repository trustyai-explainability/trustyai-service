#!/bin/bash
# Test GitHub Actions workflows locally using act
# Install act: brew install act

set -e

if ! command -v act &> /dev/null; then
    echo "❌ act not found. Install with: brew install act"
    exit 1
fi

echo "🎭 Testing GitHub Actions workflows locally with act..."
echo ""

# Check for container runtime (prefer podman, fallback to docker)
CONTAINER_RUNTIME=""
if [ -S /var/run/docker.sock ]; then
    # Docker compatibility socket exists (Podman with docker compatibility enabled)
    export DOCKER_HOST="unix:///var/run/docker.sock"
    if command -v podman &> /dev/null && podman machine list 2>&1 | grep -q "Running"; then
        CONTAINER_RUNTIME="podman"
        echo "✅ Using Podman with Docker compatibility"
    else
        CONTAINER_RUNTIME="docker"
        echo "✅ Using Docker"
    fi
elif command -v docker &> /dev/null && docker info &> /dev/null; then
    CONTAINER_RUNTIME="docker"
    echo "✅ Using Docker"
else
    echo "❌ No container runtime found or not running."
    echo ""
    if command -v podman &> /dev/null; then
        echo "   Podman is installed but not running. Start it with:"
        echo "   podman machine start"
        echo ""
        echo "   Enable Docker compatibility with:"
        echo "   sudo /opt/homebrew/Cellar/podman/*/bin/podman-mac-helper install"
    else
        echo "   Install podman: brew install podman"
        echo "   Then run: podman machine init && podman machine start"
    fi
    exit 1
fi
echo ""

echo "Available workflows:"
act -l --container-architecture linux/amd64
echo ""

# Dry-run first
echo "📋 Dry-run (showing what would execute)..."
act -n -W .github/workflows/python-tests.yaml --container-architecture linux/amd64
echo ""

read -p "🚀 Run python-tests workflow? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running python-tests.yaml..."
    act push -W .github/workflows/python-tests.yaml \
        -P ubuntu-latest=catthehacker/ubuntu:act-latest \
        --container-architecture linux/amd64 \
        --artifact-server-path /tmp/artifacts
fi

echo ""
read -p "🔒 Run security-scan workflow? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running security-scan.yaml..."
    act push -W .github/workflows/security-scan.yaml \
        -P ubuntu-latest=catthehacker/ubuntu:act-latest \
        --container-architecture linux/amd64 \
        --artifact-server-path /tmp/artifacts
fi

echo ""
echo "✨ Done! Check /tmp/artifacts for any generated files."
