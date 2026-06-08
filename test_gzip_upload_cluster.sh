#!/bin/bash
# Test script for gzip-compressed uploads to TrustyAI service on OpenShift
# This tests the GzipRequestMiddleware implementation

set -e

NAMESPACE="${NAMESPACE:-model-namespace}"
SERVICE_NAME="${SERVICE_NAME:-trustyai-service}"
PORT="${PORT:-8081}"

echo "=========================================="
echo "TrustyAI Gzip Upload Test"
echo "=========================================="
echo "Namespace: $NAMESPACE"
echo "Service: $SERVICE_NAME"
echo "Local port: $PORT"
echo ""

# Check if already port-forwarding
if lsof -i :$PORT > /dev/null 2>&1; then
    echo "⚠️  Port $PORT is already in use. Killing existing process..."
    lsof -ti :$PORT | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Start port-forward in background
echo "Starting port-forward to $SERVICE_NAME..."
oc port-forward -n $NAMESPACE svc/$SERVICE_NAME $PORT:80 > /tmp/pf.log 2>&1 &
PF_PID=$!
echo "Port-forward PID: $PF_PID"

# Wait for port-forward to be ready
echo "Waiting for port-forward to be ready..."
for i in {1..10}; do
    if curl -s http://localhost:$PORT/ > /dev/null 2>&1; then
        echo "✅ Port-forward ready"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "❌ Port-forward failed to start"
        kill $PF_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

echo ""
echo "=========================================="
echo "Test 1: Uncompressed Upload (Baseline)"
echo "=========================================="

UNCOMPRESSED_PAYLOAD=$(cat <<'EOF'
{
  "model_name": "cluster-test-uncompressed",
  "data_tag": "CLUSTER_TEST",
  "is_ground_truth": false,
  "request": {
    "inputs": [
      {
        "name": "input",
        "shape": [1],
        "datatype": "INT64",
        "data": [42]
      }
    ]
  },
  "response": {
    "outputs": [
      {
        "name": "output",
        "shape": [1],
        "datatype": "INT64",
        "data": [100]
      }
    ]
  }
}
EOF
)

echo "Sending uncompressed payload..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST http://localhost:$PORT/data/upload \
  -H "Content-Type: application/json" \
  -d "$UNCOMPRESSED_PAYLOAD")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "Response: $BODY"
echo "HTTP Status: $HTTP_CODE"

if [ "$HTTP_CODE" == "200" ]; then
    echo "✅ PASS: Uncompressed upload successful"
else
    echo "❌ FAIL: Expected 200, got $HTTP_CODE"
fi

echo ""
echo "=========================================="
echo "Test 2: Gzip-Compressed Upload"
echo "=========================================="

COMPRESSED_PAYLOAD=$(cat <<'EOF'
{
  "model_name": "cluster-test-compressed",
  "data_tag": "CLUSTER_TEST_GZIP",
  "is_ground_truth": false,
  "request": {
    "inputs": [
      {
        "name": "input",
        "shape": [1],
        "datatype": "INT64",
        "data": [99]
      }
    ]
  },
  "response": {
    "outputs": [
      {
        "name": "output",
        "shape": [1],
        "datatype": "INT64",
        "data": [200]
      }
    ]
  }
}
EOF
)

echo "Compressing payload with gzip..."
GZIP_FILE=$(mktemp)
echo "$COMPRESSED_PAYLOAD" | gzip > "$GZIP_FILE"

echo "Sending gzip-compressed payload..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST http://localhost:$PORT/data/upload \
  -H "Content-Type: application/json" \
  -H "Content-Encoding: gzip" \
  --data-binary @"$GZIP_FILE")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "Response: $BODY"
echo "HTTP Status: $HTTP_CODE"

rm "$GZIP_FILE"

if [ "$HTTP_CODE" == "200" ]; then
    echo "✅ PASS: Gzip upload successful"
else
    echo "❌ FAIL: Expected 200, got $HTTP_CODE"
    echo "This means the GzipRequestMiddleware is NOT working properly"
fi

echo ""
echo "=========================================="
echo "Test 3: Malformed Gzip Data"
echo "=========================================="

echo "Sending invalid gzip data..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST http://localhost:$PORT/data/upload \
  -H "Content-Type: application/json" \
  -H "Content-Encoding: gzip" \
  -d "not-valid-gzip-data")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "Response: $BODY"
echo "HTTP Status: $HTTP_CODE"

if [ "$HTTP_CODE" == "400" ]; then
    if echo "$BODY" | grep -q -i "decompress"; then
        echo "✅ PASS: Malformed gzip returns 400 with clear error"
    else
        echo "⚠️  PARTIAL: Got 400 but error message unclear"
    fi
else
    echo "❌ FAIL: Expected 400, got $HTTP_CODE"
fi

echo ""
echo "=========================================="
echo "Test 4: Gzip on Non-Upload Endpoint (Path Scoping)"
echo "=========================================="

echo "Testing that gzip middleware only applies to /data/upload..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X GET http://localhost:$PORT/ \
  -H "Content-Encoding: gzip")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "Response: $BODY"
echo "HTTP Status: $HTTP_CODE"

if [ "$HTTP_CODE" == "200" ]; then
    echo "✅ PASS: Root endpoint unaffected by gzip middleware"
else
    echo "❌ FAIL: Root endpoint affected (middleware not scoped correctly)"
fi

# Cleanup
echo ""
echo "=========================================="
echo "Cleanup"
echo "=========================================="
echo "Killing port-forward (PID: $PF_PID)..."
kill $PF_PID 2>/dev/null || true
wait $PF_PID 2>/dev/null || true
echo "✅ Cleanup complete"

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "All tests completed!"
echo ""
echo "If Test 2 (Gzip-Compressed Upload) passed with HTTP 200,"
echo "the GzipRequestMiddleware is working correctly on the cluster."
