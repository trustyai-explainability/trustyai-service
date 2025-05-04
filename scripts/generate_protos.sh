#!/bin/bash
# Script to generate protobuf stubs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="${SCRIPT_DIR}/../src/proto"

echo "Generating Python protobuf stubs..."
cd "${PROTO_DIR}"

PROTO_FILES=$(find . -name "*.proto")
if [ -z "$PROTO_FILES" ]; then
    echo "No .proto files found in ${PROTO_DIR}"
    exit 1
fi

for PROTO_FILE in $PROTO_FILES; do
    echo "Compiling ${PROTO_FILE}..."
    protoc --python_out=. --proto_path=. "$PROTO_FILE"
done

if [ ! -f "__init__.py" ]; then
    echo "# Generated protobuf package" > "__init__.py"
    echo "Created __init__.py file"
fi

echo "Protobuf generation completed successfully!"
