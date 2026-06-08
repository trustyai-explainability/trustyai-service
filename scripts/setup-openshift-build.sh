#!/usr/bin/env bash
# Setup OpenShift BuildConfig for TrustyAI Service Python

set -euo pipefail

NAMESPACE="trustyai-build"
QUAY_USER="rh-ee-sudsinha"

echo "=== OpenShift BuildConfig Setup for TrustyAI Service ==="
echo ""

# Check if logged into OpenShift
if ! oc whoami &>/dev/null; then
    echo "❌ Not logged into OpenShift. Please run: oc login <cluster-url>"
    exit 1
fi

echo "✓ Logged into OpenShift as: $(oc whoami)"
echo ""

# Create namespace if it doesn't exist
if oc get namespace "$NAMESPACE" &>/dev/null; then
    echo "✓ Namespace '$NAMESPACE' already exists"
else
    echo "Creating namespace '$NAMESPACE'..."
    oc create namespace "$NAMESPACE"
fi

# Prompt for Quay.io password
echo ""
echo "Enter your Quay.io password for user '$QUAY_USER':"
read -s QUAY_PASSWORD
echo ""

# Create base64 encoded auth string
AUTH_STRING=$(echo -n "${QUAY_USER}:${QUAY_PASSWORD}" | base64)

# Create dockerconfigjson secret
echo "Creating Quay.io push secret..."
cat <<EOF | oc apply -n "$NAMESPACE" -f -
apiVersion: v1
kind: Secret
metadata:
  name: quay-push-secret
  namespace: $NAMESPACE
type: kubernetes.io/dockerconfigjson
stringData:
  .dockerconfigjson: |
    {
      "auths": {
        "quay.io": {
          "auth": "$AUTH_STRING"
        }
      }
    }
EOF

echo "✓ Secret created"
echo ""

# Apply BuildConfig
echo "Creating BuildConfig..."
oc apply -f openshift-build.yaml -n "$NAMESPACE"

echo "✓ BuildConfig created"
echo ""

echo "=== Setup Complete ==="
echo ""
echo "To start a build, run:"
echo "  oc start-build trustyai-service-python -n $NAMESPACE --follow"
echo ""
echo "To watch builds:"
echo "  oc get builds -n $NAMESPACE -w"
echo ""
echo "To get build logs:"
echo "  oc logs -f bc/trustyai-service-python -n $NAMESPACE"
