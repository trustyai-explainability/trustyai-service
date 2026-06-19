#!/usr/bin/env bash
# Dismiss all open Trivy code-scanning alerts created before today.
# Usage: bash scripts/dismiss-stale-trivy-alerts.sh [--dry-run]
#
# Requires: gh (GitHub CLI) authenticated with repo scope

set -uo pipefail

REPO="trustyai-explainability/trustyai-service"
TOOL="Trivy"
CUTOFF=$(date -u +%Y-%m-%dT00:00:00Z)
REASON="won't fix"
COMMENT="Bulk dismiss: stale Trivy alerts pre-dating dedup fix. New scans will re-open any still-relevant findings."
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "DRY RUN — no alerts will be dismissed"
fi

echo "Repo:   $REPO"
echo "Tool:   $TOOL"
echo "Cutoff: $CUTOFF"
echo ""

page=1
total=0
dismissed=0
errors=0

while true; do
    alerts=$(gh api "repos/$REPO/code-scanning/alerts?state=open&tool_name=$TOOL&per_page=100&page=$page" 2>/dev/null) || {
        echo "API error on page $page, retrying..."
        continue
    }

    count=$(echo "$alerts" | jq 'length' 2>/dev/null) || count=0
    if [[ "$count" -eq 0 ]]; then
        break
    fi

    echo "Page $page: $count alerts"

    stale=$(echo "$alerts" | jq -r --arg cutoff "$CUTOFF" \
        '.[] | select(.created_at < $cutoff) | .number' 2>/dev/null) || stale=""

    for num in $stale; do
        total=$((total + 1))
        if $DRY_RUN; then
            if ((total % 100 == 0)); then
                echo "  [dry-run] $total alerts so far (page $page)..."
            fi
        else
            result=$(gh api -X PATCH "repos/$REPO/code-scanning/alerts/$num" \
                -f state=dismissed \
                -f dismissed_reason="$REASON" \
                -f dismissed_comment="$COMMENT" \
                2>&1) || true

            if echo "$result" | grep -q '"state":"dismissed"\|already dismissed'; then
                dismissed=$((dismissed + 1))
                if ((dismissed % 100 == 0)); then
                    echo "  dismissed $dismissed so far..."
                fi
            elif echo "$result" | grep -q "refused"; then
                true
            else
                errors=$((errors + 1))
                if ((errors <= 5)); then
                    echo "  failed #$num: $(echo "$result" | head -1)"
                elif ((errors == 6)); then
                    echo "  (suppressing further errors)"
                fi
            fi
        fi
    done

    page=$((page + 1))
done

echo ""
if $DRY_RUN; then
    echo "Would dismiss $total stale Trivy alerts"
else
    echo "Dismissed $dismissed / $total stale Trivy alerts ($errors errors)"
fi
