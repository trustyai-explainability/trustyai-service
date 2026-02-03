#!/usr/bin/env bash
# scripts/test_upload_endpoint.sh
#
# KServe-strict endpoint test for /data/upload
#
#
# Usage:
#   ENDPOINT="https://<your-service-route>/data/upload" \
#   MODEL="gaussian-credit-model" \
#   TAG="TRAINING" \
#   ./scripts/test_upload_endpoint.sh

set -uo pipefail

# --- Config via env vars (no secrets hardcoded) ---
: "${ENDPOINT:?ENDPOINT is required, e.g. https://.../data/upload}"
MODEL="${MODEL:-gaussian-credit-model}"
# Separate model for BYTES to avoid mixing with an existing numeric dataset
MODEL_BYTES="${MODEL_BYTES:-${MODEL}-bytes}"
TAG="${TAG:-TRAINING}"
AUTH_HEADER="${AUTH_HEADER:-}"  # e.g. 'Authorization: Bearer <token>'

CURL_OPTS=( --silent --show-error -H "Content-Type: application/json" )
[[ -n "$AUTH_HEADER" ]] && CURL_OPTS+=( -H "$AUTH_HEADER" )

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; CYAN=$'\033[36m'; RESET=$'\033[0m'
pass_cnt=0; fail_cnt=0; results=()
have_jq=1; command -v jq >/dev/null 2>&1 || have_jq=0

line(){ printf '%s\n' "--------------------------------------------------------------------------------"; }
snippet(){ if (( have_jq )); then echo "$1" | jq -r 'tostring' 2>/dev/null | head -c 240; else echo "$1" | head -c 240; fi; }

# ---------- payload builders ----------
mk_inputs_2x4_int32() {
  cat <<JSON
{
  "inputs": [
    {
      "name": "credit_inputs",
      "shape": [2, 4],
      "datatype": "INT32",
      "data": [[1,2,3,4],[5,6,7,8]]
    }
  ]
}
JSON
}

mk_inputs_empty(){ echo '{ "inputs": [] }'; }

mk_outputs_col(){ # name dtype nested_data_json ; shape [2,1]
  local name="$1" dtype="$2" data="$3"
  cat <<JSON
[
  { "name": "$name", "datatype": "$dtype", "shape": [2, 1], "data": $data }
]
JSON
}

mk_body(){ # request_json outputs_json -> uses MODEL
  local req="$1" out="$2"
  cat <<JSON
{
  "model_name": "$MODEL",
  "data_tag": "$TAG",
  "request": $req,
  "response": { "model_name": "test-model", "outputs": $out }
}
JSON
}

mk_body_for_model(){ # model request_json outputs_json
  local mdl="$1" req="$2" out="$3"
  cat <<JSON
{
  "model_name": "$mdl",
  "data_tag": "$TAG",
  "request": $req,
  "response": { "model_name": "test-model", "outputs": $out }
}
JSON
}

sample_data_for_dtype(){
  case "$1" in
    BOOL)  echo '[[1],[0]]' ;;
    INT8|INT16|INT32|INT64|UINT8|UINT16|UINT32|UINT64) echo '[[1],[2]]' ;;
    FP16|FP32|FP64) echo '[[0.1],[0.2]]' ;;
    BYTES) echo '[["a"],["b"]]' ;;
    *)     echo '[]' ;;
  esac
}

# ---------- runner ----------
run_test () {
  local name="$1" want="$2" sub="$3" payload="$4"
  local tmp http body
  tmp="$(mktemp)"
  http=$(curl -X POST "$ENDPOINT" "${CURL_OPTS[@]}" -d "$payload" -o "$tmp" -w "%{http_code}" || true)
  body="$(cat "$tmp")"; rm -f "$tmp"

  local ok=1
  [[ "$http" == "$want" ]] || ok=0
  if [[ -n "$sub" ]] && ! echo "$body" | grep -qi -- "$sub"; then ok=0; fi

  if (( ok )); then
    pass_cnt=$((pass_cnt+1)); results+=("PASS|$name|$http|$(snippet "$body")")
    printf "%s[PASS]%s %s (HTTP %s)\n" "$GREEN" "$RESET" "$name" "$http"
  else
    fail_cnt=$((fail_cnt+1)); results+=("FAIL|$name|$http|$(snippet "$body")")
    printf "%s[FAIL]%s %s (HTTP %s)\n" "$RED" "$RESET" "$name" "$http"
    [[ -n "$sub" ]] && printf "  expected code=%s and body to contain: %q\n" "$want" "$sub" || true
    printf "  body: %s\n" "$(snippet "$body")"
  fi
}

# ---------- edge cases ----------
line; echo "${CYAN}Running edge cases...${RESET}"

run_test "valid_int32_bool" "200" '"status":"success"' \
  "$(mk_body "$(mk_inputs_2x4_int32)" "$(mk_outputs_col predict BOOL '[[1],[0]]')")"

run_test "missing_model_name" "422" '"Field required"' \
  "$(cat <<JSON
{
  "data_tag": "$TAG",
  "request": $(mk_inputs_2x4_int32),
  "response": { "model_name": "test-model", "outputs": $(mk_outputs_col predict FP32 '[[0.1],[0.2]]') }
}
JSON
)"

run_test "empty_inputs" "400" "data field was empty" \
  "$(mk_body "$(mk_inputs_empty)" "$(mk_outputs_col predict FP32 '[[0.1],[0.2]]')")"

run_test "missing_request_block" "422" '"Field required"' \
  "$(cat <<JSON
{
  "model_name": "$MODEL",
  "data_tag": "$TAG",
  "response": { "model_name": "test-model", "outputs": $(mk_outputs_col predict FP32 '[[0.1],[0.2]]') }
}
JSON
)"

run_test "incorrect_shape" "422" "Declared shape (3, 3) does not match data shape (2, 4)" \
  "$(mk_body \
      "$(cat <<JSON
{
  "inputs": [
    { "name":"credit_inputs", "shape":[3,3], "datatype":"FP64", "data": [[1,2,3,4],[5,6,7,8]] }
  ]
}
JSON
)" \
      "$(mk_outputs_col predict FP32 '[[0.1],[0.2]]')" \
  )"

run_test "different_model_names" "200" '"status":"success"' \
  "$(cat <<JSON
{
  "model_name": "$MODEL",
  "request": $(mk_inputs_2x4_int32),
  "response": { "model_name": "fake-name-123", "outputs": $(mk_outputs_col predict FP32 '[[0.1],[0.2]]') }
}
JSON
)"

run_test "wrong_bool_values" "422" "must be bool or 0/1" \
  "$(mk_body "$(mk_inputs_2x4_int32)" "$(mk_outputs_col predict BOOL '[[7],[3]]')")"

# ---------- dtype sweep (incl. BYTES) ----------
line; echo "${CYAN}KServe dtype sweep (outputs)...${RESET}"

for dt in BOOL INT8 INT16 INT32 INT64 UINT8 UINT16 UINT32 UINT64 FP16 FP32 FP64; do
  data="$(sample_data_for_dtype "$dt")"
  run_test "dtype_${dt}" "200" '"status":"success"' \
    "$(mk_body "$(mk_inputs_2x4_int32)" "$(mk_outputs_col predict "$dt" "$data")")"
done

# BYTES uses a separate model to avoid dtype/storage mixing in existing datasets
data_bytes='[["a"],["b"]]'
run_test "dtype_BYTES" "200" '"status":"success"' \
  "$(mk_body_for_model "$MODEL_BYTES" "$(mk_inputs_2x4_int32)" "$(mk_outputs_col predict BYTES "$data_bytes")")"

# ---------- summary ----------
line
echo "${CYAN}Summary:${RESET}"
total=$((pass_cnt+fail_cnt))
printf "  Total: %d  %sPass:%s %d  %sFail:%s %d\n" "$total" "$GREEN" "$RESET" "$pass_cnt" "$RED" "$RESET" "$fail_cnt"
line

if (( fail_cnt > 0 )); then
  echo "${YELLOW}Details for failures:${RESET}"
  for r in "${results[@]}"; do
    IFS='|' read -r status name http body <<<"$r"
    if [[ "$status" == "FAIL" ]]; then
      printf "%s[FAIL]%s %s (HTTP %s)\n" "$RED" "$RESET" "$name" "$http"
      printf "  body: %s\n" "$body"
      line
    fi
  done
fi