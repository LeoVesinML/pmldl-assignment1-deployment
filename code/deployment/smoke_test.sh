#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Post-deployment verification.
#
# Every check is executed *inside* the deployed containers (via `docker exec`),
# which proves that
#   1. the API container is up, healthy and serving the freshly trained model,
#   2. a real prediction request returns a valid response,
#   3. the web app container is up and can reach the API over the compose
#      network (i.e. the two containers really talk to each other).
# ---------------------------------------------------------------------------
set -euo pipefail

API_CONTAINER="${API_CONTAINER:-pmldl-api}"
APP_CONTAINER="${APP_CONTAINER:-pmldl-app}"
RETRIES="${RETRIES:-20}"
SLEEP_SECONDS="${SLEEP_SECONDS:-5}"

log() { printf '%s | %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S')" "$*"; }

wait_for() {
    local container="$1" url="$2" name="$3"
    for attempt in $(seq 1 "${RETRIES}"); do
        if docker exec "${container}" curl -fsS --max-time 5 "${url}" > /dev/null 2>&1; then
            log "OK   ${name} is responding (attempt ${attempt})"
            return 0
        fi
        log "wait ${name} not ready yet (attempt ${attempt}/${RETRIES})"
        sleep "${SLEEP_SECONDS}"
    done
    log "FAIL ${name} did not become ready"
    docker logs --tail 50 "${container}" || true
    return 1
}

log "=== 1/5 API health check ==="
wait_for "${API_CONTAINER}" "http://localhost:8000/health" "model API"
HEALTH="$(docker exec "${API_CONTAINER}" curl -fsS http://localhost:8000/health)"
log "health: ${HEALTH}"
echo "${HEALTH}" | grep -q '"model_loaded":true' || { log "FAIL model is not loaded"; exit 1; }

log "=== 2/5 model metadata ==="
docker exec "${API_CONTAINER}" curl -fsS http://localhost:8000/model-info | head -c 600
echo

log "=== 3/5 prediction request ==="
PAYLOAD='{"fixed_acidity":7.0,"volatile_acidity":0.27,"citric_acid":0.36,"residual_sugar":6.1,
"chlorides":0.045,"free_sulfur_dioxide":45.0,"total_sulfur_dioxide":170.0,"density":0.9938,
"ph":3.0,"sulphates":0.45,"alcohol":11.8,"wine_type":"white"}'
PREDICTION="$(docker exec "${API_CONTAINER}" curl -fsS -X POST http://localhost:8000/predict \
    -H 'Content-Type: application/json' -d "${PAYLOAD}")"
log "prediction: ${PREDICTION}"
echo "${PREDICTION}" | grep -q '"prediction"' || { log "FAIL unexpected prediction payload"; exit 1; }

log "=== 4/5 web application health check ==="
wait_for "${APP_CONTAINER}" "http://localhost:8501/_stcore/health" "web application"

log "=== 5/5 app -> API connectivity over the compose network ==="
docker exec "${APP_CONTAINER}" curl -fsS --max-time 10 http://api:8000/health > /dev/null
log "OK   the app container reaches the API at http://api:8000"

log "Smoke test passed: API and web application are deployed and talking to each other."
