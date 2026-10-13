#!/usr/bin/env bash
# Run on the GPU host; downloads are deliberately separate from service downtime.
set -euo pipefail
service=${1:?Usage: run-model-smoke.sh SYSTEMD_SERVICE ARTIFACT NGRAM [CLI_OPTIONS...]}
artifact=$(realpath "${2:?Artifact part 1 is required}")
ngram=$(realpath "${3:?N-gram volume is required}")
shift 3
source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
run_dir="$source_dir/tp4-results/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$run_dir"
systemctl show "$service" --property=LoadState --value | grep -qx loaded
was_running=false
if systemctl is-active --quiet "$service"; then was_running=true; fi
test_container="infernix-tp4-model-$$"
monitor_pid=
restore() {
    result=$?
    trap - EXIT
    if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi
    sudo -n podman rm -f "$test_container" >/dev/null 2>&1 || true
    if [[ "$was_running" == true ]]; then
        if ! sudo -n systemctl start "$service"; then
            echo "ERROR: could not restart $service" >&2
            exit 1
        fi
    fi
    echo "Run status: $result; logs: $run_dir"
    exit "$result"
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
if [[ "$was_running" == true ]]; then sudo -n systemctl stop "$service"; fi
free -h | tee "$run_dir/host-before.txt"
(
    while true; do
        date -u +%FT%TZ
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
        grep -E 'MemAvailable|MemFree|SwapFree|Mlocked' /proc/meminfo
        sleep 2
    done
) > "$run_dir/memory.log" &
monitor_pid=$!
# Mount the whole artifact directory: the reader opens sibling parts itself.
timeout --signal=TERM --kill-after=30s 1800s sudo -n podman run --rm \
    --name "$test_container" --device nvidia.com/gpu=all --network none \
    -e "TP4_PREFLIGHT_REGEX=${TP4_PREFLIGHT_REGEX:-}" \
    --ulimit memlock=-1:-1 --security-opt label=disable \
    -v "$source_dir:/work" -v "$(dirname "$artifact"):/model:ro" \
    -v "$(dirname "$ngram"):/ngram:ro" localhost/infernix-tp4-build -lc \
    'set -e; if [[ -n "$TP4_PREFLIGHT_REGEX" ]]; then
        ctest --test-dir /work/build-tp4 --output-on-failure --timeout 300 -R "$TP4_PREFLIGHT_REGEX"
    fi; exec /work/build-tp4/apps/infernix "$@"' _ \
    "/model/$(basename "$artifact")" --ngram-volume "/ngram/$(basename "$ngram")" \
    --device 0 --parallel-devices 0,1,2,3 --max-context 131072 --max-new 64 \
    --prefill-chunk 512 --kv-dtype int8 --expert-state off \
    --greedy --no-thinking --log-colours off \
    --prompt 'What is 6 times 7? Answer in one short sentence.' "$@" \
    2>&1 | tee "$run_dir/generation.log"
