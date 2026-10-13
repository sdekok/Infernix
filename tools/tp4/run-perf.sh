#!/usr/bin/env bash
# Target-host runner: all real-model performance runs reserve >=128K INT8 KV.
set -euo pipefail
service=${1:?Usage: run-perf.sh SERVICE ARTIFACT NGRAM LABEL [BENCH_OPTIONS...]}
artifact=$(realpath "${2:?}")
ngram=$(realpath "${3:?}")
label=${4:?}
[[ "$label" =~ ^[a-zA-Z0-9_-]+$ ]]
shift 4
source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
run_dir="$source_dir/tp4-results/$(date -u +%Y%m%dT%H%M%SZ)-$label-$$"
mkdir -p "$run_dir"
systemctl show "$service" --property=LoadState --value | grep -qx loaded
was_running=false
if systemctl is-active --quiet "$service"; then was_running=true; fi
container="infernix-tp4-perf-$$"
monitor_pid=
restore() {
    result=$?
    trap - EXIT
    if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi
    sudo -n podman rm -f "$container" >/dev/null 2>&1 || true
    if [[ "$was_running" == true ]]; then sudo -n systemctl start "$service" || result=1; fi
    echo "Run status: $result; logs: $run_dir"
    exit "$result"
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
if [[ "$was_running" == true ]]; then sudo -n systemctl stop "$service"; fi
(
    while true; do
        date -u +%FT%TZ
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw,clocks.sm --format=csv,noheader,nounits
        grep -E 'MemAvailable|SwapFree' /proc/meminfo
        sleep 2
    done
) > "$run_dir/memory.log" &
monitor_pid=$!
profiler=()
if [[ "${TP4_PROFILE:-0}" == 1 ]]; then
    profiler=(/opt/nvidia/nsight-systems/2026.1.3/target-linux-x64/nsys profile
        --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --cuda-graph-trace=node
        --capture-range=cudaProfilerApi --capture-range-end=stop
        --force-overwrite=true --output=/results/timeline)
    set -- "$@" --profile-measured -r 1
fi
printf '%q ' "$@" > "$run_dir/options.txt"
timeout --signal=TERM --kill-after=30s 1800s sudo -n podman run --rm \
    --name "$container" --device nvidia.com/gpu=all --network none \
    -e "TP4_PREFLIGHT_REGEX=${TP4_PREFLIGHT_REGEX:-}" \
    -e "CUDA_LAUNCH_BLOCKING=${TP4_LAUNCH_BLOCKING:-0}" \
    --ulimit memlock=-1:-1 --security-opt label=disable \
    -v "$source_dir:/work" -v "$run_dir:/results" -v /opt/nvidia:/opt/nvidia:ro \
    -v "$(dirname "$artifact"):/model:ro" -v "$(dirname "$ngram"):/ngram:ro" \
    localhost/infernix-tp4-build -lc 'set -e
        if [[ -n "$TP4_PREFLIGHT_REGEX" ]]; then
            ctest --test-dir /work/build-tp4 --output-on-failure --timeout 300 -R "$TP4_PREFLIGHT_REGEX"
        fi
        exec "$@"' _ \
    "${profiler[@]}" "${TP4_BENCH_BINARY:-/work/build-tp4/bench/infernix_bench}" \
    --weights "/model/$(basename "$artifact")" --ngram-volume "/ngram/$(basename "$ngram")" \
    --device 0 --parallel-devices 0,1,2,3 --prefill-chunk 512 \
    -pg "${TP4_WORKLOAD:-512,256}" --warmup 2 -r 3 -o json --output-file /results/report.json \
    "$@" --max-ctx 131072 --kv-dtype int8 2>&1 | tee "$run_dir/benchmark.log"
