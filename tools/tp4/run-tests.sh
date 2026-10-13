#!/usr/bin/env bash
# Run on the GPU host. Explicitly name the systemd service permitted to stop.
set -euo pipefail
service=${1:?Usage: run-tests.sh SYSTEMD_SERVICE [BUILD_IMAGE]}
image=${2:-localhost/infernix-tp4-build}
source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
systemctl show "$service" --property=LoadState --value | grep -qx loaded
was_running=false
if systemctl is-active --quiet "$service"; then was_running=true; fi
test_container="infernix-tp4-tests-$$"
restore() {
    result=$?
    trap - EXIT
    sudo -n podman rm -f "$test_container" >/dev/null 2>&1 || true
    if [[ "$was_running" == true ]]; then
        if ! sudo -n systemctl start "$service"; then
            echo "ERROR: could not restart $service" >&2
            exit 1
        fi
    fi
    exit "$result"
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
if [[ "$was_running" == true ]]; then sudo -n systemctl stop "$service"; fi
sudo -n podman run --rm --name "$test_container" --device nvidia.com/gpu=all \
    --network none --security-opt label=disable -v "$source_dir:/work" "$image" -lc \
    'set -o pipefail; nvidia-smi --query-gpu=index,name,memory.free --format=csv; ctest --test-dir /work/build-tp4 --output-on-failure --timeout 300 -R "^infernix_(parallel_(partition|transport|weights|state|qsa|qsa_devices|gdn|gdn_fold|offloaded_moe|expert_stream)|qsa_select|linear_bf16_a16|gdn_gating|causal_conv1d_silu|offloaded_moe_wide|resident_moe|hyper_connection|qwen4_exp_prefix_state|gated_delta_net_replay_record|kv_cache_append|kv_cache|host_context_arena)_test$" 2>&1 | tee /work/build-tp4/tp4-tests.log'
