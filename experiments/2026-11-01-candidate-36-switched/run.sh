#!/usr/bin/env bash
# The candidate-36 window on the head, detached from any SSH session. It checks the node
# checkouts out from under itself, so it runs from a copy outside the checkout:
#   cp experiments/2026-11-01-candidate-36-switched/run.sh /tmp/c36-run.sh
#   setsid -f bash /tmp/c36-run.sh COMMIT > LOG 2>&1 < /dev/null
#
# 1. Takes ~/spark-hold.json (idle and request guards) and keeps its heartbeat while it runs.
# 2. Detaches every node at COMMIT (already pushed to each node), stops serving.
# 3. Builds spark-ds41f:candidate-36-direct from the prepared inputs and distributes it.
# 4. Runs the release-test bundles tests/direct and tests/stock on the head (lab kernel-local).
# 5. Six measured boots (harness.py), stock first as the reference, in the order
#    stock, direct, direct-nccl, direct-nccl, direct, stock so drift cancels across arms.
# On any exit the trap restores what served before the window: every node at RESTORE_COMMIT,
# RESTORE_PROFILE booted, the hold removed. Lab's watchdog is not started: its fallback,
# config/cluster.json, names an image this site does not have.
set -uo pipefail
cd "$HOME/projects/spark-ds41f"
COMMIT=${1:?usage: run.sh COMMIT}
E=experiments/2026-11-01-candidate-36-switched
RESTORE_COMMIT=2985b088db756b1f657b91af70385a682964674d
RESTORE_PROFILE=experiments/2026-11-01-site-levers/serving-best.json
NODES=(192.168.1.121 192.168.1.122 192.168.1.123 192.168.1.18)
REPO=$PWD
OUT=results/private/c36
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)
mkdir -p "$OUT"

log() { echo "$(date -u +%FT%TZ) $*"; }
spark() { python3 bin/spark "$@" 2>&1 | grep -vE '^WARN: 192\.|^  |docker run'; return "${PIPESTATUS[0]}"; }
lab_py() {
  python3 - "$@" <<'EOF'
import importlib.machinery, importlib.util, sys
loader = importlib.machinery.SourceFileLoader("lab", "scripts/lab.py")
spec = importlib.util.spec_from_loader("lab", loader); lab = importlib.util.module_from_spec(spec); loader.exec_module(lab)
action, note = sys.argv[1], sys.argv[2]
if action == "open":
    hold = lab.read_hold()
    if hold and not lab.hold_is_ours(hold):
        sys.exit(f"held by {hold.get('holder')!r}")
    if any(r.exists() for r in (lab.REQUEST, lab.LEGACY_REQUEST)):
        sys.exit("cluster requested")
    if not hold:
        if not lab.idle_for(30):
            sys.exit("serving requests")
        lab.write_json_atomic(lab.HOLD, lab.new_hold(0, note))
elif action == "beat":
    lab.beat(note)
elif action == "release":
    if lab.hold_is_ours(lab.read_hold()):
        lab.HOLD.unlink()
EOF
}
detach_all() {
  local commit=$1 node
  for node in "${NODES[@]}"; do
    "${SSH[@]}" "spark@$node" "git -C $REPO checkout -q --detach $commit && test \$(git -C $REPO rev-parse HEAD) = $commit" \
      || { log "FAIL: $node did not detach at $commit"; return 1; }
  done
}

BEATER=
restore() {
  local code=$?
  log "restore (exit $code): every node at ${RESTORE_COMMIT:0:8}, boot $RESTORE_PROFILE"
  if detach_all "$RESTORE_COMMIT" && spark --cluster-config "$RESTORE_PROFILE" cluster start --replace --change-shape --apply; then
    lab_py release "candidate-36 switched A/B"
    log "restored; hold released"
  else
    log "RESTORE FAILED: hold kept so nobody assumes production is up"
  fi
  [ -n "$BEATER" ] && kill "$BEATER" 2>/dev/null
  log "done"
}

lab_py open "candidate-36 switched A/B" || { log "FAIL: could not take the cluster"; exit 1; }
trap restore EXIT
( while sleep 60; do lab_py beat "candidate-36 switched A/B" || exit; done ) &
BEATER=$!
log "hold taken; window open"

detach_all "$COMMIT" || exit 1
log "every node at ${COMMIT:0:8}"
spark --cluster-config "$E/direct.json" cluster stop --apply --parallel || { log "FAIL: stop"; exit 1; }

log "build image"
spark --cluster-config "$E/direct.json" build image --apply || { log "FAIL: build image"; exit 1; }
log "distribute"
spark --cluster-config "$E/direct.json" build distribute --apply || { log "FAIL: distribute"; exit 1; }

for arm in direct stock; do
  log "release tests: $arm"
  python3 scripts/lab.py kernel-local "$E/tests/$arm" --out "$OUT/tests-$arm" > "$OUT/tests-$arm.log" 2>&1
  log "release tests $arm: exit $? ($(tail -n 1 "$OUT/tests-$arm.log"))"
done

set -- "c36-stock-1 stock --set-reference" "c36-direct-1 direct" "c36-nccl-1 direct-nccl" \
       "c36-nccl-2 direct-nccl" "c36-direct-2 direct" "c36-stock-2 stock"
for spec in "$@"; do
  read -r id arm flag <<< "$spec"
  log "boot $id ($arm)"
  python3 "$E/harness.py" run "$id" "$E/$arm.json" ${flag:-} > "$OUT/$id.log" 2>&1
  code=$?
  log "boot $id: exit $code; $(grep -E '^METRIC (score|prefill_pct|decode_pct|grid_pct|outputs_identical|hca_errors)=' "$OUT/$id.log" | tr '\n' ' ')"
  if [ "$code" != 0 ] && [ "$id" = c36-stock-1 ]; then
    log "FAIL: no reference boot; stopping"
    exit 1
  fi
done
log "all jobs finished"
