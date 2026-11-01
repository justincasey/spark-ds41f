#!/usr/bin/env bash
# One measured boot of experiments/2026-11-01-site-levers/arm.json on the site's four
# Sparks (tp4-switched), scored against the recorded stock spark-ds41f-1030 reference.
#
# 1. Opens (or refreshes) our lab window on the head: hold file, idle guard, watchdog.
# 2. Commits the working tree as a detached candidate commit (the branch is not moved),
#    pushes it to every node's checkout over SSH and detaches each node at it. Nothing is
#    published to GitHub; node checkouts stay clean and identical.
# 3. Runs harness.py on the head, detached from this SSH session, and streams its log.
#
# Run from anywhere: experiments/2026-11-01-site-levers/measure.sh
# AR_SET_REFERENCE=1 records this run as the reference (only for a stock 1030 arm).
set -euo pipefail
cd "$(dirname "$0")/../.."

EXPERIMENT=experiments/2026-11-01-site-levers
HEAD_NODE=spark@192.168.1.121
NODES=(192.168.1.121 192.168.1.122 192.168.1.123 192.168.1.18)
REPO=/home/spark/projects/spark-ds41f
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30)

# 1. The window, with this tree's harness (the head may still be on an older commit).
"${SSH[@]}" "$HEAD_NODE" "cd $REPO && python3 - window" < "$EXPERIMENT/harness.py"

# 2. The candidate commit: the working tree as a commit on HEAD, without moving the branch.
index=$(mktemp)
trap 'rm -f "$index"' EXIT
cp "$(git rev-parse --git-path index)" "$index"
GIT_INDEX_FILE=$index git add -A -- .
tree=$(GIT_INDEX_FILE=$index git write-tree)
if [ "$tree" = "$(git rev-parse 'HEAD^{tree}')" ]; then
  commit=$(git rev-parse HEAD)
else
  commit=$(git commit-tree "$tree" -p HEAD -m "site levers candidate")
fi
echo "candidate commit $commit"
for node in "${NODES[@]}"; do
  dirty=$("${SSH[@]}" "spark@$node" "git -C $REPO status --porcelain --untracked-files=all")
  if [ -n "$dirty" ]; then
    echo "FAIL: $node checkout is dirty:"; echo "$dirty"; exit 1
  fi
  GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=10" \
    git push -q -f "ssh://spark@$node$REPO" "$commit:refs/site-levers/candidate"
  "${SSH[@]}" "spark@$node" "git -C $REPO checkout -q --detach $commit && test \$(git -C $REPO rev-parse HEAD) = $commit"
done

# 3. The measured boot, detached on the head; its log streams here until it exits.
run_id="ar-$(date -u +%Y%m%dT%H%M%SZ)-${commit:0:8}"
flag=""
if [ "${AR_SET_REFERENCE:-0}" = 1 ]; then flag="--set-reference"; fi
log="results/private/autoresearch/$run_id.log"
"${SSH[@]}" "$HEAD_NODE" "cd $REPO && mkdir -p results/private/autoresearch && \
  setsid -f bash -c 'python3 $EXPERIMENT/harness.py run $run_id $flag > $log 2>&1; echo \$? > $log.exit' \
  < /dev/null > /dev/null 2>&1"
echo "run $run_id: $REPO/$log"
# Liveness only while it runs (the log's last line, never a METRIC line); the whole log,
# METRIC lines included, is printed once the run has exited.
last=""
while true; do
  sleep 30
  code=$("${SSH[@]}" "$HEAD_NODE" "cat $REPO/$log.exit 2>/dev/null" || true)
  if [ -n "$code" ]; then
    "${SSH[@]}" "$HEAD_NODE" "grep -v 'docker run' $REPO/$log" || true
    exit "$code"
  fi
  status=$("${SSH[@]}" "$HEAD_NODE" "tail -n 1 $REPO/$log 2>/dev/null" | grep -v METRIC | cut -c1-160 || true)
  if [ -n "$status" ] && [ "$status" != "$last" ]; then echo "... $status"; last=$status; fi
done
