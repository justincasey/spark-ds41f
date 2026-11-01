#!/usr/bin/env python3
"""One measured boot of an arm of this experiment, scored against a recorded reference run.

Runs on the head node from the deployment checkout (the current directory):

  python3 harness.py window          open our lab window (hold + watchdog), or refresh its heartbeat
  python3 harness.py run ID ARM [--set-reference]
  python3 harness.py close           restore production (lab window close)

``run`` boots ARM (stopping any other engine first), waits for every GPU to cool, reads each
node's RDMA error counters, runs ``suite.json`` through ``bin/spark bench --suite`` into
results/private/c36/ID, reads the counters again, and prints ``METRIC name=value`` lines
(``hca_errors`` is the counters' total growth over the bench). ``--set-reference`` records
the run as the reference (results/private/c36/reference) that later runs are scored against;
the reference must be a stock candidate-36 boot.

score = 100 * exp(1/2 * mean(log prefill ratios over 32K, 128K, 256K)
                  + 1/4 * log(c16 paired decode ratio) + 1/4 * log(c1 paired decode ratio))

so 100 is the reference and prefill and decode weigh equally. Paired decode ratios are the
median per-request decode-rate ratio over requests both runs sent with the same prompt
(servebench.report.paired), which cancels the text; prefill ratios are mean cold tok/s.
"""

import argparse
import fcntl
import importlib.machinery
import importlib.util
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "bin" / "spark").is_file():
    raise SystemExit(f"run from the deployment checkout, not {ROOT}")
EXPERIMENT = "experiments/2026-11-01-candidate-36-switched"
SUITE = f"{EXPERIMENT}/suite.json"
RESULTS = ROOT / "results" / "private" / "c36"
REFERENCE = RESULTS / "reference"
NOTE = "candidate-36 switched A/B"
# RDMA error counters a lossless switched fabric must not grow (per port, hw_counters), and
# congestion counters (ECN marks and CNPs: PFC/ECN working, not errors) recorded beside them.
HCA_ERROR_COUNTERS = ("out_of_sequence", "packet_seq_err", "implied_nak_seq_err", "out_of_buffer",
                      "local_ack_timeout_err", "rnr_nak_retry_err", "req_transport_retries_exceeded",
                      "req_cqe_error", "resp_cqe_error", "duplicate_request")
HCA_CONGESTION_COUNTERS = ("np_ecn_marked_roce_packets", "np_cnp_sent", "rp_cnp_handled")
COOL_BELOW_C = 50
COOL_TIMEOUT_S = 900
PREFILL_SIZES = (32768, 131072, 262144)
RUN_LOCK = RESULTS / ".run.lock"
# lab's watchdog closes a window whose heartbeat is 900 s old and polls every 60 s: from
# 840 s on, a refresh could race its production restore.
STALE_S = 840
RESTORE_WAIT_S = 1800

_loader = importlib.machinery.SourceFileLoader("lab", str(ROOT / "scripts" / "lab.py"))
_spec = importlib.util.spec_from_loader("lab", _loader)
lab = importlib.util.module_from_spec(_spec)
_loader.exec_module(lab)
spark = lab.spark
sys.path.insert(0, str(ROOT))


def fail(message: str, code: int = 1) -> None:
    lab.log(f"FAIL: {message}")
    raise SystemExit(code)


def requested() -> Path | None:
    return next((r for r in (lab.REQUEST, lab.LEGACY_REQUEST) if r.exists()), None)


def run_in_progress() -> bool:
    """Whether a ``run`` holds RUN_LOCK (it does for its whole boot and bench)."""
    RUN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOCK.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def watchdog_running() -> bool:
    return subprocess.run(["pgrep", "-f", "lab.py watchdog"], capture_output=True).returncode == 0


# ---------------------------------------------------------------- window

def window() -> None:
    """Open a lab window over the production profile, or refresh ours.

    lab.py's own ``window open`` also requires the commit on origin/main; this site deploys
    from its fork's ``site`` branch and this session publishes nothing, so the hold, the
    idle guard and the watchdog are taken here directly."""
    request = requested()
    if request:
        fail(f"cluster requested ({request}); not taking it", 3)
    if run_in_progress():
        fail("another harness run holds the run lock; not touching the cluster", 3)
    hold = lab.read_hold()
    if hold and not lab.hold_is_ours(hold):
        fail(f"cluster held by {hold.get('holder')!r} since {hold.get('since')}", 3)
    if hold and lab.heartbeat_age(hold) >= STALE_S:
        # The watchdog (60 s poll, 900 s limit) is restoring production or about to. Its
        # restore keeps the hold when doctor --live fails, which it always does at this site
        # (display manager), so wait for the watchdog to exit rather than for the hold to go.
        lab.log(f"window {hold['window']} heartbeat {lab.heartbeat_age(hold):.0f} s old; waiting for the "
                "watchdog's restore to finish")
        deadline = time.monotonic() + RESTORE_WAIT_S
        while watchdog_running():
            if time.monotonic() > deadline:
                fail("the stale window's watchdog is still running; inspect the head", 3)
            time.sleep(15)
        label = subprocess.run(["docker", "inspect", "spark-ds41f", "--format",
                                f'{{{{index .Config.Labels "{spark.PROFILE_LABEL}"}}}}'],
                               text=True, capture_output=True).stdout.strip()
        if label != lab.PRODUCTION_CONFIG:
            fail(f"the stale window ended with {label or 'no engine'} serving, not {lab.PRODUCTION_CONFIG}; "
                 "inspect the head", 3)
        if lab.hold_is_ours(lab.read_hold()):
            lab.HOLD.unlink()
        hold = None
    if hold:
        lab.beat(NOTE)
        lab.log(f"window {hold['window']} refreshed")
        return
    if not lab.idle_for(30):
        fail("the cluster is serving requests; not taking it", 3)
    hold = lab.new_hold(0, NOTE)
    lab.write_json_atomic(lab.HOLD, hold)
    log_path = ROOT / "results" / "private" / "lab" / f"watchdog-{hold['window']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        subprocess.Popen([sys.executable, str(ROOT / "scripts" / "lab.py"), "watchdog",
                          "--production-config", lab.PRODUCTION_CONFIG], cwd=ROOT,
                         stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    lab.log(f"window {hold['window']} open; watchdog restores {lab.PRODUCTION_CONFIG} on a stale heartbeat")


class Heartbeat(threading.Thread):
    def __init__(self, note: str) -> None:
        super().__init__(daemon=True)
        self.note, self.stop, self.lost = note, threading.Event(), False

    def run(self) -> None:
        while not self.stop.wait(60):
            try:
                lab.beat(self.note)
            except SystemExit:
                self.lost = True
                return


# ---------------------------------------------------------------- boot and cooling

def same_nodes(a: dict, b: dict) -> bool:
    """The same node set, ranks, addresses and rails. ``roce_traffic_class`` is ignored: the
    production map carries it for the 1030 tooling, candidate-36's refuses it (its sparknet
    sets class 106 on every switched fabric)."""
    def strip(nodes: dict) -> dict:
        return {**nodes, "nodes": [{k: v for k, v in node.items() if k != "roce_traffic_class"}
                                   for node in nodes["nodes"]]}
    return strip(a) == strip(b)


def boot(arm: str) -> float | None:
    """lab.boot with the ready time kept: seconds from launch to ready."""
    if not same_nodes(spark.configuration(argparse.Namespace(cluster_config=arm))[1], lab.nodes_config()):
        fail("the arm's node map differs from production's")
    if not lab.stop_other_engine(arm):
        fail("could not stop another engine")
    lab.log(f"start {arm}")
    code, lines = spark_cli("--cluster-config", arm, "cluster", "start", "--replace", "--change-shape", "--apply")
    if code:
        fail(f"boot failed (exit {code})")
    return lab.ready_seconds(lines)


def hca_counters() -> dict[str, dict[str, int]]:
    """Every node's RDMA error counters, per ``<device>/<port>/<counter>``."""
    nodes = lab.nodes_config()
    script = ("for f in /sys/class/infiniband/*/ports/*/hw_counters/{"
              + ",".join(HCA_ERROR_COUNTERS + HCA_CONGESTION_COUNTERS)
              + "}; do [ -r \"$f\" ] && echo \"$f $(cat \"$f\")\"; done")
    out = {}
    for node in nodes["nodes"]:
        text = spark.run_ssh(nodes, node, "bash", "-c", script).stdout
        counters = {}
        for line in text.splitlines():
            path, _, value = line.rpartition(" ")
            parts = path.split("/")  # /sys/class/infiniband/<dev>/ports/<n>/hw_counters/<name>
            counters[f"{parts[4]}/{parts[6]}/{parts[8]}"] = int(value)
        out[node["name"]] = counters
    return out


def counter_growth(before: dict, after: dict) -> dict[str, dict[str, int]]:
    return {node: {k: after[node][k] - v for k, v in counters.items() if after[node].get(k, v) != v}
            for node, counters in before.items()}


# Host findings this site carries on every node, outside the serving stack and the same for
# every arm: bench's live check reports them as errors, so the bench runs with
# --allow-mismatch and any other error fails the run.
SITE_ERRORS = ("display manager is active",)


def spark_cli(*arguments: str) -> tuple[int, list[str]]:
    """bin/spark with its output kept. Per-node host warnings (``WARN: <node>: ...``) and the
    kernel-corrections block repeat for every node and every boot, so they are counted, not
    printed; every other warning (a short CUDA graph ladder, for one) is printed."""
    nodes = tuple(f"WARN: {node['name']}:" for node in lab.nodes_config()["nodes"])
    process = subprocess.Popen([sys.executable, str(ROOT / "bin" / "spark"), *arguments], cwd=ROOT, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines, hidden, in_hidden = [], 0, False
    for line in process.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        starts_hidden = line.startswith(nodes) or line.startswith("WARN: Kernel corrections")
        in_hidden = starts_hidden or (in_hidden and line.startswith("  "))
        if in_hidden or "docker run" in line:
            hidden += 1
        else:
            print(line, flush=True)
    code = process.wait()
    if hidden:
        print(f"({hidden} per-node host warning lines not shown)", flush=True)
    return code, lines


def gpu_temperatures() -> dict[str, int]:
    nodes = lab.nodes_config()
    out = {}
    for node in nodes["nodes"]:
        text = spark.run_ssh(nodes, node, "nvidia-smi", "--query-gpu=temperature.gpu",
                             "--format=csv,noheader,nounits").stdout.split()
        out[node["name"]] = int(text[0]) if text else 999
    return out


def cool() -> int:
    """Wait, idle, until every GPU is at or below COOL_BELOW_C (this site has no fan-floor
    control, so bench's own cooling is skipped); returns the hottest at the start."""
    began = time.monotonic()
    while True:
        temperatures = gpu_temperatures()
        hottest = max(temperatures.values())
        if hottest <= COOL_BELOW_C:
            lab.log(f"cool: {temperatures}")
            return hottest
        if time.monotonic() - began > COOL_TIMEOUT_S:
            lab.log(f"cool: still {temperatures} after {COOL_TIMEOUT_S} s; measuring anyway")
            return hottest
        time.sleep(15)


# ---------------------------------------------------------------- metrics

def read_json(path: Path) -> dict:
    from servebench.__main__ import read_json as servebench_read
    return servebench_read(str(path))


def step(run_dir: Path, step_id: str) -> dict:
    index = json.loads((run_dir / "suite.json").read_text())
    found = next((s for s in index["steps"] if s["id"] == step_id), None)
    if found is None or found["exit"] != 0:
        fail(f"{run_dir.name}: step {step_id} did not complete ({found and found['exit']})")
    return read_json(run_dir / found["output"])


def decode_ratio(reference: dict, candidate: dict) -> tuple[float, float | None]:
    """Paired median decode-rate ratio and its 95% half-width in %; the window ratio when
    either run kept no request table."""
    from servebench import report
    pairs = report.paired(reference, candidate)
    if pairs and "decode_tps" in pairs:
        entry = pairs["decode_tps"]
        return 1 + entry["change_pct"] / 100, entry["ci95_pct"]
    a = reference["summary"]["output_tps"]["mean"]
    b = candidate["summary"]["output_tps"]["mean"]
    return b / a, None


def grid_geomean_ms(run: dict) -> float:
    from servebench import stats
    logs = []
    for row in run["engine"]:
        if not row["shape"].get("prefill"):
            kept, _ = stats.trim_slow(row["samples"])
            logs.append(math.log(sum(kept) / len(kept)))
    return math.exp(sum(logs) / len(logs))


def metrics(run_dir: Path, reference: Path) -> dict:
    from servebench import behaviour, engine
    out: dict[str, float] = {}
    logs = []
    ref_prefill = {row["size"]: row for row in step(reference, "prefill")["prefill"]}
    for row in step(run_dir, "prefill")["prefill"]:
        if row["size"] not in PREFILL_SIZES or row["failed"]:
            fail(f"prefill {row['size']}: {row['failed']} failed")
        mean = row["tps"]["mean"]
        out[f"prefill_{row['size'] // 1024}k_tps"] = mean
        logs.append(math.log(mean / ref_prefill[row["size"]]["tps"]["mean"]))
    prefill_log = sum(logs) / len(logs)
    out["prefill_pct"] = 100 * math.expm1(prefill_log)
    decode_log = 0.0
    for stream in ("c16", "c1"):
        candidate, base = step(run_dir, stream), step(reference, stream)
        ratio, ci = decode_ratio(base, candidate)
        out[f"{stream}_tps"] = candidate["summary"]["output_tps"]["mean"]
        out[f"{stream}_paired_pct"] = 100 * (ratio - 1)
        if ci is not None:
            out[f"{stream}_paired_ci95_pct"] = ci
        decode_log += 0.5 * math.log(ratio)
    out["decode_pct"] = 100 * math.expm1(decode_log)
    out["score"] = 100 * math.exp(0.5 * prefill_log + 0.5 * decode_log)
    grid, grid_ref = step(run_dir, "grid"), step(reference, "grid")
    out["grid_step_ms"] = grid_geomean_ms(grid)
    out["grid_pct"] = 100 * (engine.boot_ratios([grid], grid_ref)[0] - 1)
    model, model_ref = step(run_dir, "model"), step(reference, "model")
    out["outputs_identical"] = float(behaviour.corpus_digest(model["generation"]["sequences"])
                                    == behaviour.corpus_digest(model_ref["generation"]["sequences"]))
    acceptance = model["generation"].get("acceptance", {}).get("acceptance_length")
    if acceptance is not None:
        out["acceptance_length"] = acceptance
    return out


# ---------------------------------------------------------------- run

def run(run_id: str, arm: str, set_reference: bool) -> None:
    hold = lab.read_hold()
    if not lab.hold_is_ours(hold):
        fail("no window of ours is open (run `window` first)")
    if not set_reference and not (REFERENCE / "suite.json").is_file():
        fail(f"no reference run at {REFERENCE}; record a stock candidate-36 boot with --set-reference first")
    out = RESULTS / run_id
    if out.exists():
        fail(f"{out} exists")
    RUN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock = RUN_LOCK.open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fail("another harness run holds the run lock")
    beat = Heartbeat(f"{NOTE}: {run_id}")
    beat.start()
    try:
        ready = boot(arm)
        lab.log(f"ready +{ready} s")
        hottest = cool()
        before = hca_counters()
        code, lines = spark_cli("--cluster-config", arm, "bench", "--suite", SUITE, "--compare", "none",
                                "--allow-mismatch", "--output", str(out), "--label", run_id)
        growth = counter_growth(before, hca_counters())
        out.mkdir(parents=True, exist_ok=True)
        (out / "hca-counters.json").write_text(json.dumps({"errors": list(HCA_ERROR_COUNTERS),
                                                           "congestion": list(HCA_CONGESTION_COUNTERS),
                                                           "before": before, "growth": growth}, indent=1) + "\n")
        (out / "arm.json").write_text((ROOT / arm).read_text())
        if beat.lost:
            fail("the hold stopped being ours during the run")
        unexpected = [line for line in lines
                      if line.startswith("ERROR:") and not any(error in line for error in SITE_ERRORS)]
        if unexpected:
            fail("the live cluster differs from the arm: " + "; ".join(unexpected))
        if code:
            fail(f"bench exit {code}")
        if set_reference:
            if REFERENCE.is_symlink() or REFERENCE.exists():
                REFERENCE.unlink()
            REFERENCE.symlink_to(run_id)
            lab.log(f"reference: {run_id}")
        values = metrics(out, REFERENCE)
        if ready is not None:
            values["boot_s"] = ready
        values["start_gpu_c"] = hottest
        values["hca_errors"] = sum(n for c in growth.values() for k, n in c.items()
                                   if k.rsplit("/", 1)[1] in HCA_ERROR_COUNTERS)
        values["hca_ecn_marks"] = sum(n for c in growth.values() for k, n in c.items()
                                      if k.endswith("/np_ecn_marked_roce_packets"))
        values["hca_counters_read"] = sum(len(c) for c in before.values())
        (out / "metrics.json").write_text(json.dumps({"run": run_id, "reference": REFERENCE.resolve().name,
                                                      "metrics": values}, indent=1) + "\n")
        for name, value in values.items():
            print(f"METRIC {name}={value:.6g}", flush=True)
    finally:
        beat.stop.set()
        lock.close()
    request = requested()
    if request:
        lab.window_close(f"cluster requested ({request})")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("window")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("id")
    run_parser.add_argument("arm", help="repository-relative arm configuration")
    run_parser.add_argument("--set-reference", action="store_true")
    commands.add_parser("close")
    args = parser.parse_args()
    if args.command == "window":
        window()
    elif args.command == "run":
        run(args.id, args.arm, args.set_reference)
    else:
        raise SystemExit(0 if lab.window_close(f"{NOTE} closed") else 1)


if __name__ == "__main__":
    main()
