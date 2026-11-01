# Site levers on stock 1030 (tp4-switched)

**Base:** `b01a6287` (spark-ds41f-1030 lock with the direct-exchange series of
`experiments/2026-11-01-direct-exchange-switched`), image `spark-ds41f:1030-direct`, shape `tp4-switched`:
four DGX Sparks on one MikroTik CRS804-4DDQ at the 192.168.1.0/24 site, no fan-floor control
(`host.fan_control false`).

**Reference:** stock spark-ds41f-1030. The direct-exchange image differs from stock 1030 only in two Python
files (`pynccl.py`, `tilelang/collectives.py`); `overlay/stock/` holds their 1030 versions (vLLM tree
`ea1281f8`, blobs `64d16a71` and `5ed772e5`, taken from the prepared build tree's parent commit of
`cecd9f79`), and the stock arm mounts them over the image's. Every other arm is measured against the
first stock boot.

## Questions

One causal variable per arm, each judged on two boots (a boot and a repeat boot):

1. Direct exchange on the switch (vLLM patch `0001` of `experiments/2026-11-01-direct-exchange-switched`):
   prefill gain and identical outputs against stock 1030.
2. sparknet's switched NCCL profile against the tp4-switched settings: 8 channels instead of 4, 1 MiB NCCL
   buffers instead of 4 MiB, `NCCL_CUMEM_ENABLE=0`, a 4 MiB one-shot all-gather cap instead of 2 MiB.
3. NVFP4 draft heads (`VLLM_DS41_DRAFT_NVFP4_HEAD=1`, `VLLM_DS41_MARKOV_NVFP4=1`) with six drafts (CUDA graphs
   to 112 rows), together because they interact.
4. `max_num_batched_tokens` 24,576 (from 16,384).
5. Adaptive verification row cost 0.01 (from 0.035).

Further levers enter the same way, one variable at a time, each an arm of its own.

## Harness

`autoresearch.sh` (repository root) runs one measured boot of `arm.json`:

1. `harness.py window` on the head: takes `~/spark-hold.json` (lab format, holder `spark-lab: autoresearch
   site levers`) after 30 idle seconds, or refreshes our heartbeat, and starts `scripts/lab.py watchdog`,
   which restores `config/cluster.json` if the heartbeat goes 15 minutes stale. `lab.py window open`'s publish
   guard (origin/main) cannot pass at this site, which deploys its fork's `site` branch, and this session
   publishes nothing, so the window is taken directly.
2. The working tree becomes a detached commit (the branch does not move), pushed to every node's checkout
   over SSH (`refs/autoresearch/candidate`); every node detaches at it, clean. `config/cluster.json` is never
   changed by an arm, so the watchdog's restore boots production at any candidate commit.
3. `harness.py run` on the head: boots the arm (`cluster start --replace --change-shape --apply`), waits idle
   until every GPU is at or below 50 °C, runs `suite.json` with `bin/spark bench --suite ... --compare none`
   into `results/private/autoresearch/<run>/`, and prints the metrics.

`suite.json` (about 22 minutes): greedy outputs of 64 prompts (identity and heat soak), the engine grid,
closed loops at 16 and 1 streams (60 s warm-up, 300 s measured), cold real-text prefill at 32K, 128K and
256K (three repeats).

**Score** (primary, higher is better; 100 is the reference boot):
`100 * exp(1/2 * mean(log prefill ratio at 32K, 128K, 256K) + 1/4 * log c16 ratio + 1/4 * log c1 ratio)`.
Decode ratios are servebench's paired per-request decode-rate medians (same prompt, same index), which
cancel the text; prefill ratios are mean cold tok/s. Secondary: each prefill size, window tok/s, paired
changes with their intervals, the grid's decode geometric mean (ms) and change, greedy output identity
against the reference, acceptance length, boot time.

## Runs

| Run | Arm | Score | Prefill | c16 paired | c1 paired | Grid | Outputs identical |
|---|---|---:|---:|---:|---:|---:|---|
| `ar-20261101T055815Z-18a74fde` (reference) | stock 1030 | 100 | 6,796 / 6,835 / 6,565 tok/s | 241.9 tok/s | 77.3 tok/s | 24.75 ms | — |
