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

`measure.sh` (run during the session from the repository root as `autoresearch.sh`) runs one measured boot of `arm.json`:

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

All on 2026-11-01, one session, arms cumulative in the order run. Changes are against the reference boot
(`ar-20261101T055815Z-18a74fde`: prefill 6,796 / 6,835 / 6,565 tok/s at 32K / 128K / 256K, c16 241.9 tok/s,
c1 77.3 tok/s, grid 24.75 ms); prefill is the geometric mean of the three sizes, c16 and c1 are paired decode,
grid is the engine grid's decode geometric mean (negative is faster). Every boot's 64 greedy outputs were
identical to the reference's. Logged runs 4-5 were an arm the tooling refused before boot
(`NCCL_MIN_NCHANNELS` on a switched transport); nothing ran.

| Boot | Arm (adds to the row above unless noted) | Score | Prefill | c16 | c1 | Grid |
|---|---|---:|---:|---:|---:|---:|
| ref | stock 1030 | 100.00 | 0 | 0 | 0 | 0 |
| 1 | stock 1030, repeat | 100.07 | +0.23% | -0.20% | +0.01% | +0.19% |
| 2 | direct exchange (1030-direct) | 101.54 | +3.10% | -0.01% | -0.01% | +0.58% |
| 3 | direct exchange, repeat | 101.25 | +2.55% | +0.09% | -0.16% | +0.53% |
| 6 | + sparknet switched NCCL profile | 101.86 | +3.81% | +0.09% | -0.18% | +0.58% |
| 7 | repeat | 101.49 | +3.07% | +0.09% | -0.20% | -0.37% |
| 8 | + NVFP4 draft heads, six drafts (112-row graphs) | 101.67 | +2.58% | +0.55% | +1.00% | -3.57% |
| 9 | repeat | 101.86 | +3.03% | +0.42% | +1.00% | -3.38% |
| 10 | + `max_num_batched_tokens` 24,576 | 102.85 | +5.00% | +0.71% | +0.78% | -3.21% |
| 11 | repeat | 103.09 | +5.78% | +0.67% | +0.27% | -3.57% |
| 12 | + row cost 0.01 | 103.62 | +6.12% | +0.68% | +1.68% | -3.36% |
| 13 | repeat | 103.45 | +5.50% | +1.25% | +1.65% | -3.29% |
| 14 | + `max_num_batched_tokens` 32,768 (from 24,576) | 104.32 | +7.90% | +0.35% | +1.37% | -3.39% |
| 15 | repeat | 104.51 | +7.87% | +0.93% | +1.56% | -3.12% |
| 16 | seven drafts (128-row graphs), discarded | 103.61 | +7.50% | -0.27% | -0.01% | -1.70% |
| 17 | five drafts (96-row graphs) instead of six | 104.56 | +7.70% | +1.00% | +2.02% | -4.21% |
| 18 | repeat: **best stack** | **104.84** | +7.94% | +1.45% | +2.22% | -4.17% |
| 19 | best stack without the NCCL profile (ablation) | 104.07 | +6.65% | +1.14% | +1.97% | -4.60% |
| 20 | ablation, repeat | 104.21 | +6.81% | +1.25% | +2.10% | -3.89% |
| 21 | best stack with 16 NCCL channels, discarded | 104.86 | +8.18% | +1.13% | +2.17% | -4.59% |
| 22 | best stack, row cost 0.005, discarded | 104.42 | +8.07% | +0.13% | +1.67% | -4.00% |
| 23 | best stack without the drafter's NVFP4 vocabulary head (ablation) | 103.57 | +7.21% | -0.06% | +0.17% | -0.09% |
| 24 | best stack without the NVFP4 Markov head (ablation) | 104.31 | +7.18% | +1.08% | +1.95% | -3.90% |
| 25 | best stack, third boot | 104.70 | +7.76% | +1.31% | +2.15% | -3.74% |
| 26 | best stack, row cost 0.02, discarded | 104.27 | +7.71% | +0.93% | +0.93% | -3.94% |

Best stack, boot 18: prefill 7,331 / 7,393 / 7,076 tok/s, c16 245.1 tok/s, c1 79.0 tok/s, grid 23.72 ms.

Lowest `MemAvailable` during the suite (head / ranks): 21.0 / 22.5 GiB at a 16,384 budget, 18.5 / 19.9 GiB
at 24,576, 16.4 / 17.7 GiB at 32,768.

## Conclusions

Each item's gain is its two boots' mean against the two boots of the arm before it, so it is measured on the
stack it was added to; the same lever on another stack is not measured. A/A between the two stock boots:
0.07 score; within an arm, prefill boots spread up to 0.7% and the grid ±0.5%.

1. **Direct exchange (0009): proven.** Prefill +2.7% (+2.83% against +0.12%), every size; closed-loop decode
   and greedy outputs unchanged. The grid read +0.46% on both direct boots; over boots
   (`servebench.engine.spread`, Welch, boot as the sample) that is +0.46% ±1.05% against stock, and +0.23%
   ±0.72% over all four boots that run the direct path at a 16,384 budget: no detectable decode change. The
   old r6w figures (+1.1 to +3.4%) hold on this base, flatter across sizes.
2. **sparknet's switched NCCL profile: proven by ablation.** Removing it from the best stack costs 1.1%
   prefill (+6.73% against +7.82%, each pair within 0.25%); added at a 16,384 budget it read +0.6%, inside
   that arm's spread. Decode unchanged. Sixteen channels instead of eight add nothing beyond noise (+0.35%,
   one boot).
3. **NVFP4 draft heads: gain measured; six drafts do not pay.** Heads with six drafts (row cost 0.035,
   budget 16,384): decode +0.8% (c1 +1.2%, c16 +0.4%), the grid's step 3.5% cheaper. On the later stack (row
   cost 0.01, budget 32,768) five drafts beat six by 0.6% decode (c1 +0.6%, c16 +0.6%; grid 4.2% cheaper than
   stock against 3.3%) and seven lose 1.2%. The heads were never measured alone at five drafts; the grid's
   4.2% at five drafts, where nothing else in the stack changes a fixed-row step, is the heads' step saving.
   The old stack's k=6 does not carry over: take the heads at the recipe's five drafts.
4. **Prefill budget 24,576: proven,** +2.5 prefill points (+5.39% against +2.80%), the old figure (+2.4).
   **32,768** adds 2.1 more (+7.88%). Each 8,192 tokens costs about 2.1-2.5 GiB of the memory floor and
   lengthens decode stalls beside a cold long prefill (below).
4b. **Row cost 0.01: proven at six drafts,** decode +0.7% (c1 +1.15%, c16 +0.3%); not measured on its own at
   five.

**Proven as a whole:** the best stack's three boots (17, 18, 25) score 104.56, 104.84 and 104.70 (sd 0.14)
against stock's 100.00 and 100.07: prefill +7.8%, c16 +1.2%, c1 +2.2%. The per-item figures above sum the
steps that led there.

After the five items, on the best stack, one boot each:

- **Which head:** without the drafter's own NVFP4 vocabulary head the grid is back at stock (-0.09%) and
  decode gains nothing (+0.06%); without the NVFP4 Markov head the grid is 3.9% cheaper (best stack 3.7-4.2%)
  and decode +1.51%. The drafter head engaged (its whole saving is in the grid) and carries item 3; the
  Markov head is inside noise. Both stay in the serving profile; the Markov head could be dropped at no
  measurable cost.
- **Row cost at five drafts:** 0.005 and 0.02 each lose about 0.8% decode against 0.01 (0.005 at c16,
  0.02 at c1). 0.01 stays.

## Serving profile

`serving-best.json` is the best stack without `VLLM_SERVER_DEV_MODE` (its resolved serve arguments equal
`arm.json`'s). It is this site's serving profile from these findings, booted with
`bin/spark --cluster-config experiments/2026-11-01-site-levers/serving-best.json cluster start --replace --apply`;
`config/cluster.json` and its baselines are unchanged, so lab windows still restore stock 1030-direct until a
promotion. To trade cold prefill back for shorter decode stalls, set `--max-num-batched-tokens` 24,576 and
`--long-prefill-token-threshold` 24,480 (or the recipe's 16,384 / 16,288).

## The prefill budget's cost

`stall-suite.json` (the release suite's `stall-3` and `stall-11` steps, then one cold 1M prefill) on the
best stack at three budgets, one boot each (`arms/budget-16k.json`, `arms/budget-24k.json`, `arm.json`), at
commit `0af0d650`, inside the same window, run on the head from the checkout for each arm in turn:

```sh
python3 bin/spark --cluster-config $ARM cluster start --replace --change-shape --apply
python3 bin/spark --cluster-config $ARM bench --suite experiments/2026-11-01-site-levers/stall-suite.json \
  --compare none --allow-mismatch --output results/private/autoresearch/stall-$LABEL --label stall-$LABEL
```

with the hold's heartbeat refreshed every minute (`lab.beat`). Receipts: `runs/stall-*`.

| Budget | Decode gap during a cold 131K prefill, mean (3 / 11 streams) | Longest decode stall (3 / 11) | Short-request TTFT p50 (3 / 11) | 131K prefill beside decode | 1M cold prefill | Lowest MemAvailable, head / ranks |
|---:|---:|---:|---:|---:|---:|---:|
| 16,384 | 1.15 / 1.19 s | 2.42 / 3.21 s | 3.63 / 2.96 s | 6,195 tok/s | 5,251 tok/s | 20.8 / 22.6 GiB |
| 24,576 | 1.61 / 1.65 s | 3.50 / 5.52 s | 4.19 / 5.31 s | 6,450 tok/s | 5,306 tok/s | 18.1 / 19.8 GiB |
| 32,768 | 2.21 / 2.31 s | 4.50 / 5.94 s | 5.19 / 5.82 s | 6,880 tok/s | 5,451 tok/s | 15.7 / 17.6 GiB |

A 1M prompt prefills at every budget with more than 15 GiB to spare. The budget trades cold-prefill
throughput against every stream that shares the engine with a cold long prompt: at 32,768 the decode gap
and the short-request wait during it are about twice and one and a half times those at 16,384. Which side
matters more is the owner's call on the production workload; 24,576 is the old recipe's point.

## Not measured here

- Items 5 (persistent prefix KV cache) and 6 (weight snapshots) need vLLM patches.
- `bin/spark bench`'s live check reports `display manager is active` on all four nodes as an error, so the
  harness benches with `--allow-mismatch` and fails on any other error. `doctor` also flags the site's
  4 KiB kernel (the recipe pins `7.0.0-1019-nvidia-64k`), NVMe interrupt coalescing on, and NVMe APST
  sleeping after 100 ms (11-46 ms to wake, which the Engram reader's first read after a pause waits for).
  These are host changes for the owner; none is in any arm.

## Receipts

`runs/<run>/` holds each boot's suite outputs as the head wrote them (`suite.json`, `conditions.json`,
`metrics.json`, `grid.json`, `c16.json`, `c1.json` with their request tables, `prefill.json`; the model
step's 3.8 MB text is left on the head). Run names end in the short hash of the candidate commit the nodes
booted: the working tree committed on top of the branch head at the time, not kept on the branch. Each
`metrics.json` records its scored metrics; the Runs table names the arm. `runs/ar-20261101T055815Z-18a74fde`
is the reference.
