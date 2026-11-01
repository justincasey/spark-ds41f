# Direct exchange at four ranks on a switched fabric

## Summary

At four ranks the rank-order reduce-scatter's chunk exchange (`tilelang/collectives.py`
`exchange_into`, which sequence-parallel prefill's row exchange also uses) and PyNCCL's
`all_to_all_rows` relay half of the opposite rank's block through each neighbour, because on the
switchless four-node ring ranks two apart share no cable. `tp4-switched` keeps that relay although
every pair is one hop through the switch. The patch exchanges directly with every peer, in one group,
when sparknet renders `SPARKNET_ROCE_TOPOLOGY=direct` (as `tp4-switched` does), and keeps the relay
for any other value or none. Only bytes move: the rank-order sum and every part are the same bits.

On four GB10 behind one switch, against stock spark-ds41f-1030, two boots each: **prefill +2.7%** at
32K-256K, decode unchanged, outputs identical.

## Patch

- `vllm/0001-perf-dsv41-exchange-directly-at-four-ranks-on-a-switched-fabric.patch`: on candidate-36's
  vLLM series (`experiments/2026-10-26-spark-ds41f-candidate/candidate-36/vllm/series`: vLLM
  `19f2c20e` plus 0001..0673, expected tree `cc0fdf58`), checked by applying that series (it
  reproduces `cc0fdf58`) and then this patch with `git am --3way`. One conflict, in
  `collectives.py`'s module docstring (0634 had reworded "Fewer ranks" as "Other world sizes"),
  resolved by keeping both. Both code hunks apply unchanged.
- `rc14/0001-DS4.1-Exchange-directly-on-a-switched-fabric.patch`: the measured build, the same code
  change on spark-ds41f-1030's series (rc14, head `b6de07aa`; with it head `cecd9f79`, tree
  `6ad77da9`), built as `spark-ds41f:1030-direct`.

Code changes (24 lines): `pynccl.ring4_relay()`, cached, is false only for
`SPARKNET_ROCE_TOPOLOGY=direct`. `PyNcclCommunicator.all_to_all_rows` and `collectives.exchange_into`
take the four-rank relay only while it is true. Every other world size already exchanges directly.

**Interaction with candidate-35's 0669.** 0669 sends SP `reduce_scatter`, `all_to_all_rows` and
`exchange_into` to sparknet's one-shot when sparknet accepts them, and candidate-36 limits that to
single-slot messages. Larger prefill exchanges still take the PyNCCL path this patch changes. The
measurement below predates 0669, so on candidate-36 the gain is bounded by the share of exchanges
that still reach PyNCCL. It has not been measured there.

## Tests

The patch extends two vLLM test modules:

- `tests/kernels/test_deepseek_v41_tilelang_collectives.py`: the threaded fake fabric gains a
  switched mode, where every pair is linked. `test_switched_four_ranks_exchange_directly_with_the_same_bits`
  checks that four ranks under `direct` send to every peer, opposite included, in one group per
  exchange (the relay uses more). It also checks that every part and every rank-order sum equals the
  relay's, bit for bit. `test_every_other_topology_keeps_the_ring_relay` checks that unset, empty,
  `ring4`, `switched` and `DIRECT` keep the ring relay (the ring fabric refuses the opposite rank).
  An autouse fixture unsets the variable and clears `ring4_relay`'s cache around every test, so the
  ring tests do not depend on the environment they run in.
- `tests/models/test_deepseek_v41_sp_project_reduce_scatter.py`:
  `test_row_all_to_all_over_cabled_pairs` runs on both topologies. Under `direct`, pairs two apart
  exchange.

Both modules need TileLang on a CUDA platform (the first imports TileLang through its zero-fill
fixture, the second skips without a device). They are written and compile, and have **not been
run**; they belong to the GB10 release-test gate (`config/release-tests.json`). In a container with no
GPU device they error in TileLang setup ("No CUDA GPUs are available") before any test runs.

## Measurement (spark-ds41f-1030, `tp4-switched`)

Four DGX Sparks on one MikroTik CRS804-4DDQ (PFC/ECN, `NCCL_IB_TC=106`), stock 4 KiB kernel
`7.0.0-1024-nvidia`, driver 580.178.04, 2026-11-01, one session. **Stock** is the
`spark-ds41f:1030-direct` image with 1030's `pynccl.py` and `collectives.py` (`stock/`, vLLM blobs
`64d16a71` and `5ed772e5` of tree `ea1281f8`) mounted over the patched ones, so both arms run one
image. **Direct** is the image as built. Each boot runs one suite (about 21 minutes):

- greedy outputs of 64 prompts, 256 tokens each, with top-20 logprobs;
- the engine grid;
- closed loops at 16 and 1 streams (60 s warm-up, 300 s measured);
- cold real-text prefill at 32K, 128K and 256K, three repeats each.

All changes are against the first stock boot. Decode is servebench's paired per-request decode rate.

| Boot | Arm | Prefill 32K / 128K / 256K tok/s | Prefill (geomean) | c16 paired | c1 paired | Engine grid (decode geomean) |
|---|---|---:|---:|---:|---:|---:|
| ref | stock | 6,796 / 6,835 / 6,565 | 0 | 0 | 0 | 24.75 ms |
| A/A | stock | 6,813 / 6,851 / 6,581 | +0.23% | -0.20% | +0.01% | +0.19% |
| A | direct | 7,021 / 7,049 / 6,754 | +3.10% | -0.01% | -0.01% | +0.58% |
| B | direct | 6,990 / 6,998 / 6,725 | +2.55% | +0.09% | -0.16% | +0.53% |

- **Prefill:** +2.7% (two-boot means: +2.83% against +0.12%), at every size.
- **Decode:** closed-loop paired decode unchanged. The grid reads +0.46% over boots
  (`servebench.engine.spread`, boot as the sample), ±1.05% at 95%: no detectable change. The changed
  paths run at prefill.
- **Outputs:** greedy token ids identical on both boots. Teacher-forced over 16,384 positions, every
  top-20 distribution is identical to stock (KL 0, top-1 agreement 100%), from
  `python3 -m servebench compare <stock>/model.json.gz <direct>/model.json.gz`.
- **Not read:** switch drop counters and HCA error counters.

Receipts: `runs/<run>/`, with each boot's `suite.json` (suite definition and deployment identity),
`conditions.json`, `metrics.json`, `grid.json`, `c16.json`, `c1.json` (request tables) and
`prefill.json`. `arm.json` is the arm the boot ran, which matches the `cluster_config_sha256` the bench
recorded. The arms extend that site's configuration, and the stock arm mounts `stock/` from the
site's experiment path. The model steps' text (3.8 MB each) is not kept here.

## Still open

- Build and measure on candidate-36, where 0669 already takes single-slot exchanges.
- Run the two test modules in the release-test gate.
- Read the switch's drop and the HCAs' error counters over a prefill bench.
