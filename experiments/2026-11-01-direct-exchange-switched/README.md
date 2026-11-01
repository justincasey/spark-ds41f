# Direct exchange on a switched fabric (site)

**Base:** spark-ds41f-1030 (lock `upstreams.lock.json` at upstream `b7eac2a0`; vLLM series rc14, head `b6de07aa`,
tree `ea1281f8`), shape `tp4-switched`, four DGX Sparks on one MikroTik CRS804-4DDQ, stock 4 KiB kernel
`7.0.0-1024-nvidia`, driver 580.178.04, `kho=off`.

**One variable:** vLLM patch `vllm/0001-DS4.1-Exchange-directly-on-a-switched-fabric.patch` on top of the 1030 series
(`vllm/series`: the rc14 series re-pathed, then this patch; head `cecd9f79`, tree `6ad77da9`; `source.json`).

## Hypothesis

At four ranks, the rank-order reduce-scatter's chunk exchange and PyNCCL's `all_to_all_rows` relay half of the
opposite rank's block through each neighbour, because a switchless four-node ring has no cable between ranks two
apart. `tp4-switched` keeps that relay although every pair is one hop. Exchanging directly when sparknet renders
`SPARKNET_ROCE_TOPOLOGY=direct` removes a serialized group and a quarter of the bytes each rank sends. Outputs are
bit-identical: only bytes move, and the rank-order sum is unchanged.

Any other topology value, or none, keeps the relay, so ring shapes (`oneshot-ring4`, `nccl-ring`) are unaffected.

## Prior evidence (r6w, not this base)

Same site, same switch, 2026-10-31, relayed against direct build, one boot each: prefill +3.4% at 32K, +1.2% at
128K, +1.1% at 512K; decode c=1 within noise (the path is prefill only); the four switch ports sent 2,072 GB against
1,757 GB (-15.2%) during the prefill bench; switch drops 0; token ids and top-5 logprobs identical on six short and
three long (6K-109K) prompts; 1M needle PASS.

## Acceptance

- `bin/spark doctor` reports `configuration OK`; `build prepare` reproduces head `cecd9f79` and tree `6ad77da9`.
- Outputs bit-identical to stock 1030 on the same prompts (greedy token ids and top-5 logprobs).
- Prefill not slower than stock 1030 on `tp4-switched`, measured twice (a boot and a repeat boot).
- Switch: 0 tail drops; HCAs: no new `out_of_sequence`, `packet_seq_err` or `local_ack_timeout_err`.

## Runs

Measured 2026-11-01 in `experiments/2026-11-01-site-levers` (runs and receipts there), `tp4-switched`, one
session. Stock 1030 is the `spark-ds41f:1030-direct` image with 1030's `pynccl.py` and
`tilelang/collectives.py` mounted over the patched ones (`overlay/stock/`, vLLM blobs `64d16a71`, `5ed772e5`);
the direct arm is the image as built. Two boots each; changes against the first stock boot.

| Boot | Arm | Prefill 32K / 128K / 256K | Prefill (geomean) | c16 paired | c1 paired | Engine grid |
|---|---|---:|---:|---:|---:|---:|
| ref | stock 1030 | 6,796 / 6,835 / 6,565 tok/s | 0 | 0 | 0 | 24.75 ms |
| A/A | stock 1030 | 6,813 / 6,851 / 6,581 | +0.23% | -0.20% | +0.01% | +0.19% |
| A | direct | 7,021 / 7,049 / 6,754 | +3.10% | -0.01% | -0.01% | +0.58% |
| B | direct | 6,990 / 6,998 / 6,725 | +2.55% | +0.09% | -0.16% | +0.53% |

- **Prefill:** +2.7% (two-boot means, +2.83% against +0.12%), at every size.
- **Decode:** closed-loop paired decode unchanged. The engine grid reads +0.46% over boots
  (`servebench.engine.spread`, boot as the sample): ±1.05% at 95%, no detectable change.
- **Outputs:** greedy token ids of 64 prompts (256 tokens each) identical to stock on both boots (the
  `model` step's corpus digest; top-5 logprobs were not compared).
- Switch and HCA error counters were not read in this session.

**Conclusion:** greedy token ids and prefill meet the acceptance; the top-5 logprob comparison and the switch
and HCA counter check are still open.
