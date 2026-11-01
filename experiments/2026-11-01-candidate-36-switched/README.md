# Direct exchange and sparknet's switched NCCL profile on candidate-36 (tp4-switched)

**Base:** upstream `7b748ce1` (candidate-36), this site's configuration (`edf9062b`), and the two
proposals as committed: `experiments/2026-11-01-direct-exchange-switched` (vLLM patch) and
`experiments/2026-11-01-switched-nccl-profile` (arm). Four DGX Sparks on one MikroTik CRS804-4DDQ,
stock 4 KiB kernel `7.0.0-1024-nvidia`, driver 580.178.04, 2026-11-01.

**Questions:**
- Does the direct exchange still pay on candidate-36? Since candidate-35's 0669, sparknet's one-shot
  takes the exchanges it accepts, and since candidate-36 only single-slot ones.
- Do the patch's new tests pass on GB10?
- Does sparknet's switched NCCL profile pay at the recipe's 16,384-token budget? On 1030 it read
  +0.6% there, unresolved.
- Do the HCAs report RDMA errors?

## Build

`vllm/series` is candidate-36's 558 patches, then the direct exchange's patch.
`bin/spark build prepare --lock` locked vLLM at patch head `e6ab6a12`, tree `bc7c9cd5`. The parent
tree is `cc0fdf58`, candidate-36's own expected tree. `build check` passed.
`build image --apply` built `spark-ds41f:candidate-36-direct`
(`sha256:b58950cc…`, 464 s, lowest MemAvailable 80.6 GiB). The GPU, TileLang and sparknet import
smokes passed, and `build distribute --apply` put it on every node.

## Arms

All arms run the one image, `tp4-switched`, and the recipe's 16,384-token budget. Each sets
`VLLM_SERVER_DEV_MODE=1` for the engine grid.

- `stock.json`: candidate-36's own `pynccl.py` and `tilelang/collectives.py` (`stock/`) mounted
  over the patched ones. Checked inside the running container on ranks 0 and 3: `ring4_relay` is
  absent from both modules.
- `direct.json`: the image as built.
- `direct-nccl.json`: `direct.json` with the four values of
  `experiments/2026-11-01-switched-nccl-profile/tp4-switched.json`.

The arms use `config/nodes-switched-c36.local.json`, a site file on every node. It is the site's
switched map without `roce_traffic_class`, which candidate-36's tooling refuses because its sparknet
marks switched traffic with class 106 itself. The running containers show `NCCL_IB_TC=106` and
`SPARKNET_ROCE_TRAFFIC_CLASS=106`.

## Release tests (GB10, `lab.py kernel-local`, cluster stopped)

`tests/direct` and `tests/stock` are `scripts/release_tests.py bundle`s of the 10 release-gate test
files that exercise the four-rank exchange (`tests/*/manifest.json`).

| Bundle | Result | The changed collectives and all-to-all tests |
|---|---|---|
| direct (image as built) | 822 passed, 1 failed, 1 skipped | 158 + 173 passed, including all 68 new or changed cases |
| stock (candidate-36's two files) | 604 passed, 1 failed, 1 skipped, 218 errors | every test in the two changed modules errors: their fixture imports `ring4_relay` |

The one failure is identical in both bundles:
`tests/models/test_deepseek_v4_1_engram.py::test_disk_engram_preparation_refreshes_graph_and_rejects_stale_rows`
(`TypeError: argument of type 'NoneType' is not iterable`). The test is in a file the patch does not
touch, and it fails the same way with candidate-36's own files, so the failure is candidate-36's.

The first stock run, `runs/tests-stock-overlay-not-applied`, is void. Its overlay sat under an extra
`vllm/` level, so the stock files were never imported and it ran the direct code. Per-test JUnit
reports: `test-results/<bundle>/`.

## Boots

Order: stock, direct, direct-nccl, direct-nccl, direct, stock, so drift spreads over the arms. Each
boot cools to 50 °C and runs `suite.json`: 64 greedy outputs, the engine grid, closed loops at 16 and
1 streams, and cold prefill at 32K, 128K and 256K (three repeats). Changes are against the first
stock boot. The HCA counters cover the bench on every node's switch port: each port appears as two
PCIe functions (`rocep1s0f1`, `roceP2p1s0f1`) that read identical counters, and the `f0` functions
are not cabled (all 208 counter files are read; the port's figures are counted once below).

| Boot | Prefill 32K / 128K / 256K tok/s | Prefill | c16 paired | c1 paired | Grid | HCA errors | ECN marks (4 ports) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `c36-stock-1` (reference) | 7,241 / 7,277 / 6,948 | 0 | 0 | 0 | 21.61 ms | 0 | 49,972 |
| `c36-stock-2` | 7,288 / 7,302 / 6,976 | +0.46% | -0.42% | -0.07% | +0.18% | 0 | 43,151 |
| `c36-direct-1` | 7,505 / 7,553 / 7,197 | +3.67% | -0.11% | +0.09% | +0.08% | 0 | 28,479 |
| `c36-direct-2` | 7,486 / 7,525 / 7,176 | +3.36% | -0.40% | -0.11% | -0.22% | 0 | 36,083 |
| `c36-nccl-1` | 7,471 / 7,493 / 7,152 | +3.02% | -0.03% | +0.05% | +1.04% | 0 | 125,151 |
| `c36-nccl-2` | 7,539 / 7,558 / 7,196 | +3.85% | -0.12% | +0.27% | +0.07% | 0 | 122,842 |

`metrics.json`'s `hca_ecn_marks` sums both functions, so it is twice the column above.

**Direct exchange, against stock.**
- Prefill +3.18% at 32K, +3.43% at 128K and +3.23% at 256K (two-boot means). The arms do not
  overlap: 3.36-3.67% against 0-0.46%.
- Grid -0.16% ±1.05 over boots (`servebench.engine.spread`): no detectable decode cost. Closed-loop
  decode is flat.
- Greedy outputs and every teacher-forced top-20 distribution are identical to the reference on
  every boot (`servebench compare`: 64/64 identical, KL 0).
- 0 RDMA errors.

**NCCL profile, against direct.** Prefill +0.13% at 32K, -0.19% at 128K and -0.17% at 256K, with
the two boots at 3.02% and 3.85%: no gain at the recipe's budget. Decode is flat. The grid read
+1.04% on one boot and +0.07% on the other (+0.62% ±5.13 over boots, unresolved). ECN marks over the
four ports rose about 3.8 times against direct (about 124,000 against 32,300 a boot), and about ten
times on the head's port (59,266 against 5,762 in the first boots), with 0 errors.

## Conclusions

1. **Direct exchange: proven on candidate-36.** Prefill +3.3%, decode unchanged, outputs bit-identical,
   no RDMA errors, and the new tests pass on GB10. The gain is larger than on 1030 (+2.7%), though
   0669 now sends single-slot exchanges to sparknet.
2. **sparknet's switched NCCL profile: no gain at the recipe's 16,384 budget on candidate-36**, and
   it marks far more traffic for congestion. The +1.0% it showed on 1030 was with 32,768-token chunks
   on a stack with three other levers. As a recipe change for `tp4-switched` it is not supported by
   these boots.

## Windows

`run.sh` (the exact commands) ran three times. Each run took `~/spark-hold.json`, kept its heartbeat,
and on exit restored production: every node at `2985b088`, `serving-best.json` booted, hold
released (2 min 1 s, 2 min 1 s and 2 min 18 s).

1. `runs/window-1.log`: built, distributed and ran both test bundles. The first boot was refused:
   the site map's `roce_traffic_class`, and stock notes citing candidate-36's tree.
2. `runs/window-2.log`: reran the stock tests with the overlay fixed. The first boot was refused
   because the workers had no `nodes-switched-c36.local.json`.
3. `runs/window.log`: all six boots.
