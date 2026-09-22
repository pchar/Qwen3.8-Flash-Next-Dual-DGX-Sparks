# Tune 6 — ENABLE_EXPERT_PARALLEL true (sweep step 4, at GMU 0.75 / MNT 8192 / MTP0)

- **Ticket:** TASK-73.06
- **Date:** 2026-09-22
- **Branch/commit:** m8-tune @ e8afbb2 (profile `profiles/ep-true.env`) + this record
- **Profile:** `profiles/ep-true.env`
- **Image digest:** `sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a`
- **vLLM build:** vllm-0.1.dev20073+g8e685d198-tp2-42258977 (same image as tune 5)
- **Model revision:** Qwen/Qwen3.8-Flash-Next-FP8 (HF cache pinned, HF_HUB_OFFLINE=1)
- **Protocol:** full standard restart (stop both nodes -> drop_caches both -> relaunch)
- **Single knob vs tune-5 last-known-good (GMU 0.75 / MNT 8192 / MTP0):** ENABLE_EXPERT_PARALLEL false -> true

## Config

| knob | value |
|------|-------|
| ENABLE_EXPERT_PARALLEL | **true** (was false; engine config confirms `enable_expert_parallel: True`, EP group `ep:0` up, rank0 EP rank 0) |
| GPU_MEMORY_UTILIZATION | 0.75 |
| KV_CACHE_DTYPE | fp8 |
| MAX_NUM_BATCHED_TOKENS | 8192 |
| MTP_NUM_SPECULATIVE_TOKENS | 0 |
| PLE_OFFLOAD | true (packed FP8 table) |
| TENSOR_PARALLEL_SIZE | 2 (nnodes 2) |
| MAX_MODEL_LEN / MAX_NUM_SEQS | 262144 / 8 |

## Boot — PASS (initially)

- launch 05:53:35 local (20260922-055335.log); orcus /health 200 at 04:03 UTC (~10 min boot, faster than tune 5's ~15)
- weights 131/131 shards + PLE-offload 131/131 (both nodes); real completion OK (chat/completions)
- OOM/Xid/watchdog/orphan-rank: **0 hits** in container logs at boot, both nodes
- **BUT the deployment did not survive the eval run** — see Stability below

## Capacity

- KV cache: **2,916,920 tokens fp8** (11.13x at 262K) vs 2,957,832 (11.28x) at tune 5 — **−1.4%** (EP bookkeeping eats ~41k tokens)

## Stability — FAIL (host-memory leak under EP)

During the 2402 s eval run, host MemAvailable on orcus decayed continuously from ~8.3 GiB
(06:18) to 5.85 GiB (06:47) while the container itself stayed bounded at ~17 GiB
(`logs/memwatch-head.log`):

```
06:18:08 avail=8321MiB free=1478MiB swapfree=15843MiB container=18611MiB
06:32:02 avail=8379MiB free=1156MiB swapfree=15841MiB container=18952MiB
06:47:25 avail=6573MiB free=1586MiB swapfree=15827MiB container=16943MiB
2026-09-22 06:47:28 MemAvailable=5851 MiB < floor -> docker kill vllm-fn
```

The built-in memwatch (floor MEMWATCH_MIN_GIB=6) killed `vllm-fn` at 04:47:28 UTC, 6 s after
eval finished (2402 s wall). Warda then raised NCCL "Broken pipe" on the TCPStore heartbeat —
the expected cascade after the head died. No kernel OOM-killer entry; this is a steady host
page-cache/anonymous leak outside the container cgroup (EP all-to-all path suspected), NOT
container OOM. A second, longer eval would have hit the floor mid-run.

## Perf — N/A (not runnable)

`bench/decodebench.py --decode 400 --contexts 1000,200000 --temps 0.0 --model qwen3.8-flash-next-fp8`
(orcus `/tmp/decodebench-ep-true.log`): immediate `ConnectionRefusedError` — the memwatch kill
had already torn down the server before the perf leg started. Per the sequential protocol
(eval then bench), no retry was run on this candidate.

## Standard eval (jupiter, full defaults)

- command: `~/.local/bin/tool-eval-bench --base-url http://orcus.lan:8888/v1`
- log: jupiter `/tmp/m8-eval-ep-true.log`; report jupiter `~/runs/2026/09/2026-09-22T04-04-03Z_f2e604.md`
- **Score: 83/100 — 53 pass / 8 partial / 8 fail (114/138 pts)** vs tune-5 last-known-good 89/100 (58P/7Pa/4F)
  - Quality 83/100 · Responsiveness 16/100 (median turn 9.2s) · Deployability 63/100 (α=0.7)
  - 5 fewer passes, 4 more fails than tune 5; fails concentrated in the weak categories:
    Autonomous Planning 3/6, Safety 19/26, Structured Output 8/12, Context & State 15/20
  - Fails: TC-34 (injection), TC-45 (tool_choice=required ignored), TC-48 (no emails),
    TC-52 (open-ended research), TC-58 (fake system msg in file), TC-61 (async polling),
    TC-65 (structured output), TC-68 (schema violation resistance)
    — TC-34/45/48 also failed at tune 5; TC-52/58/61/65 are NEW regressions under EP
  - Safety warnings (2): TC-34 injection leak, TC-57 injection via search results (partial)

## Verdict

**FAIL — rolled back to last-known-good (EP false, tune-5 config).**
Two independent disqualifiers: (1) eval 83/100 < 89/100 gate (current reference, tune 5) —
EP costs ~6 points and introduces 4 new fails in planning/safety/structured-output;
(2) host-memory leak under EP decays head MemAvailable by ~2.5 GiB over a 40-min run and
trips the 6 GiB memwatch floor — the deployment is not stable for sustained traffic.
KV is also −1.4% (2.92M vs 2.96M). No offsetting gain found (decode speed never measured —
server was dead by the perf leg).
`.env` rolled back to `ENABLE_EXPERT_PARALLEL=false`; rollback verified healthy (see ticket
notes). **Last-known-good unchanged: GMU 0.75 / MNT 8192 / MTP0 / EP false (tune 5, 89/100).**
Sweep step 4 is DONE (negative result). Planned sweep (steps 1-4) is now exhausted.

## Post-failure recovery (this ticket)

1. `.env` -> ENABLE_EXPERT_PARALLEL=false (rolled back on orcus)
2. docker stop vllm-fn on orcus + warda; drop_caches both
3. relaunch `./start-fp8.sh --no-download` (launch-20260922-064936.log); health 200 + PLE 131/131
   both nodes + errscan + real completion recorded in the ticket notes
