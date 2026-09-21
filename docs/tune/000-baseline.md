# Tune 0 — baseline (validated config, no change)

- **Ticket:** TASK-73.01
- **Date:** 2026-09-22
- **Branch/commit:** m8-tune @ 86c9a70 (scaffold) — result commit pending
- **Profile:** `profiles/baseline-gmu0.70.env`
- **Image digest:** `sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a`
- **vLLM build:** vllm-0.1.dev20073+g8e685d198-tp2-98f6849f
- **Model revision:** Qwen/Qwen3.8-Flash-Next-FP8 (HF cache pinned, HF_HUB_OFFLINE=1)
- **Protocol:** full standard restart (stop vllm-fn both nodes → drop_caches both → relaunch from m8-tune)

## Config (the validated baseline)

| knob | value |
|------|-------|
| GPU_MEMORY_UTILIZATION | 0.70 |
| KV_CACHE_DTYPE | fp8 |
| MAX_NUM_BATCHED_TOKENS | 8192 |
| MTP_NUM_SPECULATIVE_TOKENS | 3 |
| ENABLE_EXPERT_PARALLEL | false |
| PLE_OFFLOAD | true (packed FP8 table) |
| CONTAINER_MEM_GIB / MEMWATCH_MIN_GIB | 40 / 6 |
| TENSOR_PARALLEL_SIZE | 2 (nnodes 2, orcus head + warda worker) |
| MAX_MODEL_LEN | 262144 |
| MAX_NUM_SEQS | 8 |

## Boot — PASS

- orcus /health 200 on poll 26 (~13 min total boot)
- weights: main 553.1 s + MTP 39.6 s
- PLE packed table 131/131 shards on **both** orcus and warda
- real completion OK (5 prompt / 8 completion tokens)
- OOM/Xid/watchdog/orphan-rank: **0 hits** in both nodes' logs
- host memory floor: memwatch avail 17.3 GiB (floor 6 GiB); container 22.5 GiB (cap 40 GiB); host avail orcus 16 GiB / warda 28 GiB

## Capacity

- KV cache: **1,540,645 tokens fp8** (GPU KV cache size log line)
- 262K-context concurrency: **5.88×**
- preemptions/recompute during eval+bench window: 0

## Perf

`bench/decodebench.py --decode 400 --temps 0.0 --model qwen3.8-flash-next-fp8` (log: orcus:/tmp/m8-decodebench-baseline.log)

| context | content | ptok | TTFT s | decode tok/s |
|---|---|---|---|---|
| 1,000 | prose | 1,076 | 1.26 | 30.8 |
| 1,000 | code | 1,073 | 0.95 | 37.5 |
| 1,000 | entropy | 1,077 | 0.72 | 34.7 |
| 1,000 | copy (best case) | 1,089 | 0.75 | 60.9 |
| 200,000 | prose (cold, 1st req) | 200,076 | 98.56 | 32.5 |
| 200,000 | code | 200,073 | 1.63 | 35.6 |
| 200,000 | entropy | 200,077 | 1.64 | 34.2 |
| 200,000 | copy (best case) | 200,089 | 2.50 | 55.6 |

- note: 200k prose TTFT 98.56s is the cold first request (cudagraph/compile + uncached 200k prefill); subsequent same-context tasks hit prefix cache (~1.6s). Decode is context-insensitive as expected (MTP-dominated): 30-37 tok/s prose/code at both 1k and 200k.

- request errors: 0 (one xgrammar FSM-advance error class seen in engine log during structured tool calls — pre-existing, not a config regression; tracked as known issue)

## Standard eval (jupiter, full defaults)

- command: `~/.local/bin/tool-eval-bench --base-url http://orcus.lan:8888/v1`
- report: jupiter `~/runs/2026/09/2026-09-21T23-04-47Z_f2e604.md`
- **Score: 83/100** — 53 pass / 8 partial / 8 fail (114/138 pts)
  - Quality 83/100 · Responsiveness 31/100 (median turn 5.1 s) · Deployability 67/100 (α=0.7)
  - Weakest category: K Safety & Boundaries (62%)
  - 304,430 tokens total · 0.4 pts/1K
  - Safety warnings: TC-33 hallucination, TC-34 partial injection compliance, TC-43 empty web_search query, TC-58 fake-key leak
- Leaderboard: this run is the reference entry for `qwen3.8-flash-next-fp8`.
  **Success gate for all later tune tickets: score ≥ 83 at equal-or-higher config.**

## Verdict

**BASELINE ESTABLISHED — PASS.** Config unchanged (this is the reference run).
Last known-good = this commit/config. Sweep may proceed from here.
