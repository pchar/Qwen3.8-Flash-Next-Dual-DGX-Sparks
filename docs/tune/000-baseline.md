# Tune 0 — baseline (validated config, no change)

- **Ticket:** TASK-73.01
- **Date:** 2026-09-22
- **Branch/commit:** m8-tune @ _pending SHA_
- **Profile:** `profiles/baseline-gmu0.70.env`
- **Image digest:** `sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a`
- **Model revision:** Qwen/Qwen3.8-Flash-Next-FP8 (HF cache pinned, `HF_HUB_OFFLINE=1`)
- **Protocol:** full standard restart (stop both, drop caches both, relaunch)

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
| TENSOR_PARALLEL_SIZE | 2 (nnodes 2) |
| MAX_MODEL_LEN | 262144 |
| MAX_NUM_SEQS | 8 |

## Boot

- orcus health 200: _pending_
- warda PLE registered: _pending_
- OOM/Xid/watchdog/orphan-rank: _pending_
- real completion request: _pending_

## Capacity

- KV-cache GiB / tokens: _pending (from boot log)_
- 262K-context concurrency: _pending_
- peak host memory (both nodes): _pending_

## Perf

- TTFT (p50/p95): _pending_
- prefill tok/s: _pending_
- decode tok/s: _pending_
- aggregate tok/s: _pending_
- request errors / preemptions: _pending_

## Standard eval (jupiter, full defaults)

- command: `~/.local/bin/tool-eval-bench --base-url http://orcus.lan:8888/v1`
- run id: _pending_
- score: _pending_
- leaderboard position: _pending_

## Verdict

_pending_
