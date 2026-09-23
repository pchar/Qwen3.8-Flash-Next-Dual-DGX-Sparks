# Tune C12 — mamba-ssm fp8 at GMU* (PHASE B knob 4) — in flight

- **Ticket:** TASK-73.20 (slot C12, PHASE B knob 4: MAMBA_SSM_CACHE_DTYPE bfloat16->fp8)
- **Date:** 2026-09-23
- **Branch/commit:** m8-tune (skeleton 45801e2; result commit appended)
- **Profile:** `profiles/c12-mamba-fp8.env`
- **Image digest:** `sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a`
- **Model revision:** Qwen/Qwen3.8-Flash-Next-FP8 (HF cache pin `236dfdf285828023ca3bcd3f37366c58a3469b13`)
- **Protocol:** full standard restart; eval after boot; perf after eval (sequential)

## Decision (recorded BEFORE deploy)

- GMU* knife-edge: C7 (MNT 8192/MTP0) survives decode load; C8 (MTP3) + C9 (MNT 16384) both SIGKILL 137 GPU-OOM. KV headroom moot (scheduler-limited at seqs=8). Quality optimum LKG 89.
- MTP -> 0 (C8 crash), MNT -> 8192 (C9 crash; 004 4096 FAIL), seqs -> 8 (deferred), EP excluded (006).
- **C12 = MAMBA_SSM_CACHE_DTYPE bf16 -> fp8** — the 36 GDN layers keep BF16 recurrent state per request (README: memory hog); fp8 shrinks it, freeing GPU room to stabilize GMU* and possibly raise KV. Keep fp8 iff eval >= 86 AND survives decodebench.

## Config (single knob vs GMU* C7)

| knob | value |
|------|-------|
| GPU_MEMORY_UTILIZATION | 0.799609375 (unchanged = GMU*) |
| MAMBA_SSM_CACHE_DTYPE | **bfloat16 -> fp8** |
| MTP_NUM_SPECULATIVE_TOKENS | 0 |
| MAX_NUM_BATCHED_TOKENS | 8192 |
| KV_CACHE_DTYPE | fp8 |
| ENABLE_EXPERT_PARALLEL | false |
| TENSOR_PARALLEL_SIZE | 2 (nnodes 2) |
| MAX_MODEL_LEN / MAX_NUM_SEQS | 262144 / 8 |

## Boot

- **FAIL at launch** — `vllm serve` rejects `--mamba-ssm-cache-dtype fp8`: `invalid choice: 'fp8' (choose from 'auto','bfloat16','float16','float32')`. Container Exited (2) during launch; memwatch/watchdog up but no engine. This vLLM build (v0.1.dev20073) does NOT support fp8 SSM cache dtype.

## Eval (standard)

- NOT RUN (no engine)

## Perf leg (decodebench)

- NOT RUN (no engine)

## Verdict

- **FAIL (invalid knob)** — mamba-ssm-cache-dtype fp8 is unsupported by this vLLM build; the knob space has no valid alternative (bf16 is LKG). MAMBA knob RESOLVED to bfloat16 (LKG); fp8 excluded-invalid. No quality/capacity lever here.
