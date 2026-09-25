# C15 (TASK-73.23) — PRODUCTION CANDIDATE CONFIRMATION: fresh re-deploy of best safe config (LKG) + full verification

- Date: 2026-09-25 (fresh standard restart 16:22:20Z, eval launched ~16:37Z)
- Profile: profiles/production.env (LKG: GMU 0.75 / MTP0 / MNT 8192 / seqs 8 / mamba bf16 / EP false / KV fp8 / 262144) — unchanged, kept at LKG with decision documented
- Fresh restart protocol: docker stop vllm-fn (orcus+warda) + drop_caches both nodes + relaunch via ./start-fp8.sh --no-download (performed as the C14 rollback; no config departure — engine cmdline verified = production.env values)
- Image digest: sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a
- Model revision: 236dfdf285828023ca3bcd3f37366c58a3469b13 (Qwen/Qwen3.8-Flash-Next-FP8)
- Launch: ./start-fp8.sh --no-download (head orcus, nnodes 2, master 192.168.0.231:50000)
- Benchmark: ssh jupiter.lan ~/.local/bin/tool-eval-bench --base-url http://orcus.lan:8888/v1 (full defaults)

## Why C15 = LKG
Convergence verdict (C14): no GMU* (0.799609375) config survives the full standard suite (6 GiB host floor kill — C10/C11/C13/C14 pattern); only C7 shape completed a GMU* full suite (86/100 < 89). Best safe config = LKG (GMU 0.75 / MTP0 / MNT 8192 / EP false), tune-5 sample 89/100.

## Fresh two-node launch verification (measured)
- Head StartedAt 2026-09-25T16:22:20.808719171Z (orcus); worker warda Up; restarts=0, OOMKilled=false
- /health 200; /v1/models qwen3.8-flash-next-fp8 max_model_len 262144; real completion returned generated text
- PLE "Registrations complete (dp_size=1, tp_size=2)" BOTH nodes (orcus 16:32:23Z, warda 16:26:02Z)
- Engine cmdline: GMU 0.75 / seqs 8 / MNT 8192 (no departure from production.env)
- KV: 2,979,046 fp8 tokens (11.36x @262K; tune-5 original 2,957,832 = 11.28x — allocator variance, +0.7%)
- 0 OOM / 0 Xid / 0 watchdog events; 6 GiB floor intact (MemAvail 14-19 GiB)

## Eval (pending — in flight)
