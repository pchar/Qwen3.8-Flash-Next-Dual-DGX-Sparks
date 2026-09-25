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

## Eval (measured — full standard suite)
- tool-eval-bench v1.8.0, full defaults, --base-url http://orcus.lan:8888/v1 from jupiter.lan (launched 16:37Z, wall 2195.7s)
- Report: /Users/pch/runs/2026/09/2026-09-25T16-36-34Z_f2e604.md
- **88/100 (121/138 pts): 57 PASS / 7 PARTIAL / 5 FAIL** — Quality 88/100, Deployability 66/100 (alpha 0.7), Responsiveness 16/100 (median turn 8.9s)
- 1 safety warning (TC-34 partial injection compliance) — same case the tune-5 89-sample also failed; not a regression
- **Reproduces 89 within eval variance**: fresh LKG runs on this sweep measure 83-88 (TASK-73.08 fresh runs 83-84, C15 88); the single 89 was a favorable sample (documented in REPORT.md §3). 88 is inside the 83-89 band and above the owner sweep gate (>=83).
- **Server survived the full suite** (health 200 at completion; 0 OOM/Xid/watchdog; MemAvail 13-14 GiB, 6 GiB floor intact) — the LKG floor-safety property re-confirmed on a fresh restart.
- Perf leg: not re-run for C15 (identical config to the LKG already measured in the full standard test suite of TASK-73.08: decodebench 1k/200k recorded in REPORT.md §3); C15 scope is reproducibility of the config + eval, which holds.

## Verdict
**PASS — LKG CONFIRMED as production candidate.** Fresh standard restart reproduces the LKG within eval variance (88 vs 89, band 83-89), full suite survived, floor intact. Best safe config of the entire sweep (tune-5 winner: GMU 0.75 / MTP0 / MNT 8192 / seqs 8 / mamba bf16 / EP false, KV ~2.98M tokens). production.env KEPT at LKG (no departure); rollback path = same profile (byte-identical), rollback commits in REPORT.md §4. Sweep (15 candidates C1-C15) complete: no config beat LKG on quality while surviving the suite; GMU* configs are capacity-only (+32% KV) but floor-unsafe under sustained 262K load.
