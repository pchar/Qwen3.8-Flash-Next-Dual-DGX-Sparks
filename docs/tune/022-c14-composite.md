# C14 (TASK-73.22) — COMPOSITE: all C8-C13 per-knob winners at GMU* (LKG base at GMU*)

- Date: 2026-09-25 (boot 15:51:59Z, eval 16:05-16:19Z)
- Profile: profiles/c14-composite.env (commit 786359e)
- Diff vs LKG (profiles/production.env): GPU_MEMORY_UTILIZATION 0.75 -> 0.799609375 ONLY (composite of resolved knobs = LKG knob values: MTP0 / MNT 8192 / seqs 8 / mamba bf16 / EP false)
- Diff vs C7 base: none (byte-identical settings) — C14 re-runs the C7-shape config as the formal composite
- Image digest: sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a
- Model revision: 236dfdf285828023ca3bcd3f37366c58a3469b13 (Qwen/Qwen3.8-Flash-Next-FP8)
- Launch: ./start-fp8.sh --no-download (head orcus, nnodes 2, master 192.168.0.231:50000)
- Benchmark: ssh jupiter.lan ~/.local/bin/tool-eval-bench --base-url http://orcus.lan:8888/v1 (full defaults); perf leg not run (server dead)

## Decision (pre-test)

Previous result: C13 EP-true at GMU* FAIL-survival (6 GiB floor kill, host-RAM leak, 006 signature) -> EP RESOLVED false.
Rule: C14 = composite of per-knob best-so-far: MTP 0 / MNT 8192 / seqs 8 / mamba bf16 / EP false, at GMU* 0.799609375 (FROZEN).
=> Composite = LKG base at GMU* = C7 settings. Expected: FAIL-survival at the 6 GiB floor under sustained 262K load (C10/C11 pattern) OR full-suite PASS making GMU* the capacity candidate for C15.

## Run state (FILLED)

- boot: PASS — container started 2026-09-25T15:51:59Z (orcus), /health 200 at 16:04:25Z (~12.5 min), OOMKilled=false, 0 Xid/OOM/watchdog at boot
- capacity: KV 3,901,854 fp8 tokens (14.88x @262K — highest of the sweep, +25.8k vs C7 3,876,094: allocator variance); PLE "Registrations complete" both nodes (orcus 16:02:13Z, warda 15:55:54Z)
- host mem: orcus MemAvailable ~7 GiB post-boot (tighter than C7 ~8.7), drained ~6.6-7.0 GiB during eval
- eval: server DIED mid-eval — 6 GiB host memwatch floor fired at **18:19:13 local (16:19:13Z), MemAvailable 6017 MiB < 6 GiB, docker kill vllm-fn** (logs/memwatch-head.log), at TC-23 (17/34 scenarios done; TC-01..TC-22 passing, 21 PASS / 1 PARTIAL up to death). Bench reported 31/100, 43/138 pts — an ARTIFACT: 47 FAILs are "All connection attempts failed" / "peer closed connection" after the kill; NOT a quality measurement (same artifact family as C13 74/138).
- perf leg: never ran (server dead)
- verdict: **FAIL-survival** — composite = LKG base at GMU* dies at the 6 GiB floor ~14 min into a standard eval, confirming the C10/C11/C13 pattern: NO GMU* config survives the full standard suite; C7 was the only GMU* run to finish a full suite (86/100). Composite does NOT beat LKG (89/100, full-suite-surviving) — **convergence verdict: LKG (GMU 0.75 base) is the production winner; GMU* base is capacity-only (KV +30%: 2,957,832 -> ~3.9M tokens, 14.88x) but unusable under sustained 262K load.**
- rollback: LKG redeployed (profiles/production.env, .env restored from .env.lkg-backup-20260925; stop + drop_caches both nodes + relaunch), /health 200 verified post-restart.
