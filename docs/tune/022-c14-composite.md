# C14 (TASK-73.22) — COMPOSITE: all C8-C13 per-knob winners at GMU* (LKG base at GMU*)

- Date: 2026-09-25
- Profile: profiles/c14-composite.env
- Diff vs LKG (profiles/production.env): GPU_MEMORY_UTILIZATION 0.75 -> 0.799609375 ONLY (composite of resolved knobs = LKG knob values: MTP0 / MNT 8192 / seqs 8 / mamba bf16 / EP false)
- Diff vs C7 base: none (byte-identical settings) — C14 re-runs the C7-shape config as the formal composite
- Image digest: sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a
- Model revision: 236dfdf285828023ca3bcd3f37366c58a3469b13 (Qwen/Qwen3.8-Flash-Next-FP8)
- Launch: ./start-fp8.sh --no-download (head orcus, nnodes 2, master 192.168.0.231:50000)
- Benchmark: ssh jupiter.lan ~/.local/bin/tool-eval-bench --base-url http://orcus.lan:8888/v1 (full defaults); then on orcus python3 bench/decodebench.py --decode 400 --contexts 1000,200000 --temps 0.0 --model qwen3.8-flash-next-fp8

## Decision (pre-test)

Previous result: C13 EP-true FAIL-survival (6 GiB floor kill, host-RAM leak, 006 signature) -> EP RESOLVED false.
Rule: C14 = composite of per-knob best-so-far: MTP 0 / MNT 8192 / seqs 8 / mamba bf16 / EP false, at GMU* 0.799609375 (FROZEN).
=> Composite = LKG base at GMU* = C7 settings. Expected: FAIL-survival at the 6 GiB floor under sustained 262K load (C10/C11 pattern) OR full-suite PASS making GMU* the capacity candidate for C15.

## Run state (FILLED)

- boot: PENDING
- OOM/Xid/watchdog/memwatch: PENDING
- KV fp8 tokens: PENDING
- eval score: PENDING
- perf 1k/200k: PENDING
- MemAvailable: PENDING
- verdict: PENDING
