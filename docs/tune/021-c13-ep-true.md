# C13 (TASK-73.21) — ENABLE_EXPERT_PARALLEL false -> true at GMU* (expert parallel A/B)

- Date: 2026-09-25 (boot 14:59:54Z, eval launched ~15:20Z)
- Profile: profiles/c13-ep-true.env (commit c8cc8e6)
- Single knob vs GMU* LKG base: EP false -> true (MTP0 / MNT 8192 / seqs 8 / mamba bf16 / KV fp8 / 262144)
- Image digest: sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a
- Model revision: 236dfdf285828023ca3bcd3f37366c58a3469b13
- Launch: ./start-fp8.sh --no-download (head orcus, nnodes 2, master 192.168.0.231:50000)
- Benchmark: ssh jupiter.lan ~/.local/bin/tool-eval-bench --base-url http://orcus.lan:8888/v1 (full defaults)

## Boot
- Head StartedAt 2026-09-25T14:59:54Z (orcus); worker warda Up
- **enable_expert_parallel=True** confirmed in engine non-default args; workers named Worker_TP0_EP0 (EP split active)
- Boot ~11 min to /health 200
- KV: 3,816,998 fp8 tokens (14.56x @262K) vs GMU* LKG base 3,876,094 (C5/C7) — EP shards the expert weights, slightly lower KV
- PLE registered both nodes (Registrations complete); OOMKilled=false
- Post-boot host MemAvailable ~8 GiB (6 GiB floor intact; margin tighter than LKG base)
- Risk baseline: 006 showed EP-true FAILs at GMU 0.75 (eval 83 + host-RAM leak 8.3->5.9 GiB over 40 min -> memwatch kill). At GMU* the post-boot margin (~7.8-8 GiB) is tighter, so the same leak is expected to trip the floor.

## Eval / perf
IN FLIGHT — tool-eval-bench launched on jupiter 2026-09-25 ~15:20Z (pid 13178, log /Users/pch/tune/c13-eval.log, wall-time capture required). Decodebench (1k/200k) runs after eval. Verdict + measured evidence to be appended by the next run.

## Verdict
PENDING — expected FAIL-survival (EP leak -> 6 GiB floor), matching 006. Keep EP=false (LKG) unless a full suite (boot+eval+decodebench) passes cleanly.
