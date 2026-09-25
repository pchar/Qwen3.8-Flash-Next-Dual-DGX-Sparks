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

## Eval / perf (measured)

- tool-eval-bench v1.8.0, launched ~15:20Z (pid 13178, log /Users/pch/tune/c13-eval.log), report /Users/pch/tune/runs/2026/09/2026-09-25T15-10-41Z_f2e604.md
- **Server died mid-eval:** 6 GiB host memwatch floor killed orcus vllm-fn at **17:25:12 local (15:25:12Z), MemAvailable 6103 MiB < 6 GiB** (memwatch-head.log), ~5 min into the run. Host-RAM leak, identical failure family to 006 (8.3->5.9 GiB over 40 min at 0.75).
- Eval finished (wall 872.2s) with the tail meaningless: **36 PASS / 2 PARTIAL / 31 FAIL, 74/138, Quality 54/100** — but 31 of the 31 FAILs are All connection attempts failed (server already dead). Pre-death the run was passing through ~TC-37, so the 74 is a connection-failure artifact, NOT a quality measurement.
- Decodebench (1k/200k) never ran (server dead).
- No GPU OOM / no Xid: the kill is the **host** RAM floor, exactly the 006 EP-leak signature.

## Verdict
**FAIL-survival (confirmed).** EP-true at GMU* leaks host RAM and trips the 6 GiB floor during the standard eval, reproducing 006. Eval gate not meaningfully met (server died ~5 min in); decodebench never ran. **ENABLE_EXPERT_PARALLEL RESOLVED to false (LKG); EP=true excluded-unsafe** at both 0.75 (006) and GMU* (C13).
