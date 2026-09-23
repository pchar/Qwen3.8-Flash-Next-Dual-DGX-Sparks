# TASK-73 convergence sweep — state document

Dicotomic (bisection) root-search over the Qwen3.8-Flash-Next-FP8 dual-DGX-Spark
deployment. Every run records the decision BEFORE testing: previous result ->
rule applied -> new search state -> config to test.

- Fork: pchar/Qwen3.8-Flash-Next-Dual-DGX-Sparks, branch `m8-tune`
- Image digest: `sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a`
- Model revision: Qwen/Qwen3.8-Flash-Next-FP8, HF cache pin `236dfdf285828023ca3bcd3f37366c58a3469b13`
- Cluster: orcus (head, 172.30.200.231 / 192.168.0.231, serves :8888) + warda (rank-1 worker, 172.30.200.228 / 192.168.0.228), TP2 nnodes=2
- Eval protocol (comparability rule): `~/.local/bin/tool-eval-bench --base-url http://orcus.lan:8888/v1` (full defaults, no extra flags)
- Perf protocol (after eval, sequential): on orcus `python3 bench/decodebench.py --decode 400 --contexts 1000,200000 --temps 0.0 --model qwen3.8-flash-next-fp8`
- Restart protocol: docker stop vllm-fn BOTH nodes -> `sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches` BOTH -> relaunch `./start-fp8.sh --no-download` (head side launches both ranks) -> poll /health until 200 (cap ~35 min; boot fail = up to 2 fix-retries then FAIL + rollback)
- **The 6 GiB memwatch floor (MEMWATCH_MIN_GIB=6) is NEVER lowered.**

## Parameter grid

Knobs fixed for the whole sweep (infrastructure / safety rails, not tunable):
MODEL_ID, SERVED_MODEL_NAME, MAX_MODEL_LEN=262144 (native; >262144 would need
YaRN — out of scope), YARN_ENABLE=false, KV_CACHE_DTYPE=fp8, PLE_OFFLOAD=true,
PLE_PACKED_TABLE_DIR, SKIP_PLE_PATCH=false, TENSOR_PARALLEL_SIZE=2,
MM_ENCODER_TP_MODE=data, CONTAINER_MEM_GIB=40, MEMWATCH_MIN_GIB=6,
FP8_DENSE=false, QSA_PROFILE=stock, REQUIRE_IDLE_GPU=true, EVICT_PAGE_CACHE=true,
NFS_SHARE=false, VLLM_ALLOW_LONG_MAX_MODEL_LEN=1, MASTER_PORT=50000, PORT=8888,
HEAD_IP/WORKER_IP/IFACE/IB_HCA/IB_GID_INDEX.

| knob | LKG value | allowed values / range | tested by |
|------|-----------|------------------------|-----------|
| GPU_MEMORY_UTILIZATION | 0.75 | 0.70–0.90, step 0.0005 (bisection) | 001 (0.70), 002 (0.75), 003 (0.80), C1–C7 |
| MTP_NUM_SPECULATIVE_TOKENS | 0 | {0, 3} | 005 (0), C8 (3) |
| MTP_DRAFT_VOCAB | (unset) | {unset, files/draft_vocab_en_code_47k.txt} | C8 (with MTP 3) |
| MAX_NUM_BATCHED_TOKENS | 8192 | {8192, 16384, 4096} | 004 (4096), C9 (16384), C10 (4096) |
| MAX_NUM_SEQS | 8 | {8, 16} | C11 (16) |
| MAMBA_SSM_CACHE_DTYPE | bfloat16 | {bfloat16, fp8} | C12 (fp8) |
| ENABLE_EXPERT_PARALLEL | false | {false, true} | 006 (true), C13 (true at GMU*) |

Note: 006 already tested EP=true at GMU 0.75 and it FAILED (eval 83 < 89 + host
mem leak -> memwatch kill). C13 re-tests EP=true only if GMU* > 0.75 (different
point in the space); otherwise it is recorded from 006 as excluded-unsafe.

## Search state

```
PHASE: A (GMU root search, slots C1–C7)
interval (safe, unsafe) = (0.796875, 0.80)    # 0.796875 PASS (C4); 0.80 FAIL (003)
GMU* = 0.796875 (last safe GMU tested; frozen after C7)
KV at GMU* = 3,862,456 fp8 tokens (MTP0, 14.73x at 262K), eval at GMU* = 86/100 (C4; inside 85-89 variance band, below LKG 89)
KV marginal slope: C3->C4 +101k fp8 tokens per 0.003125 GMU — consistent with C2->C3 (+135k / 0.00625); no cliff before 0.80 yet
MemAvail margin eroding with GMU: C1 16.96 / C2 13.17 / C3 12.30 / C4 12.0 GiB (post-boot, orcus) — a FAIL at 0.7984375 would likely be memwatch-driven like 003

Per-knob best-so-far (Phase B baseline, initialized to LKG):
  MTP  = 0
  MNT  = 8192
  seqs = 8
  mamba-ssm = bfloat16
  EP   = false
Excluded-unsafe so far: EP=true@0.75 (006), GMU 0.80 (003), MNT 4096@0.75 (004)
Score note: eval scores across stable GMUs so far — 0.70: 83 (001), 0.75: 85/89 (002/005/008),
0.775: 86 (C1), 0.7875: 86 (C2), 0.79375: 85 (C3), 0.796875: 86 (C4). GMU alone has not lifted the score above the LKG 89 reference.
```

## Results table

Prior context rows (from docs/tune/000..008 records; perf = 1k prose tok/s / 200k cold TTFT s):

| slot | decision (prev result -> rule -> state) | config diff | boot | KV tokens | 262K conc. | eval score | perf 1k/200k | MemAvail floor margin | verdict |
|------|------------------------------------------|-------------|------|-----------|-----------|------------|--------------|----------------------|---------|
| 001 | n/a (reference run) | GMU 0.70 baseline (MTP3) | PASS | 1,540,645 | 5.88x | 83 | 30.8 / 98.56s | 17.3 GiB | baseline established |
| 002 | 001 -> sweep step 1 -> 0.70 safe, raise | GMU 0.70->0.75 | PASS | 2,302,180 | 8.78x | 85 | 28.8 / 92.99s | n/a (pre-sysctl) | PASS — new best safe GMU |
| 003 | 002 -> step 2, raise upper bound | GMU 0.75->0.80 | **FAIL** (watchdog/memwatch kill at warmup) | 3,120,831 (sized, never served) | 11.91x (never) | n/a | n/a | collapsed 30.5->6.1 GiB in 10s | FAIL — first unsafe GMU; interval (0.75, 0.80) |
| 004 | 002 safe @0.75 -> knob probe, MNT down | MNT 8192->4096 (MTP3) | PASS | 2,587,756 | 9.87x | 83 | 29.3 / 91.83s | ok | FAIL — eval 83 < 85 gate; MNT 4096 excluded |
| 005 | 004 -> knob probe, speculation off | MTP 3->0 (draft vocab unset) | PASS | 2,957,832 | 11.28x | **89** | 22.2 / 84.30s | ok | PASS — new best eval; new LKG |
| 006 | 005 LKG -> knob probe, EP on | EP false->true | PASS then **FAIL** (host mem leak -> memwatch kill post-eval) | 2,916,920 | 11.13x | 83 | n/a (server dead) | 8.3->5.9 GiB over 40 min | FAIL — eval 83<89 + leak; EP=true@0.75 excluded-unsafe |
| 008 | 005 winner -> production promotion | = 005 config (LKG) | PASS (fresh two-node launch) | 2,957,832 | 11.28x | **89** | = 005 (22.2 / 84.30s) | 18.4 GiB after HugeTLB reclaim | LKG in production |

Candidate rows (C1–C15, filled by each sweep run):

| slot | decision (prev result -> rule -> state) | config diff | boot | KV tokens | 262K conc. | eval score | perf 1k/200k | MemAvail | verdict |
|------|------------------------------------------|-------------|------|-----------|-----------|------------|--------------|----------|---------|
| C1 | (0.75, 0.80) -> bisection midpoint -> test 0.775 | GMU 0.75->0.775 | PASS | 3,444,238 | 13.14x | 86 (55P/8Pa/6F) | 21.9 / 88.72s | 16.96/17.50 GiB | **PASS-stable / no score gain** — safe := 0.775; next interval (0.775, 0.80) |
| C2 | C1 0.775 PASS -> bisection midpoint of (0.775, 0.80) -> test 0.7875 | GMU 0.775->0.7875 | PASS | 3,626,072 | 13.83x | 86 (55P/8Pa/6F) | 22.2 / 87.52s | 13.17/16.39 GiB (post-boot) | **PASS-stable / no score gain** — safe := 0.7875; next interval (0.7875, 0.80) |
| C3 | C2 0.7875 PASS -> bisection midpoint of (0.7875, 0.80) -> test 0.79375 | GMU 0.7875->0.79375 | PASS | 3,760,932 | 14.35x | 85 (55P/7Pa/7F) | 21.9 / 86.99s | 12.30/15 GiB (post-boot) | **PASS-stable / no score gain** — safe := 0.79375; next interval (0.79375, 0.80) |
| C4 | C3 0.79375 PASS -> bisection midpoint of (0.79375, 0.80) -> test 0.796875 | GMU 0.79375->0.796875 | PASS | 3,862,456 | 14.73x | 86 (56P/7Pa/6F) | 22.2 / 89.25s | 12.0/12 GiB (post-boot) | **PASS-stable / no score gain** — safe := 0.796875; next interval (0.796875, 0.80) |
| C5 | C4 0.796875 PASS -> bisection midpoint of (0.796875, 0.80) -> test 0.7984375 | GMU 0.796875->0.7984375 | (in progress) | (pending) | (pending) | (pending) | (pending) | (pending) | (pending) |
| C6 | — | — | — | — | — | — | — | — | — |
| C7 | — | — | — | — | — | — | — | — | — |
| C8–C13 | — | — | — | — | — | — | — | — | — |
| C14 | — | — | — | — | — | — | — | — | — |
| C15 | — | — | — | — | — | — | — | — | — |

## Decision blocks (per run, appended newest-first)

### C5 (TASK-73.13) — derived 2026-09-23 from C4 result (EXECUTING 2026-09-23 — skeleton committed, boot in progress; see docs/tune/013-c5-gmu0.7984375.md)

- **Previous result:** C4 GMU 0.796875 PASS (stable boot, 0 OOM/Xid/watchdog, MemAvail 12.0/12 GiB post-boot; KV 3,862,456 = 14.73x; eval 86/100).
- **Rule applied:** PHASE A bisection — 0.796875 PASS -> safe := 0.796875; test midpoint of (0.796875, 0.80): (0.796875 + 0.80) / 2 = **0.7984375** (7-decimal).
- **New search state (pre-test):** interval (0.796875, 0.80), candidate GMU 0.7984375. Update on result: PASS -> safe := 0.7984375, next interval (0.7984375, 0.80); FAIL -> unsafe := 0.7984375, next interval (0.796875, 0.7984375).
- **Config to test:** GMU 0.7984375; everything else = LKG (MTP0 / MNT 8192 / seqs 8 / mamba bf16 / EP false / KV fp8 / PLE packed / 262144 ctx / 40 GiB container / 6 GiB memwatch floor). Profile: `profiles/c5-gmu0.79844.env`.
- **Risk note:** 0.7984375 is only 0.0015625 below the known-unsafe 0.80 and MemAvail has eroded to ~12 GiB (C1 16.96 / C2 13.17 / C3 12.30 / C4 12.0) — a memwatch kill at warmup is now the most likely outcome, as at 0.80 (003).

### C4 (TASK-73.12) — derived 2026-09-23 from C3 result (EXECUTED — see docs/tune/012-c4-gmu0.796875.md)

- **Previous result:** C3 GMU 0.79375 PASS (stable boot, 0 OOM/Xid/watchdog, MemAvail 12.30/15 GiB post-boot; KV 3,760,932 = 14.35x; eval 85/100).
- **Rule applied:** PHASE A bisection — 0.79375 PASS -> safe := 0.79375; test midpoint of (0.79375, 0.80): (0.79375 + 0.80) / 2 = **0.796875** (already at 5-decimal precision).
- **New search state (pre-test):** interval (0.79375, 0.80), candidate GMU 0.796875. Update on result: PASS -> safe := 0.796875, next interval (0.796875, 0.80); FAIL -> unsafe := 0.796875, next interval (0.79375, 0.796875).
- **Config to test:** GMU 0.796875; everything else = LKG (MTP0 / MNT 8192 / seqs 8 / mamba bf16 / EP false / KV fp8 / PLE packed / 262144 ctx / 40 GiB container / 6 GiB memwatch floor). Profile: `profiles/c4-gmu0.796875.env`.
- **Risk note:** 0.796875 is 0.003125 below the known-unsafe 0.80 and MemAvail margin is eroding (C1 16.96 / C2 13.17 / C3 12.30 GiB) — a memwatch kill at warmup is the most likely failure mode, as at 0.80 (003).
- **RESULT:** PASS — KV 3,862,456 (14.73x), eval 86/100 (56P/7Pa/6F), perf 22.2 tok/s / 89.25s cold TTFT, MemAvail 12.0/12 GiB. safe := 0.796875; GMU* = 0.796875; next interval (0.796875, 0.80).

### C3 (TASK-73.11) — derived 2026-09-23 from C2 result (EXECUTED — see docs/tune/011-c3-gmu0.79375.md)

- **Previous result:** C2 GMU 0.7875 PASS (stable boot, 0 OOM/Xid/watchdog, MemAvail 13.17/16.39 GiB post-boot; KV 3,626,072 = 13.83x; eval 86/100).
- **Rule applied:** PHASE A bisection — 0.7875 PASS -> safe := 0.7875; test midpoint of (0.7875, 0.80): (0.7875 + 0.80) / 2 = **0.79375** (already at 5-decimal precision).
- **New search state (pre-test):** interval (0.7875, 0.80), candidate GMU 0.79375. Update on result: PASS -> safe := 0.79375, next interval (0.79375, 0.80); FAIL -> unsafe := 0.79375, next interval (0.7875, 0.79375).
- **Config to test:** GMU 0.79375; everything else = LKG (MTP0 / MNT 8192 / seqs 8 / mamba bf16 / EP false / KV fp8 / PLE packed / 262144 ctx / 40 GiB container / 6 GiB memwatch floor). Profile: `profiles/c3-gmu0.79375.env`.
- **Risk note:** 0.79375 is 0.00625 below the known-unsafe 0.80 — the last bisection step before the boundary; if it FAILs the interval collapses to (0.7875, 0.79375) and GMU* stays 0.7875.
- **RESULT:** PASS — KV 3,760,932 (14.35x), eval 85/100, MemAvail 12.30 GiB. safe := 0.79375; next interval (0.79375, 0.80).

### C2 (TASK-73.10) — derived 2026-09-22 from C1 result (to be executed by next run)

- **Previous result:** C1 GMU 0.775 PASS (stable boot, 0 OOM/Xid/watchdog, MemAvail 16.96/17.50 GiB; KV 3,444,238 = 13.14x; eval 86/100).
- **Rule applied:** PHASE A bisection — 0.775 PASS -> safe := 0.775; test midpoint of (0.775, 0.80): (0.775 + 0.80) / 2 = **0.7875** (at 5-decimal precision).
- **New search state (pre-test):** interval (0.775, 0.80), candidate GMU 0.7875. Update on result: PASS -> safe := 0.7875, next interval (0.7875, 0.80); FAIL -> unsafe := 0.7875, next interval (0.775, 0.7875).
- **Config to test:** GMU 0.7875; everything else = LKG (MTP0 / MNT 8192 / seqs 8 / mamba bf16 / EP false / KV fp8 / PLE packed / 262144 ctx / 40 GiB container / 6 GiB memwatch floor). Profile: `profiles/c2-gmu0.7875.env`.


### C1 (TASK-73.09) — 2026-09-22

- **Previous result:** bootstrap state — LKG GMU 0.75 PASS (002/005/008, eval 89); GMU 0.80 FAIL (003, memwatch kill). Interval (safe, unsafe) = (0.75, 0.80).
- **Rule applied:** PHASE A bisection — test midpoint of current interval: (0.75 + 0.80) / 2 = **0.775** (already at 5-decimal precision).
- **New search state (pre-test):** interval (0.75, 0.80), candidate GMU 0.775. Update on result: PASS -> safe := 0.775, next interval (0.775, 0.80); FAIL -> unsafe := 0.775, next interval (0.75, 0.775).
- **Config to test:** GMU 0.775; everything else = LKG (MTP0 / MNT 8192 / seqs 8 / mamba bf16 / EP false / KV fp8 / PLE packed / 262144 ctx / 40 GiB container / 6 GiB memwatch floor). Profile: `profiles/c1-gmu0.775.env`.
