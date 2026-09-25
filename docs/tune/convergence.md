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
PHASE: A CLOSED (C1–C7 all PASS). PHASE B CLOSED (C8–C13 all resolved). C14 composite EXECUTED (FAIL-survival). C15 EXECUTED (PASS) — SWEEP COMPLETE. FINAL: LKG (GMU 0.75 / MTP0 / MNT 8192 / seqs 8 / mamba bf16 / EP false) = production candidate, confirmed on fresh restart (88/100). GMU* = 0.799609375 capacity-only, floor-unsafe.
interval (safe, unsafe) = (0.799609375, 0.80)   # C7 PASS (C7); 0.80 FAIL (003)
GMU* = 0.799609375 (last safe GMU tested; FROZEN — Phase A closed)
KV at GMU* = 3,876,094 fp8 tokens (MTP0, 14.79x at 262K), eval at GMU* = 86/100 (C7; inside 83-89 variance band, below LKG 89)
KV marginal slope: C5->C6 −21.2k fp8 tokens per +0.00078125 GMU — NON-MONOTONIC (allocator block granularity); marginal gain has hit ~0 before the 0.80 cliff, so C7 will move KV ~0 either way
MemAvail margin eroding with GMU: C1 16.96 / C2 13.17 / C3 12.30 / C4 12.0 / C5 11 / C6 9 GiB (post-boot, orcus) — floor intact at C6 (post-perf 7 GiB); C7 at 0.799609375 is 0.000390625 from 0.80, a memwatch kill is the leading risk, as at 0.80 (003).
C6 perf caveat: 200k rows measured WARM (cold prefill absorbed during the 39-min eval; re-run confirmed still warm) — cold-200k TTFT not comparable to C1-C5; decode ~24 tok/s (warm) also not directly comparable

Per-knob best-so-far (Phase B baseline, initialized to LKG):
PHASE B decision (C11, derived from C8-C12 + reframe):
- Knobs resolved: MTP -> 0 (C8 crash), MNT -> 8192 (C9 crash; 004 4096 FAIL), MAMBA -> bf16 (C12 fp8 invalid in this vLLM build), EP excluded (006: -6pts + mem leak).
- GMU* knife-edge: only C7 shape survives decode load; KV headroom moot (scheduler-limited at seqs=8). Quality optimum LKG 89.
- C11 = MAX_NUM_SEQS 8 -> 16 at GMU*, MTP0/MNT8192/mamba-bf16. Rationale: README identifies MAX_NUM_SEQS as the real concurrency lever (scheduler admits <=8 at 262K); raising to 16 lets more requests be resident. RISK: 16 concurrent GDN recurrent-states add GPU memory at knife-edge GMU* — near-certain crash. Keep seqs 16 iff eval >= 86 AND survives decodebench (else revert to 8).
PHASE B decision (C12, derived from C8-C9 FAIL + reframe):
- GMU* 0.799609375 knife-edge: C7 (MNT 8192/MTP0) survives decode load; MTP3 (C8) and MNT16384 (C9) both SIGKILL 137 GPU-OOM. KV headroom moot (scheduler-limited at seqs=8). Quality optimum is LKG 89.
- MTP knob -> 0 (C8 crash). MNT knob -> 8192 (C9 crash; 004 4096 FAIL). seqs -> 8 (untested; deferred). EP excluded (006: -6pts + mem leak).
- C12 = MAMBA_SSM_CACHE_DTYPE bfloat16 -> fp8 at GMU*, MTP0, MNT 8192. Rationale: the 36 GDN layers keep BF16 recurrent state per request (README: memory hog); fp8 shrinks it, freeing GPU room to stabilize GMU* and possibly raise KV. Genuinely untested. Keep mamba fp8 iff eval >= 86 AND survives decodebench (else revert to bf16).
PHASE B decision (derived from C1-C7 + prior 001-008):
- GMU* FROZEN = 0.799609375 (KV 3,876,094, 14.79x; eval 86, below LKG 89).
- Dicotomic candidate space (one knob per test): MTP {0, 3+47k}, MNT {8192,16384,4096}, seqs {8,16}, mamba-ssm {bf16,fp8}, EP {false,true}.
- Per-knob best-so-far (init to LKG): MTP 0 / MNT 8192 / seqs 8 / mamba bf16 / EP false.
- Excluded-unsafe so far: GMU 0.80 (003), EP=true@0.75 (006), MNT 4096@0.75 (004).
- C8 = MTP_NUM_SPECULATIVE_TOKENS 3 + MTP_DRAFT_VOCAB 47k at GMU* (speculation A/B: MTP3 vs MTP0 best-so-far). RESULT: **FAIL** — eval 85/100 completed, but engine SIGKILL 137 mid-decodebench (GPU OOM NV_ERR_NO_MEMORY); MTP3 draft head + KV reservations pushed GPU mem over the limit. MTP3+47k EXCLUDED-unsafe; MTP knob reverts to 0 (best-so-far).
- C9 = MAX_NUM_BATCHED_TOKENS 8192->16384 at GMU*, MTP0 (knob 2). RESULT: **FAIL** — eval 86/100 ok, but SIGKILL 137 on 200k decodebench prefill (GPU OOM NV_ERR_NO_MEMORY; larger chunk spikes activation). MNT 16384 EXCLUDED-crash; MNT knob RESOLVED to 8192 (004: 4096 FAIL, 16384 crash, 8192 = interactive winner).
- **Reframe (from repo knowledge + README + 001-008):** GMU* 0.799609375 is knife-edge — only MNT 8192+MTP0 survives decode load; any memory-raising knob (MTP3, MNT16384) crashes. KV headroom is MOOT: at MAX_NUM_SEQS=8 the scheduler admits fewer requests than the cache holds, so Phase A KV gains don't lift serving. Quality optimum is LKG (GMU 0.75/MNT 8192/MTP0/EP false, 89/100); no knob lifts quality above 89. Remaining genuinely-untested knobs (seqs 16, mamba fp8) are capacity/latency levers, likely to crash or not lift quality at GMU*.
  MTP  = 0
  MNT  = 8192
  seqs = 8
  mamba-ssm = bfloat16
  EP   = false
Excluded-unsafe so far: EP=true@0.75 (006), GMU 0.80 (003), MNT 4096@0.75 (004)
Score note: eval scores across stable GMUs so far — 0.70: 83 (001), 0.75: 85/89 (002/005/008),
0.775: 86 (C1), 0.7875: 86 (C2), 0.79375: 85 (C3), 0.796875: 86 (C4), 0.7984375: 83 (C5), 0.79921875: 86 (C6), 0.799609375: 86 (C7). GMU alone has NOT lifted the score above the LKG 89 reference — KV headroom (+30% vs LKG) is the Phase A gain, not eval.
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
| C5 | C4 0.796875 PASS -> bisection midpoint of (0.796875, 0.80) -> test 0.7984375 | GMU 0.796875->0.7984375 | PASS | 3,876,094 | 14.79x | 83 (114/138) | 22.0 / 87.15s | 11/14 GiB (post-boot) | **PASS-stable / no score gain** — safe := 0.7984375; next interval (0.7984375, 0.80) |
| C6 | C5 0.7984375 PASS -> bisection midpoint of (0.7984375, 0.80) -> test 0.79921875 | GMU 0.7984375->0.79921875 | PASS | 3,854,880 | 14.71x | 86 (55P/8Pa/6F) | 24.0 / n/c (warm; caveat) | 9/14 GiB post-boot, 7/11 post-perf | **PASS-stable / no score gain; KV −21k vs C5 (no marginal gain)** — safe := 0.79921875; next interval (0.79921875, 0.80) |
| C7 | C6 0.79921875 PASS -> bisection midpoint of (0.79921875, 0.80) -> test 0.799609375 | GMU 0.79921875->0.799609375 | PASS | 3,876,094 | 14.79x | 86 (118/138) | 22.0-24.0 / 87.39s cold | ~8.7 GiB | **PASS-stable / no score gain; PHASE A closed — GMU*=0.799609375; next interval (0.799609375, 0.80)** |
| C8 | C7 PASS -> PHASE B knob 1: MTP 0->3 + MTP_DRAFT_VOCAB 47k at GMU* (spec A/B vs MTP0 best-so-far) | GMU* fixed; MTP 0->3, draft 47k | **FAIL** | 3,122,296 | 11.91x | 85 (55P/7Pa/7F) eval ok | n/a | ~10.9 GiB pre-crash | **FAIL — engine SIGKILL 137 mid-decodebench (GPU OOM NV_ERR_NO_MEMORY); MTP3+47k excluded-unsafe; MTP reverts to 0 best-so-far** |
| C9 | C8 FAIL -> revert MTP to 0; knob MNT 8192->16384 at GMU* | GMU* fixed; MNT 8192->16384, MTP0 | **FAIL** | 3,347,260 | 12.77x | 86 (55P/9Pa/5F) eval ok | 22.1-24.3 @1k then crash | ~13.0 GiB pre-crash | **FAIL — SIGKILL 137 on 200k prefill (GPU OOM NV_ERR_NO_MEMORY); MNT 16384 excluded-crash; MNT knob resolves to 8192** |
| C12 | knob 4: mamba-ssm bf16->fp8 at GMU* (shrink GDN state) | GMU* fixed; MAMBA bf16->fp8 | **FAIL (invalid)** | — | — | — | — | — | — | **FAIL — vllm serve rejects --mamba-ssm-cache-dtype fp8 (invalid choice; build v0.1.dev20073 supports auto/bf16/f16/f32 only). MAMBA knob RESOLVED to bf16 (LKG); fp8 excluded-invalid** |
| C10 | C9 FAIL (MNT 16384 crash) -> MNT knob probe down: 8192->4096 at GMU* (A/B vs C9) | GMU* fixed; MNT 8192->4096, MTP0 | PASS (boot 14min, attempt 3 after host-floor kill + orphan-rank reject) | 4,074,596 | 15.54x (highest of sweep) | 88 (57P/7Pa/5F, 121/138) | n/a — perf leg never ran | 7.8 GiB post-boot (tightest of sweep), drained to floor | **FAIL-survival — 6 GiB memwatch kill 12:00:59Z post-eval (host-RAM, not GPU); eval gate met 88>=86 but no decodebench survival; MNT 4096 excluded; MNT knob RESOLVED to 8192** |
| C11 | C10 FAIL -> MNT stays 8192; knob seqs 8->16 at GMU* (README: scheduler admits <=8 at 262K) | GMU* fixed; seqs 8->16, MTP0/MNT8192 | PASS (clean re-run boot ~12min; prior run: boot PASS, eval interrupted 49/69 no score, ~24h = cluster death not slowness) | 3,894,277 | 14.85x | 89 (58P/7Pa/4F, 123/138, sweep-best) | 21.6-24.0 @1k, then 200k leg n/a | ~7-8 GiB, drained to floor at 200k prefill | **FAIL-survival — 6 GiB memwatch kill 14:37:16Z during decodebench 200k prefill (host RAM); eval gate met 89>=86; seqs 16 excluded-survival; SEQs RESOLVED to 8 (LKG)** |
| C13 | C11 FAIL -> seqs stays 8; last discrete knob: EP false->true at GMU* (A/B; 006 already excluded EP at 0.75, run for grid completeness) | GMU* fixed; EP false->true, LKG base otherwise | PASS (boot ~11min, enable_expert_parallel=True, KV 3,816,998=14.56x) | 3,816,998 | 14.56x | 74 (artifact: 31/31 FAILs = 'connection failed' after death; pre-death passing ~TC-37) | n/a (server dead) | 6103 MiB at kill | **FAIL-survival — 6 GiB memwatch kill 15:25:12Z ~5min into eval (host-RAM leak, 006 signature); EP=true excluded-unsafe; EP RESOLVED to false (LKG)** |
| C14 | C13 FAIL -> all discrete knobs resolved (MTP0/MNT8192/seqs8/mamba-bf16/EP-false); composite = resolved knobs at GMU* = LKG base at GMU* (byte-identical settings to C7) | GMU* fixed; LKG base (only diff vs production LKG: GMU 0.75->0.799609375) | PASS (boot ~12.5min, KV 3,901,854=14.88x sweep-highest, PLE both, 0 Xid/OOM) | 3,901,854 | 14.88x | 31 artifact (server dead at TC-23; 21P/1Pa pre-death) | n/a (server dead) | ~7 GiB post-boot -> floor kill 16:19:13Z | **FAIL-survival - 6 GiB memwatch floor kill mid-eval (MemAvailable 6017 MiB, TC-23; C10/C11/C13 pattern confirmed: NO GMU* config survives full standard suite); composite does NOT beat LKG 89; convergence verdict: LKG = production winner, GMU* base = capacity-only** | | pending
| C15 | C14 FAIL-survival -> best safe config = LKG (GMU 0.75, 89/100, full-suite-surviving); C15 = fresh two-node re-deploy of LKG + full verification (production candidate confirmation) | = production LKG (profiles/production.env), fresh standard restart 16:22:20Z (stop+drop_caches both) | **PASS** (health 200, /v1/models 262144, real completion, PLE both nodes, 0 OOM/Xid/watchdog, no config departure) | 2,979,046 | 11.36x | **88/100 (121/138: 57P/7Pa/5F) — reproduces 89 within variance (fresh band 83-88)** | 21.9-23.9 tok/s @1k & @200k (flat, temp 0.0; 200k-prose TTFT 83.8s cold = first-prefill artifact, remaining 200k rows 0.68-2.23s) | 13-14 GiB (floor intact; survived full suite) | **PASS — LKG CONFIRMED production candidate; production.env KEPT at LKG; SWEEP COMPLETE (C1-C15), no config beat LKG on quality while surviving the suite** |

## Decision blocks (per run, appended newest-first)
### C14 (TASK-73.22) — EXECUTED 2026-09-25 (see docs/tune/022-c14-composite.md)

- **Previous result:** C13 EP-true FAIL-survival (6 GiB floor kill, host-RAM leak, 006 signature) -> EP RESOLVED false. All five discrete knobs resolved: MTP 0 / MNT 8192 / seqs 8 / mamba bf16 / EP false.
- **Rule applied:** C14 = composite of per-knob best-so-far at GMU* 0.799609375 (FROZEN) = LKG base at GMU* = byte-identical settings to C7.
- **Result:** boot PASS ~12.5 min (KV **3,901,854** fp8 tokens = 14.88x @262K — highest of the sweep, +25.8k vs C7 allocator variance; PLE both nodes; 0 Xid/OOM/watchdog at boot; orcus MemAvail ~7 GiB post-boot). Eval: server DIED at TC-23 when the 6 GiB host memwatch floor fired at **16:19:13Z (MemAvailable 6017 MiB, docker kill)** — bench reported 31/100 (43/138) as a connection-failure artifact (47 FAILs = "All connection attempts failed"), NOT a quality measurement. Perf leg never ran.
- **Verdict:** **FAIL-survival.** Pattern confirmed (C10 MNT4096 post-eval, C11 seqs16 at 200k prefill, C13 EP at TC~20, C14 LKG-base at TC-23): **no GMU* (0.799609375) config survives the full standard suite** — the 6 GiB host floor is reached within ~14-37 min of sustained 262K load at every tested knob set. C7 was the only GMU* run to finish a full suite (86/100). The composite does NOT beat LKG (89/100, full-suite-surviving). **Convergence verdict: LKG (GMU 0.75 / MTP0 / MNT 8192 / seqs 8 / mamba bf16 / EP false) is the production winner; GMU* base is capacity-only (KV 2.96M -> ~3.90M tokens, +32%, 14.88x) but not load-safe.**
- **Rollback:** LKG redeployed (stop + drop_caches both, .env restored to production LKG, relaunch), /health 200 re-verified.
- **Decision for C15:** production-candidate confirmation — best safe config = **LKG** (highest eval score among full-suite-surviving candidates: 89 vs GMU* base 86 and every other candidate's survival/quality record). C15 = fresh two-node standard restart of profiles/production.env + full verification; it must reproduce 89 within eval variance. production.env stays at LKG (documented decision: GMU* base rejected on survival, not quality).

### C14 (TASK-73.22) — derived 2026-09-25 from C13 result (to be executed now)

- **Previous result:** C13 EP-true at GMU* FAIL-survival (6 GiB host-memwatch floor kill ~5 min into eval, MemAvailable 6103 MiB; host-RAM leak = 006 signature). **EP RESOLVED to false (LKG).** Every discrete knob is now resolved: MTP 0 (C8 GPU-OOM crash) / MNT 8192 (C9 16384 crash, 004+C10 4096 excluded) / seqs 8 (C11 16 FAIL-survival at floor) / mamba bf16 (C12 fp8 invalid in build) / EP false (C13 + 006 leak).
- **Rule applied:** C14 composite = every knob at its measured best-so-far, simultaneously, at GMU* 0.799609375 (FROZEN). All resolved values equal the LKG base -> **composite = LKG base at GMU***, byte-identical settings to the C7 config. Not byte-identical to production LKG (GMU 0.75), so the full suite runs (no skip).
- **Expected outcome:** C10/C11 pattern — every GMU* config dies at the 6 GiB host floor under sustained 262K load; C7 itself was the only GMU* config to complete a full suite (eval 86, 200k cold 87.39s). If C14 FAILs: GMU* base is capacity-only and the production winner stays LKG (GMU 0.75, 89/100). If it survives: GMU* LKG base (eval ~86, KV 14.79x) becomes the C15 candidate vs LKG — LKG still wins on quality (89 > 86).
- **Config to test:** GMU 0.799609375 / MTP0 / MNT 8192 / seqs 8 / mamba bf16 / EP false / KV fp8 / PLE packed / 262144 ctx / 40 GiB container / 6 GiB memwatch floor. Profile: profiles/c14-composite.env. Record: docs/tune/022-c14-composite.md.
### C13 (TASK-73.21) — EXECUTED 2026-09-25 (see docs/tune/021-c13-ep-true.md)

- **Previous result:** C11 seqs16 FAIL-survival; all discrete knobs but EP resolved; 006 already showed EP-true leaks host RAM at 0.75.
- **Rule applied:** one knob per candidate — last grid cell: ENABLE_EXPERT_PARALLEL false->true at GMU* LKG base.
- **Result:** boot PASS ~11 min (enable_expert_parallel=True, EP workers, KV 3,816,998 = 14.56x); host-RAM leak began immediately; 6 GiB floor killed orcus vllm-fn at 15:25:12Z (~5 min into eval, MemAvailable 6103 MiB); eval finished 872.2s with 31 'connection failed' FAILs (74/138 artifact); decodebench never ran; 0 Xid / 0 GPU OOM.
- **Verdict:** FAIL-survival. EP=true excluded-unsafe at GMU* (and already at 0.75, 006). **EP RESOLVED to false (LKG).**
- **Decision for C14:** every discrete knob is now resolved: MTP 0 / MNT 8192 / seqs 8 / mamba bf16 / EP false. **C14 composite = LKG base at GMU*** (all-resolved values + GMU 0.799609375) — no knob change vs the C7 base; run the full suite for grid completeness. Expectation from the C10/C11 pattern: FAIL-survival at the 6 GiB floor under sustained 262K load; if it survives, GMU* LKG base becomes the capacity candidate for C15 vs the GMU 0.75 production LKG.
- **Decision for C15:** production-candidate confirmation — fresh two-node launch of whichever config wins (expected: GMU 0.75 LKG, already AC#5-verified; GMU* base only if C14 survives its full suite).

### C11 (TASK-73.19) — EXECUTED 2026-09-25 clean re-run (see docs/tune/019-c11-seqs16.md)

- **Previous state:** interrupted run had boot PASS (KV 3,910,946) but eval aborted 49/69 with no score after a ~24h wall (cluster died mid-run); ticket reopened for a clean re-run.
- **Rule applied:** one knob per candidate — MAX_NUM_SEQS 8->16 at GMU* (MTP0/MNT8192/mamba-bf16/EP-false).
- **Result:** clean boot PASS ~12 min (KV 3,894,277 = 14.85x); eval **89/100** (123/138, 58P/7Pa/4F, wall 2226.6s ~37 min — the interrupted run's ~24h was cluster death, not 16-way slowness); decodebench 1k leg 21.6-24.0 tok/s (no 1k regression) then the 200k prefill hit the 6 GiB host floor at 14:37:16Z (MemAvailable 5995 MiB) — kill, warda orphaned.
- **Verdict:** FAIL-survival (keep-rule: eval >= 86 AND survives decodebench). **MAX_NUM_SEQS RESOLVED to 8 (LKG); seqs 16 excluded-survival.**
- **Cross-candidate pattern (feeds C14 composite):** every GMU* (0.799609375) config that raises a memory-pressure knob beyond the LKG base dies at the 6 GiB floor under sustained 262K load (C10 MNT4096 post-eval, C11 seqs16 at 200k prefill). The only GMU* config with a full-suite survival record so far is the LKG base itself (MTP0/MNT8192/seqs8/mamba-bf16/EP-false).
- **Decision for C13/C14:** seqs -> 8. Remaining discrete knob: ENABLE_EXPERT_PARALLEL true at GMU* (C13, already excluded at 0.75 by 006 with a mem leak — expected FAIL, run for completeness of the grid). C14 composite = LKG base at GMU* (all knobs at resolved values) + GMU 0.799609375; C15 = fresh re-run of the best safe config (GMU 0.75 LKG vs GMU* LKG-base — whichever has the better survival+eval record).

### C10 (TASK-73.18) — EXECUTED 2026-09-25 (see docs/tune/020-c10-mnt4096.md)

- **Previous result:** C9 MNT 16384 FAIL (GPU OOM SIGKILL on 200k prefill; MNT knob unresolved among 4096/8192/16384).
- **Rule applied:** MNT A/B at GMU* — 16384 excluded-crash; test the other extreme 4096 (004 already FAILed 4096 at 0.75 with eval 83; at GMU* 4096 tests whether a lower batch budget buys decode stability or just loses eval).
- **Result:** boot PASS (attempt 3; attempt 1 host-memwatch floor kill, attempt 2 REQUIRE_IDLE_GPU orphan rank); KV 4,074,596 (15.54x, highest of sweep); eval **88/100** (121/138, 57P/7Pa/5F, 37 min, 0 preemptions); then host RAM drained 116->6 GiB and the 6 GiB floor fired at 12:00:59Z, decodebench never ran.
- **Verdict:** FAIL-survival. Keep-rule (eval >= 86 AND survives decodebench) not met. MNT 4096 EXCLUDED. **MNT knob RESOLVED to 8192** (4096: 004 eval FAIL + C10 survival FAIL; 16384: crash; 8192: PASS everywhere, LKG).
- **Decision for C11:** MNT fixed at 8192; next knob = MAX_NUM_SEQS 8->16 at GMU* (the README concurrency lever; untested). RISK: 16-way GDN recurrent state at knife-edge GMU* + tightest host margins; near-certain pressure. Keep seqs 16 iff eval >= 86 AND survives decodebench, else revert to 8.


### C5 (TASK-73.13) — derived 2026-09-23 from C4 result (EXECUTED — see docs/tune/013-c5-gmu0.7984375.md)

- **Previous result:** C4 GMU 0.796875 PASS (stable boot, 0 OOM/Xid/watchdog, MemAvail 12.0/12 GiB post-boot; KV 3,862,456 = 14.73x; eval 86/100).
- **Rule applied:** PHASE A bisection — 0.796875 PASS -> safe := 0.796875; test midpoint of (0.796875, 0.80): (0.796875 + 0.80) / 2 = **0.7984375** (7-decimal).
- **New search state (pre-test):** interval (0.796875, 0.80), candidate GMU 0.7984375. Update on result: PASS -> safe := 0.7984375, next interval (0.7984375, 0.80); FAIL -> unsafe := 0.7984375, next interval (0.796875, 0.7984375).
- **Config to test:** GMU 0.7984375; everything else = LKG (MTP0 / MNT 8192 / seqs 8 / mamba bf16 / EP false / KV fp8 / PLE packed / 262144 ctx / 40 GiB container / 6 GiB memwatch floor). Profile: `profiles/c5-gmu0.79844.env`.
- **Risk note:** 0.7984375 is only 0.0015625 below the known-unsafe 0.80 and MemAvail has eroded to ~12 GiB (C1 16.96 / C2 13.17 / C3 12.30 / C4 12.0) — a memwatch kill at warmup is now the most likely outcome, as at 0.80 (003).
- **RESULT:** PASS — boot ~12 min, 0 OOM/Xid/watchdog (boot + eval + perf), KV 3,876,094 (14.79x), eval 83/100 (114/138), perf 22.0 tok/s / 87.15s cold TTFT, MemAvail 11/14 GiB post-boot (floor intact). safe := 0.7984375; GMU* = 0.7984375; next interval (0.7984375, 0.80).

### C6 (TASK-73.14) — derived 2026-09-23 from C5 result (EXECUTED — see docs/tune/014-c6-gmu0.79922.md)

- **Previous result:** C5 GMU 0.7984375 PASS (stable boot, 0 OOM/Xid/watchdog, MemAvail 11/14 GiB post-boot; KV 3,876,094 = 14.79x; eval 83/100).
- **Rule applied:** PHASE A bisection — 0.7984375 PASS -> safe := 0.7984375; test midpoint of (0.7984375, 0.80): (0.7984375 + 0.80) / 2 = **0.79921875** (8-decimal).
- **New search state (pre-test):** interval (0.7984375, 0.80), candidate GMU 0.79921875. Update on result: PASS -> safe := 0.79921875, next interval (0.79921875, 0.80); FAIL -> unsafe := 0.79921875, next interval (0.7984375, 0.79921875).
- **Config to test:** GMU 0.79921875; everything else = LKG (MTP0 / MNT 8192 / seqs 8 / mamba bf16 / EP false / KV fp8 / PLE packed / 262144 ctx / 40 GiB container / 6 GiB memwatch floor). Profile: `profiles/c6-gmu0.79921875.env`.
- **Risk note:** 0.79921875 is only 0.00078125 below the known-unsafe 0.80 and MemAvail has eroded to ~10-11 GiB (C5 11) — a memwatch kill at warmup is now the near-certain outcome, as at 0.80 (003). Expected KV gain over C5: ~7-9k tokens if it boots.
- **RESULT:** PASS — boot ~12.4 min, 0 OOM/Xid/watchdog/memwatch (boot + eval + perf), KV 3,854,880 (14.71x; −21,214 vs C5 — no marginal KV gain), eval 86/100 (55P/8Pa/6F), decode ~24 tok/s warm (200k rows warm — cold TTFT n/c, see record), MemAvail 9/14 GiB post-boot, 7/11 post-perf (floor intact). safe := 0.79921875; GMU* = 0.79921875; next interval (0.79921875, 0.80).

### C7 (TASK-73.15) — derived 2026-09-23 from C6 result (to be executed by next run)

- **Previous result:** C6 GMU 0.79921875 PASS (stable boot, 0 OOM/Xid/watchdog, MemAvail 9/14 GiB post-boot; KV 3,854,880 = 14.71x (−21k vs C5); eval 86/100).
- **Rule applied:** PHASE A bisection — 0.79921875 PASS -> safe := 0.79921875; test midpoint of (0.79921875, 0.80): (0.79921875 + 0.80) / 2 = **0.799609375** (9-decimal, final bisection step).
- **New search state (pre-test):** interval (0.79921875, 0.80), candidate GMU 0.799609375. Update on result: PASS -> safe := 0.799609375, next interval (0.799609375, 0.80); FAIL -> unsafe := 0.799609375, next interval (0.79921875, 0.799609375). After C7: GMU* = max safe GMU tested, FROZEN for Phase B.
- **Config to test:** GMU 0.799609375; everything else = LKG (MTP0 / MNT 8192 / seqs 8 / mamba bf16 / EP false / KV fp8 / PLE packed / 262144 ctx / 40 GiB container / 6 GiB memwatch floor). Profile: `profiles/c7-gmu0.799609375.env`.
- **Risk note:** 0.799609375 is 0.000390625 below the known-unsafe 0.80 and MemAvail post-boot is ~9 GiB (post-perf 7) — a memwatch kill at warmup is the most likely outcome, as at 0.80 (003). Expected KV delta ~0 (allocator granularity). This is the last PHASE A step: whichever outcome, GMU* is set and Phase B (C8-C13) starts at GMU*.

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
