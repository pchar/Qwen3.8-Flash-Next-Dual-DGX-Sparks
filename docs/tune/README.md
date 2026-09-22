# m-8 tune loop — sweep matrix and protocol (TASK-73)

Owner decisions (2026-09-22): any `.env` knob may be tuned, **one per ticket**,
@nxt discretion. Success gate: healthy boot + no crash/OOM/Xid + tool-eval-bench
score >= current leaderboard best at equal-or-higher config. On boot failure:
max 2 fix-retries, then roll back to last known-good and record the failure.

## Standard protocol (every ticket, no exceptions)

1. `docker stop vllm-fn` on **orcus and warda**
2. `sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches` on **both**
3. change ONE param in `.env` (orcus repo) + update tracked `profiles/<candidate>.env`
   + result record → **commit → push origin** (fork is source of truth)
4. launch both nodes (head: `./start-fp8.sh`; worker per launch flow)
5. watch startup: poll `/health` on orcus:8888 (~30 min max), PLE registered on
   both nodes, no OOM / Xid / watchdog / orphan rank
6. if healthy: standard eval on jupiter —
   `~/.local/bin/tool-eval-bench --base-url http://orcus.lan:8888/v1` (full
   defaults, model auto-detected); then `~/.local/bin/tool-eval-bench --leaderboard`
7. record verdict + evidence on the backlog ticket (SHA, run ids, scores, memory)

Result records: `docs/tune/NNN-<candidate>.md` (NNN = tune sequence number).
Profiles: `profiles/<candidate>.env` (no secrets — HF_TOKEN stays in `.env`).

## Sweep plan

| # | knob | values | notes |
|---|------|--------|-------|
| 0 | (none — baseline) | GMU 0.70 as validated | TASK-73.01, seeds leaderboard |
| 1 | GPU_MEMORY_UTILIZATION | 0.75 → **0.80** (stop) | DONE — 0.75 PASS (85/100), 0.80 FAIL (watchdog kill); sweep stopped at first unsafe, 0.75 = last known-good |
| 2 | MAX_NUM_BATCHED_TOKENS | 4096 vs 8192 | 4096 FAIL (eval 83<85); **8192 = winner**, GMU sweep done — step 2 DONE |
| 3 | MTP_NUM_SPECULATIVE_TOKENS | 0 vs 3 | **DONE — MTP0 PASS (89/100 new best, KV 2.96M +28.5%); MTP3 wins decode tok/s (+25-130%) but costs ~4 eval pts; MTP0 = last-known-good** |
| 4 | ENABLE_EXPERT_PARALLEL | true | ONLY if all above stable |
| + | (discretion) | any other .env knob | one per ticket, @nxt discretion |

## Status

| # | candidate | commit | boot | eval score | verdict | ticket |
|---|-----------|--------|------|-----------|---------|--------|
| 0 | baseline gmu0.70 | 86c9a70 (+result) | PASS | **83/100** (53P/8Pa/8F) | BASELINE — pass, ref established | TASK-73.01 |
| 2 | gmu0.75 | cc74c2a (+record) | PASS | **85/100** (54P/9Pa/6F) | PASS — new best safe GMU, KV +49% | TASK-73.02 |
| 3 | gmu0.80 | e7a7ec1 (+record) | **FAIL** (watchdog kill @ KV sizing) | n/a (never served) | **FAIL — first unsafe GMU, sweep STOPPED, rolled back to 0.75** | TASK-73.03 |
| 4 | mnt4096 | 781c02a (+record) | PASS | **83/100** (54P/6Pa/9F) | **FAIL — eval 83 < 85 gate, 3 scen partial->fail; 8192 = batch-budget winner, rolled back to 0.75/8192** | TASK-73.04 |
| 5 | mtp0 | 104657f (+record) | PASS | **89/100** (58P/7Pa/4F) | **PASS — new best eval, KV +28.5% (2.96M), MTP0 = last-known-good; MTP3 keeps decode speed advantage** | TASK-73.05 |
