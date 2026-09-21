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
| 1 | GPU_MEMORY_UTILIZATION | 0.75 → 0.80 → 0.835 | stop sweep at first unsafe; +0.01 ≈ +95k fp8 KV tokens |
| 2 | MAX_NUM_BATCHED_TOKENS | 4096 vs 8192 | at best safe GMU; interactive vs long-context winner |
| 3 | MTP_NUM_SPECULATIVE_TOKENS | 0 vs 3 | acceptance rate, decode tok/s, KV impact |
| 4 | ENABLE_EXPERT_PARALLEL | true | ONLY if all above stable |
| + | (discretion) | any other .env knob | one per ticket, @nxt discretion |

## Status

| # | candidate | commit | boot | eval score | verdict | ticket |
|---|-----------|--------|------|-----------|---------|--------|
| 0 | baseline gmu0.70 | _pending_ | _running_ | _pending_ | _pending_ | TASK-73.01 |
