# Production promotion and stability verification — TASK-73.08

## Provenance and selected configuration

- Fork branch: `m8-tune`; selected profile commit: `64b6ef8996169ab6c376233ad525e1ee19a13a58`.
- `profiles/production.env` equals tune-5 `profiles/mtp0.env`: GMU 0.75,
  MTP0, empty draft vocabulary, MNT 8192, EP false, packed PLE, FP8 KV,
  262144 context, 8 sequences, container cap 40 GiB, watchdog floor 6 GiB.
- Image ID: `sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a`.
- Image repository digest: `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`.
- Model revision: `236dfdf285828023ca3bcd3f37366c58a3469b13`.
- Launch command on orcus: `./start-fp8.sh --no-download`.
- Standard restart: stop `vllm-fn` on both nodes, sync and drop caches on
  both nodes, then launch from the tracked revision using the ignored `.env`.
- Full-default quality command on jupiter:
  `~/.local/bin/tool-eval-bench --base-url http://orcus.lan:8888/v1`.
- Sequential performance command on orcus:
  `python3 bench/decodebench.py --decode 400 --contexts 1000,200000 --temps 0.0 --model qwen3.8-flash-next-fp8`.

## Failed initial verification

Initial head launch: 2026-09-22 06:20:17 UTC. Health and PLE initially worked,
with 3,000,260 KV tokens. The host watchdog killed the head at 06:48:35 UTC:
`MemAvailable=6130 MiB < floor`. Exit 137, Docker OOMKilled=false.
Both nodes also logged NVIDIA `NV_ERR_NO_MEMORY` allocation failures.

Jupiter report `~/runs/2026/09/2026-09-22T06-35-13Z_f2e604.md`:
**45/100 (62/138)**; connection broke during TC-32 and subsequent scenarios
failed to connect. This is a failed run, not the prior tune-5 score of 89.
Evidence retained on orcus under `logs/task-73.08-initial/`.

## Retry 1 and stability diagnosis

Clean head start: 06:53:54 UTC; health 200 verified at 07:06 UTC.
Both nodes completed PLE 131/131 and TP2 registration; KV 2,977,531 tokens
(11.36x). Orcus kernel logged 16 NVIDIA allocation-failure lines during
warm-up at 07:05:15–18 UTC. Worker kernel had zero matching lines.
The full-default eval began on jupiter, but was deliberately interrupted
before completion to apply the identified host fix and rerun cleanly.
It is not a scored verification run. Log: `/tmp/task-73.08-retry1-eval.log`.
Other retry evidence: orcus `logs/task-73.08-retry1/`.

Orcus reserved **8 GiB of unused static HugeTLB memory**:
4096 total and free 2 MiB pages, zero reserved pages, empty `/dev/hugepages`.
Warda reserved zero. `/etc/sysctl.d/99-vllm.conf` persisted this setting.
Releasing it with `sudo sysctl -w vm.nr_hugepages=0` immediately raised
MemAvailable from roughly 10 GiB to 18,361,040 KiB. This explains the
baseline head/worker memory gap; fresh boot and load tests must establish
whether it also resolves the allocation failures and watchdog shutdown.

## Host prerequisite and rollback

Tracked host setting: `profiles/production-host-sysctl.conf`.
On orcus, changed only `vm.nr_hugepages` from 4096 to 0 in
`/etc/sysctl.d/99-vllm.conf`, retaining `vm.swappiness=10`.
Backup: `/etc/sysctl.d/99-vllm.conf.task-73.08-backup`.
Warda already has zero static hugepages. No model knob or safety limit changed.

For another host, inspect `/proc/meminfo` and current workload ownership
before applying this setting: do not reclaim HugeTLB pages in use.
For rollback, stop both model containers first, restore the backed-up sysctl
file and apply it with `sudo sysctl -p /etc/sysctl.d/99-vllm.conf`.
The original model profile remains available at `64b6ef8`; restoring the
8 GiB reservation also restores the observed memory-pressure risk.
Do not lower the 6 GiB watchdog floor to obtain a passing run.

## Verification after the host fix

Pending fresh two-node boot, full-default quality eval and 1k/200k decodebench.
The production promotion is not yet verified complete.
