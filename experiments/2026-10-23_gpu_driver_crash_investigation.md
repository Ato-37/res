# PC crash investigation: NVIDIA driver bugchecks during GPU experiments (2026-10-23)

Read-only investigation from WSL (Windows event logs via PowerShell). No dump file could be read (see below). Hardware: RTX 5070 Ti, Gigabyte B850M DS3H (BIOS F10), 2x16 GB DDR5-4800, NVIDIA driver 32.0.16.1742 (2026-09-30), WSL2 GPU compute.

## Verified facts
- Bugchecks with dumps in the last 30 days: 2026-10-09 21:32 (0x133 DPC_WATCHDOG_VIOLATION), 2026-10-23 03:02 (0xD1 DRIVER_IRQL_NOT_LESS_OR_EQUAL) and 2026-10-23 14:42 (0xD1). The two 0xD1 reports name `nvlddmkm.sys` as the possibly related driver (event 1019) and have the same parameter pattern (arg2 = 2, arg3 = 8, arg4 = arg1 = faulting address, i.e. an instruction fetch at an invalid kernel address at DISPATCH_LEVEL). No WHEA hardware-error events, no Display TDR (4101) events.
- Earlier "Kernel-Power 41 without dump" events (10-09, 10-17, 10-19, 10-20) were manual restarts after a loose GPU power cable made the fan run away (user statement), so they are not crashes.
- nvlddmkm event 153 ("Error occurred on GPUID: 100", \Device\Video3) occurred 64 times in 30 days, always inside GPU-experiment periods: 3 on 09-29, a burst of ~60 at 10-09 21:30:01 (together with 1,412 event-13 and 37 event-14 records, two minutes before the 0x133 bugcheck), and on 10-23 at 01:19:44, 13:51:00 and 14:38:37 (four minutes before the 14:42 bugcheck). Event 153 therefore does not always precede a crash (01:19 and 13:51 were followed by normal run completion), but it precedes the observed crashes.
- Both 0xD1 crashes happened while an annotation-key-bias run (smoke/full) was in progress: 03:02 about 7.5 minutes after the ln4 smoke finished (decode/model-load/forward phase of the full run) and 14:42 about 27 minutes after the ln256 smoke finished (decode finished, GPU forward phase). Dozens of other GPU runs of 20-85 minutes finished without a crash.
- Dumps: `C:\Windows\Minidump\*.dmp` have mode `----------` for the WSL user (permission denied); no access was forced.

## Correction of an earlier statement
An earlier report said the identical low 16 bits (`d080`) of the faulting address in both 0xD1 crashes showed a reproducible driver-function offset. That is too strong: kernel module bases can differ by multiples of 4 KiB, so the low 12 bits matching is expected for the same code offset, but bits 12-15 match by chance with probability about 1/16, and the address was not shown to lie inside `nvlddmkm`. Without the dump this is only suggestive.

## Interpretation (hypotheses, not established)
1. NVIDIA driver / WSL GPU-PV software fault under sustained compute load (supported by: bugchecks name nvlddmkm, GPU error events 153/13/14 shortly before, third-party reports of events 153/13/14 on an RTX 5070 Ti that stopped after rolling back the driver; the reports are anecdotal and not for this driver version).
2. GPU/board stability under sustained load (power delivery, thermals, VRAM). A cable re-seat was done and judged unlikely by the user; no thermal or power data existed before.
3. Less likely: memory or PSU faults (no WHEA records; crashes tied to GPU work).

## Actions and next steps
- Started `scripts/log_gpu_telemetry.sh` (temperature, power, clocks, utilization, PCIe link, throttle reasons every ~2 s into `data/gpu_telemetry/gpu_YYYYMMDD.csv`, untracked) so the seconds before the next crash survive a reboot.
- New long runs (`scripts/run_pathway_knockout.py`) append one record per video to a partial JSONL and can resume, so a crash costs at most one video.
- Needs the user: install a different NVIDIA driver version with a clean removal (DDU) and compare; optionally lower the GPU power limit; copy the minidumps to a readable folder if a stack analysis is wanted.

## Dump analysis (2026-10-23, after the user copied the minidumps; header-level parsing only, no symbol-based stack analysis)
Four kernel minidumps (Windows build 26100, 8 CPUs; PAGEDU64 triage dumps):
| dump | bugcheck | parameters |
|---|---|---|
| 2026-08-23 22:24 | **0x116 VIDEO_TDR_ERROR** | arg3 = 0xC000009A (STATUS_INSUFFICIENT_RESOURCES), arg4 = 4 |
| 2026-10-09 21:32 | 0x133 DPC_WATCHDOG_VIOLATION | (1, 0x1e00, ...) |
| 2026-10-23 03:02 | 0xD1 | arg1 = arg4 = 0xfffff8018b50d080, arg2 = 2, arg3 = 8 (execute) |
| 2026-10-23 14:42 | 0xD1 | arg1 = arg4 = 0xfffff801cd8cd080, arg2 = 2, arg3 = 8 (execute) |
- `nvlddmkm.sys` is in the loaded-driver list of all four dumps. The 0xD1 faulting address lies below the kernel image and could not be attributed to a module (the module-base table was not parsed); the low-16-bit coincidence (`d080`) stays inconclusive.
- **The first NVIDIA-driver bugcheck is from 2026-08-23, before the October experiments**, and a different driver generation: NVIDIA component re-configuration events (nvvad/nvhda) appear on 07-07, 08-12 and 09-14 and the installed display driver is 32.0.16.1742 (2026-09-30, installed after 09-14), so the instability occurred under at least two driver versions and on workloads other than these experiments (the experiments only started in September).
- Kind of failures seen: TDR with insufficient-resources status (Aug), DPC watchdog after a burst of GPU exceptions (Oct 9), instruction fetch at an invalid address at DISPATCH_LEVEL (Oct 23, twice). They are different bugcheck types, which fits a failing/unstable GPU or platform (power, thermals, VRAM, PCIe) at least as well as a single driver bug.
- Not established: whether these are a driver bug or a hardware fault; that needs a different-driver trial or a hardware test (e.g. a VRAM/GPU stress test with the telemetry logger running, power-limit trial, another PSU rail or cable).
