# I6.7F — one bounded South Building cached fusion recovery

This is a **single, explicit recovery attempt** on the user's existing, interrupted
South Building run. The 87-view SfM solution, depth and normal maps, original failed
worker records, and previously failed two-thread recovery remain untouched. The
original production worker is NOT modified. This script calls ONLY geometric
`pycolmap.stereo_fusion` and writes separate outputs under a NEW recovery UUID.

## Why these settings

The previous no-cache two-thread fusion attempt reached 8.04 GiB RSS during
workspace loading, triggering the resource guard. Switching from unlimited to
two threads did not fit in the 15 GiB system. This one experiment uses
`use_cache=True`, a 1 GiB workspace cache, and one requested fusion thread.
The cached-workspace implementation may serialize work, so more threads are
not assumed to improve throughput in this mode. It retains full image
resolution (`max_image_size=-1`), 87 views, the original geometric fusion
thresholds, frame identity, and output visibility checks. The execution
strategy changes; the scientific settings do not.

The installed PyCOLMAP runtime in the pinned original request advertises
`use_cache`, `cache_size`, and `num_threads`. The real workstation's actual
option bindings and workspace behavior must still pass the explicit dry run
and the first live log. Cached execution MAY reduce peak memory, but is not
proved sufficient. The unchanged watchdog stops at worker RSS >= 8 GiB,
MemAvailable < 3 GiB, free disk < 10 GiB, or 240 minutes. It polls once per
second and cannot prevent every sudden OOM spike. Close VS Code and other
RAM-heavy processes; run the detached supervisor from a normal terminal.

The script permits only one cached-recovery launch, even if that attempt
fails. After a second memory guard or a runtime incompatibility, STOP:
move this exact workspace to a higher-RAM host (preferably >=32 GiB,
with 64 GiB as the more comfortable target), or get explicit approval for
lower-resolution/fewer-view scientific output. Do not silently reduce 87
views or max image size or retry with higher memory guards.

Success yields an independently counted PLY and `.vis` in a new recovery
folder; it is **not a final dense admission**. The pre-existing production
admission schema is not fabricated or bypassed. The full GPU viewer is gated
on subsequent independent admission.

The original stereo maps were not hashed per-file by the interrupted run.
The preflight checks the exact expected map filenames, headers, dimensions,
byte lengths, and original authoritative SfM/image checksums, but does not
claim to prove every map's scientific validity.
