# I6.7F South Building: admitted sparse JSON input-bound correction

## Evidence and motivation

South Building, session `03d5f129-781f-419f-90aa-95673a5306fc`, produced an accepted PyCOLMAP sparse result with 87 registered views. Its unmodified `sfm_result.json` is 94,421,034 bytes (SHA256 `65f6a2b75a89cf6442e8445cdb3df4c4d0ec74269de04aba2a95a42e1e31766d`). The isolated post-sparse worker previously used one 64 MiB JSON ceiling for both compact `offline_result.json` and potentially much larger authoritative SfM evidence, causing rejection before MVS and before a dense run folder was created.

## Narrow correction

The authoritative `SfM result` alone now has a 128 MiB bounded input ceiling; `offline result` and other labels retain 64 MiB. No input is silently truncated. The worker still reads and validates the full typed SfM, binds it to the accepted session/summary, preserves arbitrary gauge and explicit-only provider selection, and independently qualifies the final `fused.ply`. No existing session or admission evidence is rewritten.

The active I6.7F test runner's exact-source hash is updated for this **explicit corrective exception** to the earlier worker-source freeze; all other pinned frozen source hashes are unchanged. The older I6.7E production runner is a historical frozen-state gate and is not the current I6.7F acceptance command.

## Local tests

Synthetic file-stat fixtures exercise 64 MiB + 1, 94,421,034 bytes, and 128 MiB SfM acceptance, rejection beyond 128 MiB without reading JSON, and unchanged 64 MiB metadata behavior. The actual 94 MiB South Building file is **not available in the package**: on the user's workstation, first run a read-only admitted-sparse load against the preserved session and confirm the original SHA256 before starting a new dense attempt. This fixes only the observed input-size gate, not a promise that COLMAP-MVS with 87 views will finish on 6 GiB VRAM.

A GUI dense failure is terminal for its previous sparse session by design. Do not reuse that failed dense run ID or treat its absence of a provider folder as success. For a new Qt-supervised dense attempt, create and admit a new One Start sparse session in the GUI after validation. Do not re-run the old session automatically. Preserve the previous sparse result and any failure logs as negative evidence.
