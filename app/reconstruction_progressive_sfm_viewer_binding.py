"""I6.6D narrow binding between validated controller snapshots and the page viewer.

The I6.6C transport/controller stays independent from Qt viewport code.  This
module owns only the one-way presentation binding introduced by I6.6D.
"""
from __future__ import annotations


def bind_progressive_sfm_viewer(page, controller) -> None:
    """Connect passive provisional SfM state to the page presentation sink."""

    controller.progressive_sparse_snapshot_changed.connect(
        page.set_progressive_sparse_snapshot
    )
    snapshot = controller.progressive_sparse_snapshot
    if snapshot is not None:
        page.set_progressive_sparse_snapshot(snapshot)


__all__ = ["bind_progressive_sfm_viewer"]
