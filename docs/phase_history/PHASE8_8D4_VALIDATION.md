# Phase 8.8D4 — Offline Sparse Spatial Coverage

## Scope

Phase 8.8D4 adds presentation-only spatial inspection for a completed offline sparse-SfM result. It is a bridge from generic offline reconstruction inspection toward later object workflows, without introducing object measurement, dense geometry, reference comparison, or new reconstruction algorithms.

D4 derives its presentation exclusively from the bounded D1 `offline_visualization.json` artifact already loaded by the application. PyCOLMAP remains external and unchanged.

## Presentation contract

The Reconstruction page adds **SPARSE SPATIAL COVERAGE** to the existing Offline Inspection group:

- `Envelope` — robust X/Y/Z axis-aligned sparse-sample extents in reconstruction **gauge** units.
- `Camera span` — axis-aligned diagonal span of registered camera centers in gauge units.
- `Standoff` — min / median / max distance from camera centers to the robust sparse-sample center, in gauge units.
- `Basis` — sampled sparse-point count versus the authoritative full SfM landmark count.

The viewport adds a separately toggleable, subtle wireframe presentation envelope.

## Robust envelope

The envelope uses per-axis 2% and 98% quantiles of the bounded sparse landmark sample. This intentionally reduces visual sensitivity to isolated SfM outliers while remaining deterministic and inexpensive on the UI thread.

The envelope is **not**:

- object dimensions;
- a semantic object bounding box;
- metres or another metric unit;
- canonical geometry;
- certified metrology;
- a quality/acceptance gate.

It is only a presentation-space coverage aid in the arbitrary reconstruction gauge.

## Authority and isolation

D4 does not modify:

- `core/reconstruction`;
- the PyCOLMAP worker or its environment;
- Stella or the live protocol;
- the D3 authoritative sparse-SfM health assessment;
- canonical geometry authority;
- metric scale authority;
- measurement or reference-comparison logic.

A new offline or live session clears the previous spatial envelope and readouts so stale reconstruction coverage cannot leak into another session.


## Sidebar overflow hardening

The D4 screenshot gate exposed a real layout issue: adding D2 camera inspection, D3 SfM health, and D4 spatial coverage to the fixed-height right sidebar could make Qt compress the Offline Inspection rows until labels overlapped. The correction keeps the existing panel width and fixed title, but places the settings body in a vertical `QScrollArea` with horizontal scrolling disabled. This preserves normal row geometry on 768p/900p/1080p desktops and gives later 8.8E inspection controls safe vertical growth without stealing more reconstruction viewport width.

The scroll container is presentation-only and does not alter reconstruction state, controller bindings, backend execution, or authority semantics.

## Automated validation

Run:

```bash
./scripts/run_phase8_8d4_tests.sh quick
./scripts/run_phase8_8d4_tests.sh protocol
./scripts/run_phase8_8d4_tests.sh backward
```

The focused tests cover:

- robust envelope behavior in the presence of an extreme sparse outlier;
- gauge-only camera span and standoff presentation;
- unavailable state for insufficient sparse samples;
- strict presentation-only source contract;
- UI readouts and safety wording;
- independent envelope show/hide control;
- reset on a new offline/live session;
- D1/D2/D3 backward compatibility.

## No-backend visual preview

```bash
python scripts/run_phase8_8d4_spatial_coverage_preview.py
```

Check that:

- the D1 sparse cloud remains visible;
- D2 registered-camera inspection remains usable;
- D3 Sparse SfM Health remains visible;
- a restrained wireframe sparse envelope frames the main sample;
- the envelope toolbar button hides/shows only that presentation layer;
- all spatial readouts explicitly say `gauge` or `presentation envelope only`;
- no text presents the envelope as object dimensions or metric accuracy.

## Real South Building validation

```bash
rm -rf /tmp/phase8_8d4_offline_spatial_coverage

python scripts/run_phase8_8d4_offline_spatial_coverage_smoke.py \
  --images /home/eiatech/Desktop/phase8_3c_south_building/south-building/images \
  --sfm-python ~/anaconda3/envs/eia-pycolmap-probe/bin/python
```

Required properties include:

- real external PyCOLMAP completion;
- D3 authoritative quality assessment remains usable;
- D4 robust spatial envelope is available and visible;
- positive envelope diagonal and camera span;
- gauge-only standoff readouts;
- bounded sparse sample count remains separate from authoritative full landmark count;
- selected offline camera remains inspectable;
- no live pose/trajectory is mixed into the offline result;
- scene remains `partial`;
- scale remains `arbitrary`;
- canonical geometry remains non-authoritative;
- certified metrology remains false;
- production process does not import PyCOLMAP.

## Manual sanity

Run `python main.py` and check only behavior affected by D4 plus a small backward sanity check. Entering Reconstruction must not autostart a backend. Existing live C1–C5 and offline D1–D3 presentation must remain intact.


### Pre-commit sidebar gutter correction

The D4 scrollable settings body reserves a small right-side gutter so Linux/Qt
styles that visually overlay the vertical scrollbar cannot clip right-aligned
settings values. This is presentation-only and does not change reconstruction,
quality, spatial-coverage, authority, or backend behavior.

### Pre-commit horizontal minimum-width correction

The populated D2/D3/D4 inspection stack exposed a second Qt layout edge case:
the four camera-navigation `QPushButton`s could keep a minimum horizontal size
large enough to make the scroll body wider than its viewport. With horizontal
scrolling intentionally disabled, this appeared as right-edge clipping across
unrelated settings rows. The camera-navigation buttons are now horizontally
shrinkable while preserving their equal-row layout. The D4 UI regression test
asserts that the scroll body never exceeds the viewport width at a 1366x768-class
window. This remains presentation-only.

### Final responsive-sidebar correction

The initial D4 scroll-area correction exposed a Qt sizing detail: `QSizePolicy.Expanding`
still honors widget `minimumSizeHint()` values. Long camera identifiers, the camera combo,
and the four D2 navigation buttons could therefore keep the scroll body wider than a
constrained settings viewport even with horizontal scrolling disabled. D4 now marks the
camera combo, navigation buttons, and settings value labels horizontally `Ignored`, so
they may shrink/wrap without imposing an artificial horizontal minimum. The regression
test validates these size-policy contracts rather than comparing transient offscreen
QSplitter geometry. Vertical scrolling remains the only allowed settings scrollbar.

### Pre-commit responsive-row allocation correction

The settings values remain horizontally shrinkable, but the status-row layout now gives the value widget the row's remaining width directly instead of placing an expanding spacer before it. This prevents shrinkable, word-wrapped labels from collapsing to nearly zero width and requesting excessive height (large blank vertical gaps) on the real Linux/Qt preview. The D4 UI regression gate checks representative settings values retain usable width and bounded row height.


### Pre-commit UI test correction — offscreen geometry

The responsive-sidebar regression test must validate layout contracts rather than
absolute QLabel heights. Qt's offscreen platform can transiently compress a
QSplitter child far below the Reconstruction Settings panel's designed minimum
width; word-wrapped labels then legitimately report very large heights. The test
therefore verifies that status rows contain exactly label + stretch-bearing value
(no expanding spacer ahead of the value), while the real interactive preview
remains the authoritative visual check for normal row spacing.

### Pre-commit UI test correction — nested group traversal

The responsive-sidebar regression helper now descends through both nested
`QLayoutItem.layout()` objects and widget-owned layouts such as `QGroupBox.layout()`.
The previous helper stopped at group-box widget items, so it could not discover the
actual two-widget status rows and failed with `row is None` even though the application
layout was correct. This correction is test/documentation-only and does not change
application behavior.

