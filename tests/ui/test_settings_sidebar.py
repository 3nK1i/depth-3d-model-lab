"""Structural tests for the right settings sidebar.

Locks in the current accordion-card layout: three independent expanded
cards share the available sidebar height according to configured stretch
factors, while collapsed cards return their stretch share to the others.
"""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QSizePolicy

from app.main_window import MainWindow
from app.widgets.settings_card import CollapsibleSettingsCard
from app.widgets.settings_controls import (
    CONTROL_HEIGHT,
    SettingsComboBox,
    StatusBadge,
    ValueBox,
)
from app.widgets.settings_panel import SIDEBAR_MIN_WIDTH
from app.widgets.toggle_switch import ToggleSwitch


@pytest.fixture
def sidebar(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1536, 864)
    win.show()
    qtbot.waitExposed(win)
    yield win._settings_panel


def test_three_independent_cards(sidebar):
    cards = sidebar.findChildren(CollapsibleSettingsCard)
    assert len(cards) == 3
    titles = [c._header._title.text() for c in cards]
    assert titles == ["MODEL", "DEPTH SETTINGS", "RENDER SETTINGS"]


def test_expanded_cards_share_available_height(sidebar):
    cards = sidebar.findChildren(CollapsibleSettingsCard)
    assert cards

    # The production layout intentionally gives expanded cards vertical
    # stretch shares (33/36/31) so they fill the sidebar without leaving
    # a large dead region below the last card.
    assert set(sidebar._card_stretches) == set(cards)
    for card in cards:
        assert card.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Expanding
        assert card._body.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Expanding


def test_visible_gaps_between_cards(sidebar):
    cards = sidebar.findChildren(CollapsibleSettingsCard)
    for previous, following in zip(cards, cards[1:]):
        gap = following.geometry().y() - previous.geometry().bottom()
        assert 8 <= gap <= 14


def test_model_selector_and_badge_share_one_row(sidebar):
    # Address the exact model badge instead of relying on findChildren()
    # ordering, which is not a stable Qt API contract.  Map both widget
    # centres into sidebar coordinates before comparing them.
    combo_center = sidebar._model_combo.mapTo(
        sidebar, sidebar._model_combo.rect().center()
    )
    badge_center = sidebar._model_badge.mapTo(
        sidebar, sidebar._model_badge.rect().center()
    )
    assert abs(badge_center.y() - combo_center.y()) <= 3


def test_slider_values_are_boxed(sidebar):
    boxes = sidebar.findChildren(ValueBox)
    assert len(boxes) == 4
    assert [b.text() for b in boxes] == ["0.1 m", "8.0 m", "0.6", "0.4"]


def test_combo_boxes_are_compact(sidebar):
    combos = sidebar.findChildren(SettingsComboBox)
    assert combos
    for combo in combos:
        # Height is pinned to the shared constant; the bound just guards
        # against anything re-inflating the controls.
        assert combo.height() == CONTROL_HEIGHT
        assert CONTROL_HEIGHT <= 36


def test_toggles_use_compact_shared_size(sidebar):
    toggles = sidebar.findChildren(ToggleSwitch)
    assert len(toggles) == 4
    for toggle in toggles:
        assert 40 <= toggle.width() <= 44
        assert 20 <= toggle.height() <= 22


def test_controls_share_a_right_edge(sidebar):
    # Every control except the model selector (which shares its row with
    # the badge) ends at the same x.
    combos = [c for c in sidebar.findChildren(SettingsComboBox) if c is not sidebar._model_combo]
    boxes = sidebar.findChildren(ValueBox)
    edges = {c.geometry().right() for c in combos} | {b.geometry().right() for b in boxes}
    assert len(edges) == 1


def test_sidebar_respects_minimum_width(sidebar):
    assert sidebar.minimumWidth() == SIDEBAR_MIN_WIDTH


def test_card_collapses_and_expands(sidebar, qtbot):
    card = sidebar.findChildren(CollapsibleSettingsCard)[1]
    assert card._body.isVisible()
    card._header._on_clicked()
    assert not card._body.isVisible()
    card._header._on_clicked()
    assert card._body.isVisible()

def test_vsync_is_enabled_at_canvas_creation_and_not_runtime_toggleable(sidebar):
    assert sidebar._vsync_toggle.isChecked()
    assert not sidebar._vsync_toggle.isEnabled()
    assert "VisPy canvas is created" in sidebar._vsync_toggle.toolTip()
