"""Appearance themes: token contrast, stylesheet validity, selection, assets."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from PySide6.QtCore import QtMsgType, qInstallMessageHandler
from PySide6.QtGui import QColor, QFontDatabase
from PySide6.QtWidgets import QApplication, QPushButton, QWidget
from pytestqt.qtbot import QtBot

from nlab import settings_keys
from nlab.views import general_settings_dialog as dialog_module
from nlab.views import theme
from nlab.views.theme import ThemeColors, ThemeMode


def _luminance(hex_color: str) -> float:
    color = QColor(hex_color)
    channels = []
    for value in (color.redF(), color.greenF(), color.blueF()):
        channels.append(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(foreground: str, background: str) -> float:
    lighter, darker = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _text_pairs(c: ThemeColors) -> list[tuple[str, str, str]]:
    """Text that must stay readable (WCAG AA, 4.5:1)."""
    pairs = [
        ("text/bg", c.text, c.bg),
        ("text/raised", c.text, c.bg_raised),
        ("text/tint", c.text, c.bg_tint),
        ("muted/bg", c.text_muted, c.bg),
        ("muted/raised", c.text_muted, c.bg_raised),
        ("accent/bg", c.text_accent, c.bg),
        ("on-action/action", c.on_action, c.action),
        ("on-action/hover", c.on_action, c.action_hover),
        ("on-action/pressed", c.on_action, c.action_pressed),
        ("on-running/running", c.on_running, c.running_bg),
        ("on-stop/stop", c.on_stop, c.stop_bg),
        ("on-stop/pressed", c.on_stop, c.stop_pressed),
        ("tooltip", c.tooltip_text, c.tooltip_bg),
        ("console", c.console_text, c.console_bg),
        ("console-debug", c.console_debug, c.console_bg),
        ("plot axis", c.plot_foreground, c.plot_background),
        ("text/success-tint", c.text, c.status_success_tint),
        ("text/error-tint", c.text, c.status_error_tint),
    ]
    for status in ("success", "warning", "error"):
        color = getattr(c, f"status_{status}")
        pairs += [
            (f"{status}/bg", color, c.bg),
            (f"{status}/raised", color, c.bg_raised),
            (f"{status}/tint", color, getattr(c, f"status_{status}_tint")),
        ]
    for kind in ("offline", "replay", "active", "calibrated"):
        badge = getattr(c, f"badge_{kind}")
        pairs.append((f"badge {kind}", badge.foreground, badge.background))
    return pairs


def _reduced_pairs(c: ThemeColors) -> list[tuple[str, str, str]]:
    """Disabled text and non-text UI: 3:1. Offline panels are read while locked."""
    return [
        ("disabled/bg", c.text_disabled, c.bg),
        ("disabled/raised", c.text_disabled, c.bg_raised),
        ("control border", c.line_strong, c.bg),
        ("focus ring", c.focus, c.bg),
        ("plot ink", c.plot_ink, c.plot_background),
        ("start disabled", c.action_disabled_text, c.action_disabled_bg),
        ("stop disabled", c.stop_disabled_text, c.stop_disabled_bg),
    ]


_EWT_THEMES = [theme.LIGHT, theme.DARK]


@pytest.mark.parametrize(
    ("label", "foreground", "background"),
    [(f"{c.mode}:{label}", fg, bg) for c in _EWT_THEMES for label, fg, bg in _text_pairs(c)],
)
def test_text_pairs_meet_wcag_aa(label: str, foreground: str, background: str) -> None:
    assert _contrast(foreground, background) >= 4.5, label


@pytest.mark.parametrize(
    ("label", "foreground", "background"),
    [(f"{c.mode}:{label}", fg, bg) for c in _EWT_THEMES for label, fg, bg in _reduced_pairs(c)],
)
def test_disabled_text_and_controls_keep_three_to_one(
    label: str, foreground: str, background: str,
) -> None:
    assert _contrast(foreground, background) >= 3.0, label


def test_brand_values_match_design_system() -> None:
    # Owner-set screen values of the CMYK masters; never adjust them here.
    assert (theme.BRAND_INK, theme.BRAND_BLUE, theme.BRAND_CYAN, theme.BRAND_WHITE) == (
        "#00000C", "#0472B9", "#00B9D4", "#FFFFFF",
    )
    # Cyan is a fill, never text on white; blue is too dark as text on ink.
    assert theme.LIGHT.text_accent == theme.BRAND_BLUE
    assert theme.DARK.text_accent == theme.BRAND_CYAN


def test_classic_keeps_pre_redesign_colors() -> None:
    classic = theme.CLASSIC
    assert classic.plot_background.lower() == "#f8f9fa"
    assert classic.status_error.lower() == "#b00020"
    assert classic.console_bg.lower() == "#1e1e1e"
    stylesheet = theme.application_stylesheet(classic)
    assert "#4CAF50" in stylesheet and "#F44336" in stylesheet
    assert "QComboBox" not in stylesheet  # Native style everywhere else.


@pytest.fixture
def qt_warnings() -> Iterator[list[str]]:
    messages: list[str] = []

    def handler(mode: QtMsgType, _context: object, message: str) -> None:
        if mode != QtMsgType.QtDebugMsg:
            messages.append(message)

    previous = qInstallMessageHandler(handler)
    try:
        yield messages
    finally:
        qInstallMessageHandler(previous)


@pytest.mark.parametrize("mode", list(ThemeMode))
@pytest.mark.parametrize("svg_icons", [True, False])
def test_application_stylesheet_parses(
    qtbot: QtBot, qt_warnings: list[str], mode: ThemeMode, svg_icons: bool,
) -> None:
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.setStyleSheet(theme.application_stylesheet(theme.THEMES[mode], svg_icons=svg_icons))
    button = QPushButton("Start", widget)
    theme.set_button_role(button, "start")
    widget.show()
    QApplication.processEvents()
    assert not [message for message in qt_warnings if "parse" in message.lower()]
    assert button.property(theme.BUTTON_ROLE_PROPERTY) == "start"


class _FakeSettings:
    store: dict[str, Any] = {}

    def value(self, key: str, default: Any = None) -> Any:
        return self.store.get(key, default)

    def setValue(self, key: str, value: Any) -> None:  # noqa: N802 - Qt API
        self.store[key] = value


@pytest.fixture
def fake_settings(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    store: dict[str, Any] = {}
    monkeypatch.setattr(_FakeSettings, "store", store)
    monkeypatch.setattr(theme, "QSettings", _FakeSettings)
    monkeypatch.delenv(theme.THEME_ENVIRONMENT_VARIABLE, raising=False)
    return store


def test_theme_selection_defaults_persists_and_env_overrides(
    fake_settings: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert theme.startup_theme_mode() is ThemeMode.LIGHT
    theme.save_theme_mode(ThemeMode.DARK)
    assert fake_settings[settings_keys.APPEARANCE_THEME] == "dark"
    assert theme.startup_theme_mode() is ThemeMode.DARK
    monkeypatch.setenv(theme.THEME_ENVIRONMENT_VARIABLE, "system")
    assert theme.startup_theme_mode() is ThemeMode.CLASSIC
    monkeypatch.setenv(theme.THEME_ENVIRONMENT_VARIABLE, "bogus")
    assert theme.startup_theme_mode() is ThemeMode.DARK
    fake_settings[settings_keys.APPEARANCE_THEME] = "unknown"
    assert theme.saved_theme_mode() is ThemeMode.LIGHT


def test_general_settings_saves_theme_on_accept(
    qtbot: QtBot, fake_settings: dict[str, Any],
) -> None:
    theme.save_theme_mode(ThemeMode.CLASSIC)
    dialog = dialog_module.GeneralSettingsDialog("192.0.2.10", {})
    qtbot.addWidget(dialog)
    assert dialog.theme_mode is ThemeMode.CLASSIC
    dialog.theme_combo.setCurrentIndex(dialog.theme_combo.findData(ThemeMode.DARK.value))
    assert fake_settings[settings_keys.APPEARANCE_THEME] == "classic"  # Not until OK.
    dialog.accept()
    assert fake_settings[settings_keys.APPEARANCE_THEME] == "dark"


class _RecordingApp:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def setStyle(self, _style: str) -> None:  # noqa: N802 - Qt API
        self.calls.append("style")

    def setPalette(self, _palette: object) -> None:  # noqa: N802 - Qt API
        self.calls.append("palette")

    def setFont(self, _font: object) -> None:  # noqa: N802 - Qt API
        self.calls.append("font")

    def setStyleSheet(self, sheet: str) -> None:  # noqa: N802 - Qt API
        self.calls.append("sheet")
        self.sheet = sheet


@pytest.mark.parametrize("mode", list(ThemeMode))
def test_apply_activates_theme_colors(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, mode: ThemeMode,
) -> None:
    monkeypatch.setattr(theme, "_active", theme.LIGHT)
    monkeypatch.setattr(theme, "configure_plots", lambda _theme: None)
    app = _RecordingApp()
    applied = theme.apply_application_theme(cast(QApplication, app), mode)
    assert applied is theme.THEMES[mode] is theme.colors()
    if mode is ThemeMode.CLASSIC:
        assert app.calls == ["sheet"]  # Native style, palette and font.
    else:
        assert app.calls[:2] == ["style", "palette"] and app.calls[-1] == "sheet"
    assert theme.badge_style("active").count(applied.badge_active.background) == 2
    assert theme.status_color(error=True) == applied.status_error


def test_bundled_fonts_register_every_weight(qtbot: QtBot) -> None:
    assert theme.register_fonts()
    styles = set(QFontDatabase.styles(theme.FONT_FAMILY))
    assert {"Regular", "Medium", "Bold"} <= styles
    assert theme.application_font().families()[0] == theme.FONT_FAMILY


@pytest.mark.parametrize("mode", list(ThemeMode))
def test_splash_follows_theme_and_draws_brand_lockup(qtbot: QtBot, mode: ThemeMode) -> None:
    palette = theme.THEMES[mode]
    style = theme.splash_style(palette)
    image = theme.brand_splash_pixmap(520, 320, palette).toImage()
    assert (image.width(), image.height()) == (520, 320)
    assert image.pixelColor(10, 300).name().upper() == style.background.upper()
    assert style is (theme.DARK_SPLASH if palette.dark else theme.LIGHT_SPLASH)
    # The "Eastern Wall Technologies" wordmark is drawn from the bundled SVG in
    # the wordmark colour for this ground: ink on light, white on dark.
    wordmark = QColor(theme.BRAND_WHITE if palette.dark else theme.BRAND_INK)
    wordmark_pixels = sum(
        1
        for x in range(150, 370, 2)
        for y in range(120, 206, 2)
        if abs(image.pixelColor(x, y).lightness() - wordmark.lightness()) < 40
    )
    assert wordmark_pixels > 200
    assert _contrast(style.title, style.background) >= 4.5
    assert _contrast(style.message, style.background) >= 4.5


def test_dark_lockup_only_recolours_the_wordmark() -> None:
    brand = Path(__file__).resolve().parents[1] / "resources" / "brand"
    light = (brand / "ewt-vertical-color-on-light.svg").read_text(encoding="utf-8")
    dark = (brand / "ewt-vertical-color-on-dark.svg").read_text(encoding="utf-8")
    assert dark == light.replace('fill="#00000C"', 'fill="#FFFFFF"')
    assert 'fill="#00000C"' not in dark

_COMBO_POPUP_PROBE = """
import os, sys
from PySide6.QtWidgets import QApplication, QComboBox
app = QApplication([sys.argv[0], "-platform", "offscreen"])
from nlab.views import theme
theme.apply_application_theme(app, theme.ThemeMode(sys.argv[1]))
combo = QComboBox()
combo.addItems([f"Debug source {index}" for index in range(10)])
combo.show()
app.processEvents()
combo.showPopup()
app.processEvents()
view = combo.view()
needed = sum(view.sizeHintForRow(row) for row in range(combo.count()))
print(view.verticalScrollBar().maximum(), view.viewport().height(), needed, flush=True)
os._exit(0)
"""


@pytest.mark.parametrize("mode", [ThemeMode.LIGHT, ThemeMode.DARK])
def test_combo_popup_shows_every_row_up_to_max_visible(mode: ThemeMode) -> None:
    """Padding on the popup view used to clip the last row behind a scroll arrow.

    Needs Fusion plus the application-wide stylesheet, so it runs in its own
    process rather than restyling this test session's QApplication.
    """
    result = subprocess.run(
        [sys.executable, "-c", _COMBO_POPUP_PROBE, mode.value],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    scroll_maximum, viewport_height, needed = map(int, result.stdout.split())
    assert scroll_maximum == 0
    assert viewport_height >= needed


@pytest.mark.parametrize("palette", _EWT_THEMES, ids=lambda c: c.mode.value)
def test_series_colors_become_legible_text_without_losing_hue(palette: ThemeColors) -> None:
    okabe_ito = ("#c23b22", "#0072b2", "#009e73", "#cc79a7", "#e69f00", "#6f4e7c")
    for color in okabe_ito:
        legible = theme.legible_text(color, palette)
        assert _contrast(legible, palette.bg) >= 4.5, (color, legible)
        if _contrast(color, palette.bg) >= 4.5:
            assert legible == color  # Readable colours are left alone.
    assert theme.legible_text("#e69f00", theme.CLASSIC) == "#e69f00"


@pytest.mark.parametrize("mode", list(ThemeMode))
def test_plot_styles_follow_theme(qtbot: QtBot, mode: ThemeMode) -> None:
    palette = theme.THEMES[mode]
    region = theme.region_style(palette)
    colormap = theme.persistence_colormap(palette)
    if mode is ThemeMode.CLASSIC:
        assert region == {}  # pyqtgraph defaults, as before the redesign.
        assert colormap.name == "viridis"
        return
    assert {"brush", "hoverBrush", "pen", "hoverPen"} <= set(region)
    # An empty persistence buffer (0) is the plot background, not a colour block.
    assert colormap.map(0.0, mode="qcolor").name().upper() == palette.plot_background.upper()
    # The Raw trace is a thin line: graphics contrast (3:1) on the canvas.
    assert _contrast(palette.scope_trace, palette.plot_background) >= 3.0
