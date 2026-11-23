"""Application appearance: Classic, EWT light and EWT dark themes.

One module owns every interface colour, the typeface and the application
stylesheet, so widgets ask for semantic roles (``start``/``stop`` buttons,
status text, badges, plot canvas) instead of literal colours. Read colours
through :func:`colors` when a widget is built: the theme is chosen once at
start-up (``General Settings > Appearance``) and applies after a restart.

EWT token values come from the Eastern Wall Technologies design system
(Claude design-system artifact, tokens.json v1, 2026-11-14):

* Brand colours (``BRAND_*``) are owner-set screen values of the CMYK masters.
* Light and dark neutrals are the design system's themes, which it marks as
  derived proposals.
* Status colours, hover/pressed/disabled shades and the dark-theme status and
  tint values are NOT defined by the design system ("Not yet defined: status
  colours"). They are derived here, checked for WCAG contrast in
  ``tests/test_theme.py``, and should be proposed back to the design system.

The design system's typeface (HK Grotesk) is kept for documents; the GUI uses
Roboto, whose shorter ascenders suit dense controls.

``CLASSIC`` reproduces the pre-redesign interface: the platform style and font,
the original green/red Start/Stop buttons and the original status colours.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from PySide6.QtCore import QRectF, QSettings, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QImageReader,
    QPainter,
    QPalette,
    QPixmap,
)
from PySide6.QtWidgets import QApplication, QWidget

from nlab import settings_keys

log = logging.getLogger(__name__)


class ThemeMode(StrEnum):
    CLASSIC = "classic"
    LIGHT = "light"
    DARK = "dark"


THEME_LABELS = {
    ThemeMode.LIGHT: "EWT light",
    ThemeMode.DARK: "EWT dark",
    ThemeMode.CLASSIC: "Classic (system style)",
}
DEFAULT_THEME = ThemeMode.LIGHT
# Overrides the saved choice for one launch, e.g. NLAB_THEME=classic.
THEME_ENVIRONMENT_VARIABLE = "NLAB_THEME"

# --- Brand (design system: owner-set) ----------------------------------------
BRAND_INK = "#00000C"
BRAND_BLUE = "#0472B9"
BRAND_CYAN = "#00B9D4"
BRAND_WHITE = "#FFFFFF"

# Dynamic property read by the application stylesheet.
BUTTON_ROLE_PROPERTY = "nlabRole"


@dataclass(frozen=True)
class Badge:
    background: str
    foreground: str
    border: str


@dataclass(frozen=True)
class ThemeColors:
    """Every colour role the interface uses, for one theme."""

    mode: ThemeMode
    dark: bool
    bg: str
    bg_raised: str
    bg_tint: str
    bg_pressed: str
    line: str
    line_strong: str
    text: str
    text_muted: str
    text_accent: str
    text_disabled: str
    focus: str
    action: str
    action_hover: str
    action_pressed: str
    action_disabled_bg: str
    action_disabled_text: str
    on_action: str
    running_bg: str
    on_running: str
    stop_bg: str
    stop_pressed: str
    stop_disabled_bg: str
    stop_disabled_text: str
    on_stop: str
    status_success: str
    status_warning: str
    status_error: str
    status_success_tint: str
    status_warning_tint: str
    status_error_tint: str
    tooltip_bg: str
    tooltip_text: str
    console_bg: str
    console_text: str
    console_debug: str
    plot_background: str
    plot_foreground: str
    plot_ink: str  # Neutral guides and markers drawn on the plot canvas.
    scope_trace: str  # The single Raw-mode Scope waveform.
    # Scope Persistence colour stops, 0 = no hits (the plot background) to
    # 1 = the most recent, most frequent trace. Empty selects viridis.
    persistence_stops: tuple[tuple[float, str], ...]
    icon_suffix: str
    badge_offline: Badge
    badge_replay: Badge
    badge_active: Badge
    badge_calibrated: Badge


LIGHT = ThemeColors(
    mode=ThemeMode.LIGHT,
    dark=False,
    bg=BRAND_WHITE,
    bg_raised="#F2F4F6",
    bg_tint="#E6F7FA",
    bg_pressed="#E4E8EC",
    line="#C9D0D6",
    line_strong="#77838F",
    text=BRAND_INK,
    text_muted="#4A5561",
    text_accent=BRAND_BLUE,
    text_disabled="#77838F",  # Offline panels stay legible while locked.
    focus=BRAND_BLUE,
    action=BRAND_BLUE,
    action_hover="#03619D",
    action_pressed="#024F80",
    action_disabled_bg="#D7E6F2",
    action_disabled_text="#4F6F88",
    on_action=BRAND_WHITE,
    running_bg=BRAND_CYAN,
    on_running=BRAND_INK,
    stop_bg="#C0262D",
    stop_pressed="#9A1C22",
    stop_disabled_bg="#F6DCDD",
    stop_disabled_text="#8E4A4D",
    on_stop=BRAND_WHITE,
    status_success="#16733F",
    status_warning="#A14E00",
    status_error="#C0262D",
    status_success_tint="#E8F5EC",
    status_warning_tint="#FDF0E1",
    status_error_tint="#FBE7E8",
    tooltip_bg=BRAND_INK,
    tooltip_text="#EEF1F4",
    console_bg=BRAND_INK,
    console_text="#EEF1F4",
    console_debug=BRAND_CYAN,
    plot_background=BRAND_WHITE,
    plot_foreground="#4A5561",
    plot_ink=BRAND_INK,
    scope_trace=BRAND_BLUE,
    persistence_stops=(
        (0.0, BRAND_WHITE),
        (0.08, "#C7EEF5"),
        (0.35, BRAND_CYAN),
        (0.7, BRAND_BLUE),
        (1.0, BRAND_INK),
    ),
    icon_suffix="",
    badge_offline=Badge(BRAND_INK, BRAND_WHITE, BRAND_INK),
    badge_replay=Badge(BRAND_CYAN, BRAND_INK, BRAND_CYAN),
    badge_active=Badge("#16733F", BRAND_WHITE, "#16733F"),
    badge_calibrated=Badge("#E6F7FA", BRAND_INK, "#C9D0D6"),
)

DARK = ThemeColors(
    mode=ThemeMode.DARK,
    dark=True,
    bg=BRAND_INK,
    bg_raised="#0D1218",
    bg_tint="#062530",
    bg_pressed="#1A232C",
    line="#26313B",
    line_strong="#5C6875",
    text="#EEF1F4",
    text_muted="#9AA5B1",
    text_accent=BRAND_CYAN,  # Blue is only 4.1:1 on ink.
    text_disabled="#6B7784",
    focus=BRAND_CYAN,
    action=BRAND_BLUE,
    action_hover="#03619D",
    action_pressed="#024F80",
    action_disabled_bg="#0B2A40",
    action_disabled_text="#8FB3CF",
    on_action=BRAND_WHITE,
    running_bg=BRAND_CYAN,
    on_running=BRAND_INK,
    stop_bg="#C0262D",
    stop_pressed="#9A1C22",
    stop_disabled_bg="#3A1416",
    stop_disabled_text="#D8999B",
    on_stop=BRAND_WHITE,
    status_success="#4CC38A",
    status_warning="#F0A23B",
    status_error="#FF6B6E",
    status_success_tint="#0B2618",
    status_warning_tint="#2E1D06",
    status_error_tint="#33100F",
    tooltip_bg="#26313B",
    tooltip_text="#EEF1F4",
    console_bg=BRAND_INK,
    console_text="#EEF1F4",
    console_debug=BRAND_CYAN,
    plot_background=BRAND_INK,
    plot_foreground="#9AA5B1",
    plot_ink="#EEF1F4",
    scope_trace=BRAND_CYAN,
    persistence_stops=(
        (0.0, BRAND_INK),
        (0.08, "#062530"),
        (0.35, BRAND_BLUE),
        (0.7, BRAND_CYAN),
        (1.0, "#EEF1F4"),
    ),
    icon_suffix="-dark",
    badge_offline=Badge("#EEF1F4", BRAND_INK, "#EEF1F4"),
    badge_replay=Badge(BRAND_CYAN, BRAND_INK, BRAND_CYAN),
    badge_active=Badge("#16733F", BRAND_WHITE, "#16733F"),
    badge_calibrated=Badge("#062530", "#EEF1F4", "#26313B"),
)

# Pre-redesign values, so Classic reproduces the original interface. Only the
# roles read by widget code matter: Classic installs no palette and no
# stylesheet beyond the Start/Stop rules.
CLASSIC = ThemeColors(
    mode=ThemeMode.CLASSIC,
    dark=False,
    bg="#FFFFFF",
    bg_raised="#F0F0F0",
    bg_tint="#E5F1FB",
    bg_pressed="#E0E0E0",
    line="#C8C8C8",
    line_strong="#7A7A7A",
    text="#000000",
    text_muted="#505050",
    text_accent="#0066CC",
    text_disabled="#8A8A8A",
    focus="#0078D4",
    action="#4CAF50",
    action_hover="#4CAF50",
    action_pressed="#2E7D32",
    action_disabled_bg="#A5D6A7",
    action_disabled_text="#E0E0E0",
    on_action="#FFFFFF",
    running_bg="#2E7D32",
    on_running="#FFFFFF",
    stop_bg="#F44336",
    stop_pressed="#C62828",
    stop_disabled_bg="#EF9A9A",
    stop_disabled_text="#E0E0E0",
    on_stop="#FFFFFF",
    status_success="#2E7D32",
    status_warning="#B54708",
    status_error="#B00020",
    status_success_tint="#DFF0D8",
    status_warning_tint="#FFF3CD",
    status_error_tint="#F8D7DA",
    tooltip_bg="#FFFFDC",
    tooltip_text="#000000",
    console_bg="#1E1E1E",
    console_text="#D4D4D4",
    console_debug="#7EC8E3",
    plot_background="#F8F9FA",
    plot_foreground="#969696",
    plot_ink="#000000",
    scope_trace="#00BFFF",
    persistence_stops=(),
    icon_suffix="",
    badge_offline=Badge("#455A64", "#FFFFFF", "#455A64"),
    badge_replay=Badge("#7353A6", "#FFFFFF", "#7353A6"),
    badge_active=Badge("#2E7D32", "#FFFFFF", "#2E7D32"),
    badge_calibrated=Badge("#2E7D32", "#FFFFFF", "#2E7D32"),
)

THEMES = {theme.mode: theme for theme in (LIGHT, DARK, CLASSIC)}
_active: ThemeColors = LIGHT


def colors() -> ThemeColors:
    """The active theme's colours (the light theme until one is applied)."""
    return _active


# --- Splash ---------------------------------------------------------------------
@dataclass(frozen=True)
class SplashStyle:
    background: str
    title: str
    message: str
    rule: str
    lockup_resource: str


# The design system's colour lockup is drawn for light grounds. The on-dark
# file is derived from it by changing only the wordmark fill from brand ink to
# brand white (see resources/brand/README.md) until an official reversed
# lockup exists.
LIGHT_SPLASH = SplashStyle(
    background=BRAND_WHITE,
    title=BRAND_INK,
    message=LIGHT.text_muted,
    rule=LIGHT.line,
    lockup_resource=":/brand/ewt-vertical-color-on-light.svg",
)
DARK_SPLASH = SplashStyle(
    background=BRAND_INK,
    title=DARK.text,
    message=DARK.text_muted,
    rule=DARK.line,
    lockup_resource=":/brand/ewt-vertical-color-on-dark.svg",
)


def splash_style(theme: ThemeColors | None = None) -> SplashStyle:
    """Splash colours for a theme (the active one by default); Classic is light."""
    return DARK_SPLASH if (theme or colors()).dark else LIGHT_SPLASH

# --- Typography (EWT themes) ---------------------------------------------------
FONT_FAMILY = "Roboto"
FONT_FALLBACKS = ("Segoe UI", "Helvetica Neue", "Arial", "sans-serif")
APP_FONT_POINT_SIZE = 9.0
_FONT_RESOURCES = (
    ":/fonts/Roboto-Regular.ttf",
    ":/fonts/Roboto-Medium.ttf",
    ":/fonts/Roboto-Bold.ttf",
)
_fonts_loaded: bool | None = None


def badge_style(kind: str) -> str:
    """QLabel stylesheet for a status-bar badge: offline/replay/active/calibrated."""
    badge: Badge = getattr(colors(), f"badge_{kind}")
    return (
        f"QLabel {{ background: {badge.background}; color: {badge.foreground}; "
        f"border: 1px solid {badge.border}; border-radius: 4px; "
        "font-weight: 700; padding: 1px 8px; }"
    )


def status_color(*, error: bool = False, warning: bool = False) -> str:
    """Text colour for a status message: error, warning, otherwise success."""
    palette = colors()
    if error:
        return palette.status_error
    if warning:
        return palette.status_warning
    return palette.status_success


def set_button_role(button: QWidget, role: str) -> None:
    """Tag a button ``start`` or ``stop`` for the application stylesheet."""
    button.setProperty(BUTTON_ROLE_PROPERTY, role)
    style = button.style()
    style.unpolish(button)
    style.polish(button)


def saved_theme_mode() -> ThemeMode:
    """The theme chosen in General Settings, or the light default."""
    value = QSettings().value(settings_keys.APPEARANCE_THEME, DEFAULT_THEME.value)
    try:
        return ThemeMode(str(value))
    except ValueError:
        return DEFAULT_THEME


def save_theme_mode(mode: ThemeMode) -> None:
    QSettings().setValue(settings_keys.APPEARANCE_THEME, mode.value)


def startup_theme_mode() -> ThemeMode:
    """The theme for this launch: the environment override, else the saved one."""
    override = os.environ.get(THEME_ENVIRONMENT_VARIABLE, "").strip().lower()
    if override == "system":  # Accept the original variable value.
        override = ThemeMode.CLASSIC.value
    if override:
        try:
            return ThemeMode(override)
        except ValueError:
            log.warning("Ignoring unknown %s=%s", THEME_ENVIRONMENT_VARIABLE, override)
    return saved_theme_mode()


def register_fonts() -> bool:
    """Register the bundled Roboto faces once; False when none loaded."""
    global _fonts_loaded
    if _fonts_loaded is None:
        try:
            from nlab.ui import resources_rc  # noqa: F401 - registers ":/fonts"
        except ImportError:
            log.warning("Qt resources are not compiled; run scripts/build_ui.py")
        loaded = False
        for resource in _FONT_RESOURCES:
            if QFontDatabase.addApplicationFont(resource) >= 0:
                loaded = True
            else:
                log.warning("Could not load bundled font %s", resource)
        _fonts_loaded = loaded
    return _fonts_loaded


def application_font() -> QFont:
    font = QFont()
    font.setFamilies([FONT_FAMILY, *FONT_FALLBACKS])
    font.setPointSizeF(APP_FONT_POINT_SIZE)
    return font


def application_palette(theme: ThemeColors) -> QPalette:
    palette = QPalette()
    roles = QPalette.ColorRole
    if theme.dark:
        bevels = {
            roles.Light: "#2A3540",
            roles.Midlight: "#1A232C",
            roles.Mid: theme.line,
            roles.Dark: "#05080C",
            roles.Shadow: "#000000",
        }
    else:
        bevels = {
            roles.Light: BRAND_WHITE,
            roles.Midlight: theme.bg_raised,
            roles.Mid: theme.line,
            roles.Dark: theme.line_strong,
            roles.Shadow: theme.text_muted,
        }
    values = {
        roles.Window: theme.bg,
        roles.WindowText: theme.text,
        roles.Base: theme.bg,
        roles.AlternateBase: theme.bg_raised,
        roles.Text: theme.text,
        roles.PlaceholderText: theme.text_muted,
        roles.Button: theme.bg,
        roles.ButtonText: theme.text,
        roles.BrightText: BRAND_WHITE,
        roles.Highlight: theme.action,
        roles.HighlightedText: theme.on_action,
        roles.Link: theme.text_accent,
        roles.LinkVisited: theme.text_accent,
        roles.ToolTipBase: theme.tooltip_bg,
        roles.ToolTipText: theme.tooltip_text,
        **bevels,
    }
    for role, value in values.items():
        palette.setColor(role, QColor(value))
    disabled = QPalette.ColorGroup.Disabled
    for role in (roles.WindowText, roles.Text, roles.ButtonText):
        palette.setColor(disabled, role, QColor(theme.text_disabled))
    palette.setColor(disabled, roles.Base, QColor(theme.bg_raised))
    palette.setColor(disabled, roles.Button, QColor(theme.bg_raised))
    palette.setColor(disabled, roles.Highlight, QColor(theme.line_strong))
    return palette


def _svg_images_supported() -> bool:
    return b"svg" in QImageReader.supportedImageFormats()


def _role_button_rules(c: ThemeColors) -> str:
    return f"""
QPushButton[nlabRole="start"] {{
    background: {c.action}; color: {c.on_action}; border: 1px solid {c.action};
    font-weight: 700;
}}
QPushButton[nlabRole="start"]:hover {{
    background: {c.action_hover}; border-color: {c.action_hover};
}}
QPushButton[nlabRole="start"]:pressed {{
    background: {c.action_pressed}; border-color: {c.action_pressed};
}}
QPushButton[nlabRole="start"]:disabled {{
    background: {c.action_disabled_bg}; color: {c.action_disabled_text};
    border-color: {c.action_disabled_bg};
}}
/* Running: a checked Start is highlighted. */
QPushButton[nlabRole="start"]:checked,
QPushButton[nlabRole="start"]:checked:disabled {{
    background: {c.running_bg}; color: {c.on_running}; border: 1px solid {c.running_bg};
}}
QPushButton[nlabRole="stop"] {{
    background: {c.stop_bg}; color: {c.on_stop}; border: 1px solid {c.stop_bg};
    font-weight: 700;
}}
QPushButton[nlabRole="stop"]:hover, QPushButton[nlabRole="stop"]:pressed,
QPushButton[nlabRole="stop"]:checked {{
    background: {c.stop_pressed}; border-color: {c.stop_pressed};
}}
QPushButton[nlabRole="stop"]:disabled {{
    background: {c.stop_disabled_bg}; color: {c.stop_disabled_text};
    border-color: {c.stop_disabled_bg};
}}
"""


def classic_stylesheet() -> str:
    """Only the original Start/Stop button rules; everything else is native."""
    c = CLASSIC
    return f"""
QPushButton[nlabRole="start"] {{
    background-color: {c.action}; color: white; font-weight: bold;
}}
QPushButton[nlabRole="start"]:checked {{
    background-color: {c.running_bg}; border: 2px solid #1B5E20;
}}
QPushButton[nlabRole="start"]:disabled {{
    background-color: {c.action_disabled_bg}; color: {c.action_disabled_text};
}}
QPushButton[nlabRole="stop"] {{
    background-color: {c.stop_bg}; color: white; font-weight: bold;
}}
QPushButton[nlabRole="stop"]:checked {{
    background-color: {c.stop_pressed}; border: 2px solid #B71C1C;
}}
QPushButton[nlabRole="stop"]:disabled {{
    background-color: {c.stop_disabled_bg}; color: {c.stop_disabled_text};
}}
"""


def application_stylesheet(theme: ThemeColors | None = None, *, svg_icons: bool = True) -> str:
    """Return the stylesheet for an EWT theme (light by default).

    ``svg_icons`` styles combo-box and spin-box arrows with the bundled SVG
    chevrons; without Qt's SVG image plugin those rules are omitted and the
    Fusion style draws its own arrows.
    """
    c = theme or LIGHT
    if c.mode is ThemeMode.CLASSIC:
        return classic_stylesheet()
    icon = c.icon_suffix
    arrows = ""
    if svg_icons:
        arrows = f"""
QComboBox::drop-down {{
    subcontrol-origin: padding; subcontrol-position: center right;
    width: 20px; border: none;
}}
QComboBox::down-arrow {{ image: url(:/icons/chevron-down{icon}.svg); width: 12px; height: 12px; }}
QComboBox::down-arrow:disabled {{ image: url(:/icons/chevron-down{icon}-disabled.svg); }}
QAbstractSpinBox {{ padding-right: 18px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    subcontrol-origin: border; width: 16px; border: none;
    border-left: 1px solid {c.line}; background: transparent;
}}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; border-top-right-radius: 4px; }}
QAbstractSpinBox::down-button {{
    subcontrol-position: bottom right; border-bottom-right-radius: 4px;
}}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
    background: {c.bg_raised};
}}
QCheckBox::indicator:checked, QGroupBox::indicator:checked {{
    image: url(:/icons/check.svg);
}}
QAbstractSpinBox::up-arrow {{
    image: url(:/icons/chevron-up{icon}.svg); width: 10px; height: 10px;
}}
QAbstractSpinBox::down-arrow {{
    image: url(:/icons/chevron-down{icon}.svg); width: 10px; height: 10px;
}}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{
    image: url(:/icons/chevron-up{icon}-disabled.svg);
}}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{
    image: url(:/icons/chevron-down{icon}-disabled.svg);
}}
/* Under a style sheet Qt draws the dock float icon at a 10 px cap. Blank that
   icon and draw the themed one as the button image, which fills the button. */
QDockWidget {{ titlebar-normal-icon: url(:/icons/blank.svg); }}
QDockWidget::float-button {{ image: url(:/icons/undock{icon}.svg); }}
"""
    return f"""
/* Eastern Wall Technologies {c.mode.value} theme — generated by nlab.views.theme */
QToolTip {{
    background: {c.tooltip_bg}; color: {c.tooltip_text}; border: none;
    border-radius: 4px; padding: 6px 8px;
}}

/* Chrome: menu bar, main tab strip and status bar sit on the raised neutral. */
QMenuBar {{ background: {c.bg_raised}; border-bottom: 1px solid {c.line}; padding: 2px 4px; }}
QMenuBar::item {{ background: transparent; padding: 4px 10px; border-radius: 4px; }}
QMenuBar::item:selected {{ background: {c.bg_tint}; }}
QMenuBar::item:pressed {{ background: {c.bg_pressed}; }}
QMenu {{ background: {c.bg}; border: 1px solid {c.line}; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 26px; border-radius: 4px; }}
QMenu::item:selected {{ background: {c.bg_tint}; color: {c.text}; }}
QMenu::item:disabled {{ color: {c.text_disabled}; }}
QMenu::separator {{ height: 1px; background: {c.line}; margin: 4px 8px; }}
QMenu::indicator {{ width: 14px; height: 14px; left: 6px; }}

QStatusBar {{ background: {c.bg_raised}; border-top: 1px solid {c.line}; color: {c.text_muted}; }}
QStatusBar::item {{ border: none; }}

QMainWindow > QWidget#centralwidget {{ background: {c.bg_raised}; }}
QTabWidget#mainTabs::pane {{
    border: 1px solid {c.line}; border-radius: 0; top: -1px; background: {c.bg};
}}
QTabWidget#mainTabs > QTabBar {{ background: transparent; }}
QTabWidget#mainTabs > QTabBar::tab {{
    background: transparent; color: {c.text_muted}; font-weight: 500;
    border: none; border-bottom: 3px solid transparent;
    padding: 7px 14px 6px 14px; margin-right: 2px;
}}
QTabWidget#mainTabs > QTabBar::tab:hover {{ color: {c.text}; background: {c.bg_pressed}; }}
QTabWidget#mainTabs > QTabBar::tab:selected {{
    color: {c.text}; background: {c.bg}; border-bottom-color: {c.action};
}}
QTabWidget#mainTabs > QTabBar::tab:disabled {{ color: {c.text_disabled}; }}

/* Section tabs (filter pages, dock channel tabs). */
QTabWidget::pane {{
    border: 1px solid {c.line}; border-radius: 4px; top: -1px; background: {c.bg};
}}
QTabBar::tab {{
    background: transparent; color: {c.text_muted}; border: none;
    border-bottom: 2px solid transparent; padding: 5px 12px; margin-right: 2px;
}}
QTabBar::tab:hover {{ color: {c.text}; }}
QTabBar::tab:selected {{ color: {c.text}; border-bottom-color: {c.action}; }}
QTabBar::tab:bottom {{ border-bottom: none; border-top: 2px solid transparent; }}
QTabBar::tab:bottom:selected {{ border-top-color: {c.action}; }}
QTabBar::tab:disabled {{ color: {c.text_disabled}; }}

QDockWidget {{ color: {c.text_muted}; }}
QDockWidget::title {{
    background: {c.bg}; text-align: left;
    padding: 5px 10px; border-bottom: 1px solid {c.line};
}}
QDockWidget::float-button {{
    border: none; border-radius: 4px; padding: 2px; background: transparent;
}}
QDockWidget::float-button:hover {{ background: {c.bg_raised}; }}

QGroupBox {{
    border: 1px solid {c.line}; border-radius: 8px; font-weight: 700;
    margin-top: 12px; padding: 10px 8px 8px 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 10px; padding: 0 4px; color: {c.text};
}}
QGroupBox::title:disabled {{ color: {c.text_disabled}; }}

/* Buttons */
QPushButton, QToolButton {{
    background: {c.bg}; color: {c.text}; border: 1px solid {c.line_strong};
    border-radius: 8px; padding: 3px 12px; min-height: 18px;
}}
QToolButton {{ padding: 3px 6px; }}
QPushButton:hover, QToolButton:hover {{ background: {c.bg_raised}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {c.bg_pressed}; }}
QPushButton:checked, QToolButton:checked {{ background: {c.bg_tint}; border-color: {c.action}; }}
QPushButton:default {{ border: 2px solid {c.focus}; padding: 2px 11px; }}
QPushButton:disabled, QToolButton:disabled {{
    background: {c.bg_raised}; color: {c.text_disabled}; border-color: {c.line};
}}
QPushButton:focus, QToolButton:focus {{ outline: none; }}
QPushButton::menu-indicator {{ subcontrol-position: right center; right: 6px; }}
{_role_button_rules(c)}
/* Inputs */
QLineEdit, QAbstractSpinBox, QComboBox, QDateTimeEdit {{
    background: {c.bg}; color: {c.text}; border: 1px solid {c.line_strong};
    border-radius: 4px; padding: 2px 6px; min-height: 18px;
    selection-background-color: {c.action}; selection-color: {c.on_action};
}}
QLineEdit:hover, QAbstractSpinBox:hover, QComboBox:hover {{ border-color: {c.text_muted}; }}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus {{
    border: 2px solid {c.focus}; padding: 1px 5px;
}}
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {{
    background: {c.bg_raised}; color: {c.text_disabled}; border-color: {c.line};
}}
QLineEdit[readOnly="true"] {{ background: {c.bg_raised}; }}
QComboBox {{ padding-right: 20px; }}
QComboBox:focus {{ padding-right: 19px; }}
/* Qt sizes the popup from row heights plus its frame and ignores padding on
   the view itself: padding there clips the last row behind a scroll arrow.
   Space the rows through ::item, which row heights do include. */
QComboBox {{ combobox-popup: 0; }}
QComboBox QAbstractItemView {{
    background: {c.bg}; color: {c.text}; border: 1px solid {c.line}; outline: 0;
    selection-background-color: {c.bg_tint}; selection-color: {c.text};
}}
QComboBox QAbstractItemView::item {{ padding: 3px 6px; }}
QComboBox QAbstractItemView::item:selected {{ background: {c.bg_tint}; color: {c.text}; }}
{arrows}
QTextEdit, QPlainTextEdit, QTextBrowser {{
    background: {c.bg}; color: {c.text}; border: 1px solid {c.line}; border-radius: 4px;
    selection-background-color: {c.action}; selection-color: {c.on_action};
}}

QCheckBox, QRadioButton {{ spacing: 6px; }}
QCheckBox:disabled, QRadioButton:disabled {{ color: {c.text_disabled}; }}
QCheckBox::indicator, QGroupBox::indicator, QRadioButton::indicator {{
    width: 12px; height: 12px; background: {c.bg}; border: 1px solid {c.line_strong};
}}
QCheckBox::indicator, QGroupBox::indicator {{ border-radius: 3px; }}
QRadioButton::indicator {{ border-radius: 7px; }}
QCheckBox::indicator:hover, QGroupBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {c.focus};
}}
QCheckBox::indicator:checked, QGroupBox::indicator:checked {{
    background: {c.action}; border-color: {c.action};
}}
QRadioButton::indicator:checked {{
    background: {c.on_action}; border: 4px solid {c.action}; width: 6px; height: 6px;
}}
QCheckBox::indicator:disabled, QGroupBox::indicator:disabled,
QRadioButton::indicator:disabled {{
    background: {c.bg_raised}; border-color: {c.line};
}}
QCheckBox::indicator:checked:disabled, QGroupBox::indicator:checked:disabled {{
    background: {c.line_strong}; border-color: {c.line_strong};
}}
QRadioButton::indicator:checked:disabled {{
    background: {c.bg_raised}; border-color: {c.line_strong};
}}

QSlider::groove:horizontal {{ height: 4px; background: {c.line}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {c.action}; border-radius: 2px; }}
QSlider::sub-page:horizontal:disabled {{ background: {c.line_strong}; }}
QSlider::handle:horizontal {{
    background: {c.bg}; border: 1px solid {c.line_strong};
    width: 12px; height: 12px; margin: -5px 0; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ border-color: {c.focus}; }}
QSlider::handle:horizontal:disabled {{ background: {c.bg_raised}; border-color: {c.line}; }}
QSlider::groove:vertical {{ width: 4px; background: {c.line}; border-radius: 2px; }}
QSlider::add-page:vertical {{ background: {c.action}; border-radius: 2px; }}
QSlider::handle:vertical {{
    background: {c.bg}; border: 1px solid {c.line_strong};
    width: 12px; height: 12px; margin: 0 -5px; border-radius: 7px;
}}

/* Tables and lists */
QTableView, QTableWidget, QTreeView, QListView {{
    background: {c.bg}; color: {c.text}; alternate-background-color: {c.bg_raised};
    border: 1px solid {c.line}; border-radius: 4px; gridline-color: {c.line};
    selection-background-color: {c.bg_tint}; selection-color: {c.text};
}}
QHeaderView::section {{
    background: {c.bg_raised}; color: {c.text_muted}; font-weight: 500;
    border: none; border-right: 1px solid {c.line}; border-bottom: 1px solid {c.line};
    padding: 3px 4px;
}}
QTableCornerButton::section {{ background: {c.bg_raised}; border: none; }}

QProgressBar {{
    background: {c.bg_raised}; border: 1px solid {c.line}; border-radius: 4px;
    text-align: center; color: {c.text}; min-height: 14px;
}}
QProgressBar::chunk {{ background: {BRAND_CYAN}; border-radius: 3px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: {c.line}; border-radius: 3px; min-height: 24px; margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {c.line}; border-radius: 3px; min-width: 24px; margin: 2px;
}}
QScrollBar::handle:hover {{ background: {c.line_strong}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QSplitter::handle {{ background: {c.bg_raised}; }}
QSplitter::handle:hover {{ background: {c.line}; }}
"""


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG 2 contrast ratio between two colours."""

    def luminance(value: str) -> float:
        color = QColor(value)
        channels = [
            part / 12.92 if part <= 0.04045 else ((part + 0.055) / 1.055) ** 2.4
            for part in (color.redF(), color.greenF(), color.blueF())
        ]
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    lighter, darker = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def legible_text(color: str, theme: ThemeColors | None = None, minimum: float = 4.5) -> str:
    """``color`` moved toward the theme's text colour until readable as text.

    Keeps a series colour recognisable (same hue family) where it labels text,
    such as a table row matching a plot line. Classic returns it unchanged.
    """
    c = theme or colors()
    if c.mode is ThemeMode.CLASSIC or contrast_ratio(color, c.bg) >= minimum:
        return QColor(color).name()
    source, target = QColor(color), QColor(c.text)
    for step in range(1, 21):
        weight = step / 20
        mixed = QColor.fromRgbF(
            source.redF() + (target.redF() - source.redF()) * weight,
            source.greenF() + (target.greenF() - source.greenF()) * weight,
            source.blueF() + (target.blueF() - source.blueF()) * weight,
        )
        if contrast_ratio(mixed.name(), c.bg) >= minimum:
            return mixed.name()
    return c.text


def region_style(theme: ThemeColors | None = None) -> dict[str, Any]:
    """Keyword arguments styling a selection region (ROI, fit range).

    Replaces pyqtgraph's default pure-blue fill with a translucent brand-cyan
    one and a solid edge in the theme's accent. Classic keeps the defaults.
    """
    import pyqtgraph as pg

    c = theme or colors()
    if c.mode is ThemeMode.CLASSIC:
        return {}
    cyan = QColor(BRAND_CYAN)
    edge = c.text_accent
    return {
        "brush": pg.mkBrush(cyan.red(), cyan.green(), cyan.blue(), 40),
        "hoverBrush": pg.mkBrush(cyan.red(), cyan.green(), cyan.blue(), 75),
        "pen": pg.mkPen(edge, width=1.5),
        "hoverPen": pg.mkPen(edge, width=2.5),
    }


def persistence_colormap(theme: ThemeColors | None = None) -> Any:
    """Scope Persistence colour map for a theme (the active one by default)."""
    import pyqtgraph as pg

    stops = (theme or colors()).persistence_stops
    if not stops:
        return pg.colormap.get("viridis")
    positions, values = zip(*stops, strict=True)
    return pg.ColorMap(list(positions), [QColor(value) for value in values])


def configure_plots(theme: ThemeColors) -> None:
    """Default pyqtgraph canvas and axis colours; call before creating plots."""
    import pyqtgraph as pg

    pg.setConfigOptions(background=theme.plot_background, foreground=theme.plot_foreground)


def apply_application_theme(app: QApplication, mode: ThemeMode | None = None) -> ThemeColors:
    """Activate a theme for this process; call once before building windows.

    Classic keeps the platform style, palette and font and only colours the
    Start/Stop roles. The EWT themes use Fusion, their palette, Roboto and the
    full stylesheet.
    """
    global _active
    theme = THEMES[mode or startup_theme_mode()]
    _active = theme
    if theme.mode is ThemeMode.CLASSIC:
        app.setStyleSheet(classic_stylesheet())
        return theme
    app.setStyle("Fusion")
    app.setPalette(application_palette(theme))
    if register_fonts():
        app.setFont(application_font())
    app.setStyleSheet(application_stylesheet(theme, svg_icons=_svg_images_supported()))
    configure_plots(theme)
    return theme


def brand_splash_pixmap(
    width: int = 520, height: int = 320, theme: ThemeColors | None = None,
) -> QPixmap:
    """The EWT lockup and product name on the theme's splash ground."""
    style = splash_style(theme)
    register_fonts()  # Also registers the compiled resources holding the lockup.
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(style.background))
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        lockup_height = 170.0
        try:
            from PySide6.QtSvg import QSvgRenderer
        except ImportError:  # pragma: no cover - QtSvg ships with PySide6
            renderer = None
        else:
            renderer = QSvgRenderer(style.lockup_resource)
        if renderer is not None and renderer.isValid():
            size = renderer.defaultSize()
            aspect = size.width() / size.height() if size.height() else 1.0
            lockup_width = lockup_height * aspect
            renderer.render(
                painter,
                QRectF((width - lockup_width) / 2, 36, lockup_width, lockup_height),
            )
        title_font = application_font()
        title_font.setPointSizeF(14)
        title_font.setWeight(QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(style.title))
        painter.drawText(
            QRectF(0, 36 + lockup_height + 14, width, 28),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            "Nuclear Lab Digitizer",
        )
        painter.setPen(QColor(style.rule))
        painter.drawLine(0, height - 1, width, height - 1)
        painter.fillRect(0, 0, width, 4, QColor(BRAND_CYAN))
    finally:
        painter.end()
    return pixmap
