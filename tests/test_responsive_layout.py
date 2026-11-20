from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import cast

import pytest
from PySide6.QtCore import QRect, QSettings
from PySide6.QtGui import QScreen
from PySide6.QtWidgets import QBoxLayout, QScrollArea, QWidget
from pytestqt.qtbot import QtBot

from nlab.app import MainAppWindow
from nlab.controllers.mca_controller import CoincidenceRole, MCAController
from nlab.hardware.digitizer.offline import create_offline_digitizer
from nlab.ui.ui_external_device_view import Ui_ExternalDeviceView
from nlab.ui.ui_global_view import Ui_GlobalView
from nlab.ui.ui_mca_view import Ui_MCAView
from nlab.ui.ui_psd_view import Ui_PSDView
from nlab.ui.ui_psu_view import Ui_PSUView
from nlab.ui.ui_scope_view import Ui_ScopeView
from nlab.views.responsive_layout import (
    MCAResponsiveLayout,
    configure_external_layout,
    configure_global_layout,
    configure_mca_layout,
    configure_psd_layout,
    configure_psu_layout,
    configure_scope_layout,
)

ConfigureLayout = Callable[[QWidget, object], None]
configure_mca_720p = partial(configure_mca_layout, screen_height=720)


class _Screen:
    def __init__(self, width: int, height: int) -> None:
        self._geometry = QRect(0, 0, width, height)

    def availableGeometry(self) -> QRect:  # noqa: N802 - mirrors Qt's API
        return self._geometry


class _Window:
    def __init__(self, width: int, height: int) -> None:
        self._screen = _Screen(width, height)
        self.size: tuple[int, int] | None = None

    def screen(self) -> _Screen:
        return self._screen

    def resize(self, width: int, height: int) -> None:
        self.size = width, height

    def isMaximized(self) -> bool:  # noqa: N802 - mirrors Qt's API
        return False

    def isFullScreen(self) -> bool:  # noqa: N802 - mirrors Qt's API
        return False

    def _resize_for_available_screen(self, screen: _Screen | None = None) -> None:
        MainAppWindow._resize_for_available_screen(
            cast(MainAppWindow, self),
            cast(QScreen | None, screen),
        )

    def _resize_after_screen_layout_change(self) -> None:
        self._resize_for_available_screen()


@pytest.mark.parametrize(
    ("available", "expected"),
    (
        ((1280, 680), (1248, 648)),
        ((1920, 1040), (1280, 800)),
    ),
)
def test_initial_window_size_respects_available_desktop(
    available: tuple[int, int],
    expected: tuple[int, int],
) -> None:
    window = _Window(*available)
    MainAppWindow._resize_for_available_screen(cast(MainAppWindow, window))
    assert window.size == expected


def test_window_resizes_when_moved_to_smaller_screen() -> None:
    window = _Window(2560, 1400)
    destination = _Screen(1280, 680)

    MainAppWindow._on_screen_changed(
        cast(MainAppWindow, window),
        cast(QScreen, destination),
    )

    assert window.size == (1248, 648)


@pytest.mark.parametrize(
    ("ui_type", "configure"),
    (
        (Ui_ScopeView, configure_scope_layout),
        (Ui_MCAView, configure_mca_720p),
        (Ui_PSDView, configure_psd_layout),
        (Ui_PSUView, configure_psu_layout),
        (Ui_GlobalView, configure_global_layout),
        (Ui_ExternalDeviceView, configure_external_layout),
    ),
)
def test_primary_views_fit_inside_720p_content_area(
    qtbot: QtBot,
    ui_type: type[object],
    configure: ConfigureLayout,
) -> None:
    view = QWidget()
    qtbot.addWidget(view)
    ui = ui_type()
    ui.setupUi(view)  # type: ignore[attr-defined]
    configure(view, ui)

    minimum = view.minimumSizeHint()
    assert minimum.width() <= 1024
    assert minimum.height() <= 560

    view.resize(1180, 560)
    assert view.layout() is not None
    view.layout().activate()
    assert view.size().width() == 1180
    assert view.size().height() == 560


@pytest.mark.parametrize(
    "name",
    (
        "scopeControlScroll",
        "mcaControlScroll",
        "psuControlScroll",
        "globalContentScroll",
        "externalControlScroll",
    ),
)
def test_dense_control_regions_are_scrollable(
    qtbot: QtBot,
    name: str,
) -> None:
    view = QWidget()
    qtbot.addWidget(view)
    forms = {
        "scopeControlScroll": (Ui_ScopeView, configure_scope_layout),
        "mcaControlScroll": (Ui_MCAView, configure_mca_720p),
        "psuControlScroll": (Ui_PSUView, configure_psu_layout),
        "globalContentScroll": (Ui_GlobalView, configure_global_layout),
        "externalControlScroll": (Ui_ExternalDeviceView, configure_external_layout),
    }
    ui_type, configure = forms[name]
    ui = ui_type()
    ui.setupUi(view)
    configure(view, ui)

    scroll = view.findChild(QScrollArea, name)
    assert scroll is not None
    assert scroll.widgetResizable()


def test_global_controls_use_available_width_in_two_columns(qtbot: QtBot) -> None:
    view = QWidget()
    qtbot.addWidget(view)
    ui = Ui_GlobalView()
    ui.setupUi(view)

    configure_global_layout(view, ui)

    scroll = view.findChild(QScrollArea, "globalContentScroll")
    assert scroll is not None
    content_layout = scroll.widget().layout()
    assert content_layout is not None
    panel_columns = content_layout.itemAt(0).layout()
    assert isinstance(panel_columns, QBoxLayout)
    assert panel_columns.objectName() == "globalPanelColumns"
    assert panel_columns.direction() == QBoxLayout.Direction.LeftToRight
    # Natural-width columns packed left; the trailing stretch takes spare width.
    assert panel_columns.stretch(0) == panel_columns.stretch(1) == 0
    assert panel_columns.itemAt(2).spacerItem() is not None
    assert panel_columns.stretch(2) == 1

    control_column = panel_columns.itemAt(0).layout()
    assert isinstance(control_column, QBoxLayout)
    assert control_column.indexOf(ui.groupSoftwareStart) == 0
    assert control_column.indexOf(ui.groupTemperatureCorrection) == 1
    assert panel_columns.indexOf(ui.groupDiagnostics) == 1
    assert ui.layoutSyncButtons.direction() == QBoxLayout.Direction.TopToBottom
    sync_actions = ui.layoutSyncButtons.itemAt(1).layout()
    assert isinstance(sync_actions, QBoxLayout)
    assert sync_actions.indexOf(ui.btnSoftwareStart) == 0
    assert sync_actions.indexOf(ui.btnSoftwareReset) == 1
    assert ui.labelSoftwareStartHelp.wordWrap()
    assert ui.labelSoftwareStartHelp.maximumWidth() == 520
    assert ui.btnPrepareSoftware.maximumWidth() == 240
    assert ui.btnSoftwareStart.maximumWidth() == 260
    assert ui.btnSoftwareReset.maximumWidth() == 240
    assert all(
        group.maximumWidth() > 720
        for group in (
            ui.groupSoftwareStart,
            ui.groupTemperatureCorrection,
            ui.groupDiagnostics,
        )
    )


def test_psu_controls_stack_in_one_column_beside_the_plot(qtbot: QtBot) -> None:
    view = QWidget()
    qtbot.addWidget(view)
    ui = Ui_PSUView()
    ui.setupUi(view)

    configure_psu_layout(view, ui)
    view.resize(1180, 560)
    view.show()
    assert view.layout() is not None
    view.layout().activate()

    scroll = view.findChild(QScrollArea, "psuControlScroll")
    assert scroll is not None
    assert ui.horizontalLayout.direction() == QBoxLayout.Direction.LeftToRight
    order = (
        ui.groupSipm,
        ui.groupSipmCompens,
        ui.groupHv,
        ui.groupHvCompens,
        ui.groupTemp,
        ui.groupMonitoring,
    )
    assert [ui.controlLayout.indexOf(group) for group in order] == list(range(len(order)))
    assert scroll.geometry().right() < ui.plotHvVoltage.geometry().left()
    assert 300 <= scroll.width() <= 460
    assert ui.plotHvVoltage.width() > scroll.width()
    assert scroll.horizontalScrollBar().maximum() == 0


def test_mca_header_groups_move_into_scrollable_controls(qtbot: QtBot) -> None:
    view = QWidget()
    qtbot.addWidget(view)
    ui = Ui_MCAView()
    ui.setupUi(view)
    configure_mca_layout(view, ui, screen_height=720)

    assert ui.layoutMca.direction() == QBoxLayout.Direction.TopToBottom
    assert ui.layoutMca.count() == 4
    assert ui.layoutMeasurement.direction() == QBoxLayout.Direction.TopToBottom
    assert ui.layoutMeasurement.count() == 3
    assert ui.controlLayout.indexOf(ui.groupMca) == 0
    assert ui.controlLayout.indexOf(ui.groupMeasurement) == 1
    assert ui.controlLayout.indexOf(ui.groupSignal) == 2
    assert all(
        ui.mainLayout.itemAt(index).layout() is not ui.topRow
        for index in range(ui.mainLayout.count())
    )

    view.resize(1000, 560)
    assert view.layout() is not None
    view.layout().activate()

    scroll = view.findChild(QScrollArea, "mcaControlScroll")
    assert scroll is not None
    assert ui.plotDebug.geometry().y() == scroll.geometry().y()
    assert ui.plotDebug.height() >= 100
    assert ui.plotHistogram.height() >= 140


def test_mca_restores_original_layout_after_returning_to_1080p(qtbot: QtBot) -> None:
    view = QWidget()
    qtbot.addWidget(view)
    ui = Ui_MCAView()
    ui.setupUi(view)
    responsive = configure_mca_layout(view, ui, screen_height=720)
    responsive.apply_screen_height(1080)

    assert ui.layoutMca.direction() == QBoxLayout.Direction.LeftToRight
    assert ui.layoutMca.count() == 7
    assert ui.layoutMeasurement.direction() == QBoxLayout.Direction.LeftToRight
    assert ui.layoutMeasurement.count() == 5
    assert ui.mainLayout.itemAt(0).layout() is ui.topRow
    assert ui.topRow.indexOf(ui.groupMca) == 0
    assert ui.topRow.indexOf(ui.groupMeasurement) == 1
    # The wide layout scrolls its control column too, at the column's fixed width.
    scroll = view.findChild(QScrollArea, "mcaControlScroll")
    assert scroll is not None
    assert scroll.widget() is ui.controlPanel
    assert scroll.minimumWidth() == scroll.maximumWidth() > ui.controlPanel.maximumWidth()

    responsive.apply_screen_height(720)
    assert ui.controlLayout.indexOf(ui.groupMca) == 0
    assert ui.controlLayout.indexOf(ui.groupMeasurement) == 1
    assert view.findChild(QScrollArea, "mcaControlScroll") is scroll
    assert scroll.widget() is ui.controlPanel


@pytest.mark.parametrize(
    ("available", "compact"),
    (
        ((1536, 816), True),  # 1920x1080 at 125%: 1080 physical lines, 816 px usable
        ((1920, 1032), False),  # 1920x1080 at 100%
        ((1280, 984), True),  # 1280x1024: too narrow for the header row
        ((1280, 672), True),  # 1280x720
        ((2048, 1104), False),  # 2560x1440 at 125%
    ),
)
def test_mca_layout_follows_the_device_independent_screen_area(
    qtbot: QtBot, available: tuple[int, int], compact: bool,
) -> None:
    view = QWidget()
    qtbot.addWidget(view)
    ui = Ui_MCAView()
    ui.setupUi(view)
    responsive = configure_mca_layout(view, ui, screen_height=720 if not compact else 1080)

    responsive._apply_screen(cast(QScreen, _Screen(*available)))

    in_header_row = ui.mainLayout.itemAt(0).layout() is ui.topRow
    assert in_header_row is not compact
    assert ui.controlLayout.indexOf(ui.groupMca) == (0 if compact else -1)


def test_mca_wide_layout_height_does_not_grow_with_the_control_column(qtbot: QtBot) -> None:
    view = QWidget()
    qtbot.addWidget(view)
    ui = Ui_MCAView()
    ui.setupUi(view)
    configure_mca_layout(view, ui, screen_height=1080)

    # The column alone is taller than a 1080p work area once window chrome is added.
    assert ui.controlPanel.minimumSizeHint().height() > view.minimumSizeHint().height()
    assert view.minimumSizeHint().height() <= 560


@pytest.fixture
def isolated_qsettings(tmp_path: Path) -> Iterator[None]:
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "ini"))
    yield
    QSettings.setDefaultFormat(QSettings.Format.NativeFormat)


@pytest.mark.parametrize(
    ("available", "maximized_client", "check_width"),
    (
        # 1920x1080 at 125% (the Windows default on many laptops): compact MCA.
        ((1536, 816), (1536, 786), True),
        # 1920x1080 at 100%: wide MCA. The offscreen test platform draws text as
        # boxes much wider than real glyphs (its header row is ~1950 px here, about
        # 1390 px with Windows fonts), so only the height is meaningful.
        ((1920, 1032), (1920, 1001), False),
    ),
)
def test_two_channel_window_fits_a_maximized_full_hd_screen(
    qtbot: QtBot,
    isolated_qsettings: None,
    monkeypatch: pytest.MonkeyPatch,
    available: tuple[int, int],
    maximized_client: tuple[int, int],
    check_width: bool,
) -> None:
    """The largest tab sets the window minimum; it must fit the work area.

    A larger minimum pushed the window past the screen edges, and a maximized
    window then squeezed the channel docks so their tab bar covered controls.
    """
    monkeypatch.setattr(
        MCAResponsiveLayout,
        "_apply_screen",
        lambda self, _screen: self.apply_screen_height(available[1], available[0]),
    )
    window = MainAppWindow(channels=2, offline=True)
    qtbot.addWidget(window)
    window.resize(*maximized_client)
    window.show()
    tabs = window.ui.mainTabs
    for index in range(tabs.count()):  # hidden pages update their minimum when shown
        tabs.setCurrentIndex(index)
        qtbot.wait(10)

    minimum = window.minimumSizeHint()
    assert minimum.height() <= maximized_client[1]
    assert window.height() == maximized_client[1]
    if check_width:
        assert minimum.width() <= maximized_client[0]
        assert window.width() == maximized_client[0]
    assert window.findChild(QScrollArea, "coincidenceControlScroll") is not None


def test_mca_dma_checkbox_stays_above_file_button_across_layout_changes(qtbot: QtBot) -> None:
    view = QWidget()
    qtbot.addWidget(view)
    ui = Ui_MCAView()
    ui.setupUi(view)
    responsive = configure_mca_layout(view, ui, screen_height=1080)
    checkbox = ui.cbDmaEnable
    checkbox.setChecked(True)
    view.show()
    for height in (1080, 720, 1080, 720):
        responsive.apply_screen_height(height)
        view.resize(1180, 650)
        assert view.layout() is not None
        view.layout().activate()
        qtbot.wait(10)
        assert ui.cbDmaEnable is checkbox and checkbox.isChecked()
        assert ui.groupMeasurement.isAncestorOf(checkbox)
        assert not ui.groupMca.isAncestorOf(checkbox)
        assert ui.vDmaFile.indexOf(checkbox) == 0
        assert ui.vDmaFile.indexOf(ui.btnDmaFile) == 1
        assert checkbox.geometry().bottom() < ui.btnDmaFile.geometry().top()


def test_mca_relocated_dma_checkbox_retains_configuration_and_interlocks(qtbot: QtBot) -> None:
    view = MCAController(create_offline_digitizer(0).mca)
    qtbot.addWidget(view)
    view.ui.cbDmaEnable.setChecked(True)
    assert view.configuration_settings()["dma_enabled"] is True
    view._set_controls_enabled(False)
    assert view.ui.groupMeasurement.isEnabled()
    assert not view.ui.cbDmaEnable.isEnabled()
    view._set_controls_enabled(True)
    assert view.ui.cbDmaEnable.isEnabled()
    view._coincidence_role = CoincidenceRole.ACTIVE
    view._set_controls_enabled(True)
    assert not view.ui.cbDmaEnable.isEnabled()
    view._coincidence_role = CoincidenceRole.NONE
    view.set_replay_hardware_interlock(True)
    assert not view.ui.cbDmaEnable.isEnabled()
    view.set_replay_hardware_interlock(False)
    assert view.ui.cbDmaEnable.isEnabled()
    assert view.ui.cbDmaEnable.isChecked()


def test_psd_controls_reflow_without_changing_widgets(
    qtbot: QtBot,
) -> None:
    view = QWidget()
    qtbot.addWidget(view)
    ui = Ui_PSDView()
    ui.setupUi(view)
    configure_psd_layout(view, ui)

    ratio_max_index = ui.controlsLayout.indexOf(ui.spinRatioMax)
    cut_index = ui.controlsLayout.indexOf(ui.spinCut)
    assert ui.controlsLayout.getItemPosition(ratio_max_index)[:2] == (2, 1)
    assert ui.controlsLayout.getItemPosition(cut_index)[:2] == (2, 3)
    scale_index = ui.controlsLayout.indexOf(ui.comboMatrixScale)
    assert ui.controlsLayout.getItemPosition(scale_index)[:2] == (1, 4)
    clear_index = ui.controlsLayout.indexOf(ui.btnClear)
    assert ui.controlsLayout.getItemPosition(clear_index) == (0, 5, 3, 1)
