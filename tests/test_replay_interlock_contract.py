"""Every workspace controller shares one replay-interlock signature."""

from __future__ import annotations

import inspect

import pytest

from nlab.controllers.coincidence_controller import CoincidenceController
from nlab.controllers.current_monitor_controller import CurrentMonitorController
from nlab.controllers.external_device_controller import ExternalDeviceController
from nlab.controllers.global_controller import GlobalController
from nlab.controllers.interlock import ReplayInterlock
from nlab.controllers.mca_controller import MCAController
from nlab.controllers.psu_controller import PSUController
from nlab.controllers.scope_controller import ScopeController

CONTROLLERS = (
    CoincidenceController, CurrentMonitorController, ExternalDeviceController,
    GlobalController, MCAController, PSUController, ScopeController,
)


@pytest.mark.parametrize("controller", CONTROLLERS, ids=lambda cls: cls.__name__)
def test_controller_implements_the_replay_interlock_contract(controller: type) -> None:
    expected = inspect.signature(ReplayInterlock.set_replay_hardware_interlock)
    actual = inspect.signature(controller.set_replay_hardware_interlock)
    assert list(actual.parameters) == list(expected.parameters)
    option = actual.parameters["resume_background"]
    assert option.kind is inspect.Parameter.KEYWORD_ONLY and option.default is True
