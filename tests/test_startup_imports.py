"""Keep heavy analysis/report libraries out of application start-up.

Measured with ``python -X importtime -c "import nlab.app"``: lmfit (with its
matplotlib/pandas imports), scipy, xmlschema and the matplotlib/reportlab
report stack added ~1.5 s before the main window could appear. They are only
needed when a fit, N42 validation or report export runs, so they are imported
inside those functions. A new top-level import would silently undo that.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

DEFERRED = ("lmfit", "scipy", "matplotlib", "reportlab", "xmlschema", "uproot")


def test_importing_the_application_does_not_load_heavy_analysis_libraries() -> None:
    probe = (
        "import json, sys\n"
        "import nlab.app  # noqa: F401\n"
        f"print(json.dumps(sorted(m for m in {DEFERRED!r} if m in sys.modules)))\n"
    )
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        check=True,
        env=env,
        text=True,
        timeout=120,
    )

    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded == [], f"imported at start-up: {loaded}"
