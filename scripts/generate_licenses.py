"""Generate LICENSES_THIRD_PARTY.txt from installed packages.

Requires pip-licenses (included in the [dev] extras):
    python scripts/generate_licenses.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "LICENSES_THIRD_PARTY.txt"
BUNDLED_ASSET_LICENSES = (
    ("Roboto (font)", ROOT / "resources" / "fonts" / "OFL.txt"),
)


def _find_exe(name: str) -> Path:
    scripts_dir = Path(sys.executable).parent
    suffix = ".exe" if sys.platform == "win32" else ""
    exe = scripts_dir / f"{name}{suffix}"
    if not exe.exists():
        raise FileNotFoundError(
            f"{exe} not found — make sure the venv is active and pip-licenses is installed."
        )
    return exe


def main() -> None:
    print("Collecting third-party licenses...")
    exe = _find_exe("pip-licenses")
    subprocess.run(
        [
            str(exe),
            "--format=plain-vertical",
            "--with-license-file",
            "--no-license-path",
            "--output-file", str(OUTPUT),
        ],
        check=True,
    )
    # Assets compiled into the Qt resources are not Python packages, so
    # pip-licenses cannot see them; append their licences explicitly.
    with OUTPUT.open("a", encoding="utf-8") as output:
        for name, license_file in BUNDLED_ASSET_LICENSES:
            output.write(f"\n\n{name}\nBundled asset (Qt resources)\n")
            output.write(license_file.read_text(encoding="utf-8"))
    print(f"Written to {OUTPUT}")


if __name__ == "__main__":
    main()
