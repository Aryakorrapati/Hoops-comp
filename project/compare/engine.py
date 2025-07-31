"""
engine.py – compatibility wrapper
Runs your existing standalone script unchanged, captures its outputs,
and returns base-64 images + the text report.

Place your current monolithic script (unchanged) at:
    project/compare/legacy_script.py

Ensure the script saves:
    input_vs_comp_radar.png
    nba_floor_comp_ceiling_radar.png
    nba_comp_vs_pred_radar.png
    comp_output.txt
"""

from __future__ import annotations

import os
import sys
import runpy
import base64
import builtins
from pathlib import Path

# Use a headless backend for servers
import matplotlib
matplotlib.use("Agg")

# Path to your untouched script
MONOLITH_PATH = Path(__file__).with_name("legacy_script.py")

# Filenames your script already writes
PNG_FILES = [
    "input_vs_comp_radar.png",
    "nba_floor_comp_ceiling_radar.png",
    "nba_comp_vs_pred_radar.png",
]
TXT_FILE = "comp_output.txt"


def _b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def run_all(cbb_url: str) -> dict[str, list[str] | str]:
    """
    Execute the legacy script with `input()` pre-filled as `cbb_url`,
    then return {"images":[...3 b64 pngs...], "text": report}.
    """
    if not MONOLITH_PATH.exists():
        raise ValueError(
            f"Expected monolith at {MONOLITH_PATH}. "
            "Paste your current big script there as 'legacy_script.py'."
        )

    script_dir = MONOLITH_PATH.parent

    # Run from the script's directory so relative CSV paths work.
    prev_cwd = Path.cwd()
    os.chdir(script_dir)

    # Patch input() so the script reads your URL without prompting.
    old_input = builtins.input
    old_argv = sys.argv
    try:
        builtins.input = lambda prompt="": cbb_url
        # Some scripts look at argv[0]; keep it sensible
        sys.argv = [str(MONOLITH_PATH)]

        try:
            runpy.run_path(str(MONOLITH_PATH), run_name="__main__")
        except SystemExit as e:
            # Your script sometimes calls sys.exit(); that's fine.
            # We'll still try to collect outputs below.
            pass

        # Collect outputs
        imgs: list[str] = []
        for fname in PNG_FILES:
            p = script_dir / fname
            if not p.exists():
                raise ValueError(
                    f"Expected output image '{fname}' not found. "
                    "Did the script finish successfully?"
                )
            imgs.append(_b64(p))

        txt_path = script_dir / TXT_FILE
        if not txt_path.exists():
            raise ValueError(
                f"Expected report '{TXT_FILE}' not found. "
                "Did the script write comp_output.txt?"
            )
        report_str = txt_path.read_text(encoding="utf-8", errors="replace")

        return {"images": imgs, "text": report_str}

    finally:
        # Restore environment
        builtins.input = old_input
        sys.argv = old_argv
        os.chdir(prev_cwd)
