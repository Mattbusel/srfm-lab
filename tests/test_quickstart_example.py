"""Smoke test: the README quick start (examples/bh_quickstart.py) runs on bundled data."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "bh_quickstart.py"


def _load():
    spec = importlib.util.spec_from_file_location("bh_quickstart", _EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_quickstart_runs_and_writes_chart(tmp_path, capsys):
    mod = _load()
    out = tmp_path / "chart.png"
    assert mod.main(["--out", str(out)]) == 0
    printed = capsys.readouterr().out
    for sym in ("ES", "NQ", "YM"):
        assert sym in printed
    assert out.exists() and out.stat().st_size > 10_000


def test_quickstart_finds_black_holes_on_daily_data():
    mod = _load()
    close = mod.load_daily_close("ES")
    res = mod.run_bh(close, mod.PARAMS["ES"]["cf"], mod.PARAMS["ES"]["bh_form"])
    assert len(res) > 1000
    assert set(res["bit"].unique()) <= {"UNKNOWN", "TIMELIKE", "SPACELIKE"}
    assert 0 < res["active"].mean() < 0.5
