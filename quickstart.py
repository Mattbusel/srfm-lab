"""One command from a fresh copy of srfm-lab:

    python quickstart.py

It makes a virtual environment in .venv (first run only), installs the pinned
key-free core from requirements-core.txt into it, and runs
examples/bh_quickstart.py on the bars bundled in tools/data_cache/.
No API keys, and no network after the install.

Anything after the script name goes to the example, for instance:

    python quickstart.py --symbols ES --horizon 10

Research code, not financial advice.
"""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQS = ROOT / "requirements-core.txt"
EXAMPLE = ROOT / "examples" / "bh_quickstart.py"
STAMP = VENV / ".srfm-core-installed"


def venv_python() -> Path:
    sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return VENV / sub


def main(argv: list[str]) -> int:
    if argv[:1] in (["-h"], ["--help"]):
        print(__doc__)
        return 0
    if sys.version_info < (3, 12):
        print(f"srfm-lab needs Python 3.12 or newer; this is {sys.version.split()[0]}.\n"
              "Install 3.12+ from https://www.python.org/downloads/ and run it with that, "
              "e.g. `py -3.12 quickstart.py` on Windows or `python3.12 quickstart.py`.",
              file=sys.stderr)
        return 1

    py = venv_python()
    if not py.exists():
        print(f"[1/3] creating {VENV.name}/ ...", flush=True)
        venv.EnvBuilder(with_pip=True).create(VENV)
    else:
        print(f"[1/3] using existing {VENV.name}/", flush=True)

    if not STAMP.exists() or STAMP.read_text() != REQS.read_text():
        print("[2/3] installing the core requirements (numpy, pandas, scipy, matplotlib, ...) ...",
              flush=True)
        rc = subprocess.call([str(py), "-m", "pip", "install", "--disable-pip-version-check",
                              "-q", "-r", str(REQS)])
        if rc != 0:
            print("pip could not install requirements-core.txt. Check your network connection, "
                  f"or delete {VENV.name}/ and try again.", file=sys.stderr)
            return rc
        STAMP.write_text(REQS.read_text())
    else:
        print("[2/3] core requirements already installed", flush=True)

    print("[3/3] running examples/bh_quickstart.py on the bundled data\n", flush=True)
    return subprocess.call([str(py), str(EXAMPLE), *argv], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
