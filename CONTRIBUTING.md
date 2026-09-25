# Contributing to srfm-lab

Thanks for looking. This is a large research monorepo; the Python core is the part that is tested and the easiest place to start.

## Setup

Python 3.12+:

```bash
git clone --filter=blob:none https://github.com/Mattbusel/srfm-lab
cd srfm-lab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Before you open a PR

Run the same checks as CI:

```bash
ruff check --isolated --target-version py311 --select E9,F63,F7,F82 lib spacetime/engine ml execution tests examples
pytest tests -q
python examples/bh_quickstart.py --no-plot
```

- Add a test in `tests/` for any bug you fix.
- Keep changes small and focused; this repo mixes many experiments, so avoid drive-by reformatting.
- `lib/` is put first on `sys.path` by `tests/conftest.py`. Do not add a package directory under `lib/` with the same name as a module next to it or a top-level package (for example `lib/regime/` next to `lib/regime.py`): the package silently shadows the module.
- Do not commit API keys, `.env` files or live trading databases.
- No performance claims in docs or PRs unless a script in the repo reproduces them.

## Where to start

Issues labelled [good first issue](https://github.com/Mattbusel/srfm-lab/labels/good%20first%20issue) are small and self-contained.

## Other languages

Rust (`crates/`, `extensions/`), Go (`idea-engine/`, `market-data/`), Julia, R, C/C++, Zig, Elixir and TypeScript components are not covered by CI yet. If you work on one, say in the PR which commands you ran to build and test it.
