<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/hero-dark.svg">
  <img alt="SRFM Lab: SPY daily closes drawn as a spacetime worldline, with a light cone at every bar, timelike bars in teal, spacelike bars dashed in red, and black-hole wells shaded amber" src="assets/hero-light.svg" width="100%">
</picture>

<p align="center">
  <a href="https://srfm-lab.vercel.app/"><b>Project site</b></a> &nbsp;·&nbsp;
  <a href="#quick-start-one-command">Quick start</a> &nbsp;·&nbsp;
  <a href="#how-it-works">How it works</a> &nbsp;·&nbsp;
  <a href="#examples">Examples</a> &nbsp;·&nbsp;
  <a href="docs/REFERENCE.md">Full reference</a>
  <br><br>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

# SRFM Lab

<!-- outcome-block -->
**A quant research pipeline in one repo: raw ticks to paper trading, plus a tool that hunts for new ideas.**

Most quant repos are one strategy in a notebook. This one is the whole stack: 9 languages, 1.7M lines, covering data, signals, backtests, execution and research.

If it doesn't do what this README says, [open an issue](https://github.com/Mattbusel/srfm-lab/issues) and I'll fix it.
<!-- outcome-block -->

English | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

A Python lab that uses special relativity math on price bars. Each bar is labeled an ordinary move or a jump. Runs of ordinary moves are flagged as "black holes". On top of that signal: backtests, Monte Carlo, paper trading and an idea engine. It is the hub of the SRFM (Special Relativity in Financial Modeling) repos.

It is a research monorepo, not a product. The core runs offline on bundled data. The rest of the lab (Rust, Go, Julia, R, C/C++, Zig, Elixir, TypeScript) needs API keys and services.

> [!IMPORTANT]
> **Research code, not financial advice.** Backtest and paper-trading results are experiments on past or simulated data. They do not show any strategy makes money.

## Quick start: one command

Needs Python 3.12+. No API keys. No network after the install.

```bash
git clone -c core.longpaths=true https://gitlab.com/mattbusel/srfm-lab
cd srfm-lab
python quickstart.py
```

[`quickstart.py`](quickstart.py) makes `.venv/`, installs the pinned core from `requirements-core.txt`, and runs [`examples/bh_quickstart.py`](examples/bh_quickstart.py) on the hourly SPY, QQQ and DIA bars in `tools/data_cache/`. `-c core.longpaths=true` lets the clone finish on Windows, where some files pass the 260-character path limit. There is no installer. The [project site](https://srfm-lab.vercel.app/) shows the same results with nothing to install.

<img alt="Terminal output of the quick start: a table of bars, timelike share, black-hole active share, onsets and forward returns for ES, NQ and YM" src="assets/quickstart-terminal.svg" width="100%">

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/physics-explainer-dark.svg">
  <img alt="Four steps from a price bar to a black hole: 1, beta = absolute return divided by c, the speed limit (0.5% a day for SPY); 2, beta under 1 is timelike, an ordinary move, beta of 1 or more is spacelike, a jump; 3, timelike bars add mass (mass times 0.95 plus absolute return times 100 times a streak boost), spacelike bars cut it by 30% and reset the streak; 4, a black hole forms when mass passes 1.5 after five timelike bars in a row." src="assets/physics-explainer-light.svg" width="100%">
</picture>

Every bar gets a **speed** β: its return against a per-instrument "speed of light" `c`. Slow bars stay inside their **light cone** (timelike, `ds² = c²dt² − dx²` is positive). Fast bars break out (spacelike). Orderly bars build **mass**. Enough mass after five orderly bars in a row makes a **black hole**, which points the way price has been drifting. The **Lorentz factor** γ = 1/√(1 − v²) turns each bar's speed into proper time. All of it is two small classes in [`lib/srfm_core.py`](lib/srfm_core.py): `MinkowskiClassifier` and `BlackHoleDetector`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/bh-formation-dark.svg">
  <img alt="Real SPY daily bars from 2025-12-10 to 2026-01-28: each bar's speed beta against the limit of 1, the count of timelike bars in a row, and black-hole mass. Wells formed on 2025-12-30 and 2026-01-16, each after five timelike bars in a row with mass above 1.5." src="assets/bh-formation-light.svg" width="100%">
</picture>

The chart above is those classes on real bundled bars ([`scripts/figures/make_physics_explainer.py`](scripts/figures/make_physics_explainer.py), made 2026-09-28). The rest of the lab plugs into what they output:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/pipeline-dark.svg">
  <img alt="Pipeline: bars, Minkowski classifier, black-hole detector, signal stack, backtest, paper trader, with the idea engine feeding parameters back" src="assets/pipeline-light.svg" width="100%">
</picture>

Deep dives: [BH physics](docs/bh_physics.md) · [IAE architecture](docs/iae_architecture.md) · [Execution stack](docs/execution_stack.md) · [Monte Carlo](docs/monte_carlo.md)

## Examples

Real output from 2026-09-28, on the bundled data.

**1. The default run** (the table above): ES, NQ and YM (SPY, QQQ, DIA as stand-ins), 2020-02-10 to 2026-04-02. Black holes were active on 1.2% to 2.9% of days. The 5-day return after each onset, signed by the well's direction, is negative on all three. So with these settings the bare signal does not predict the next week.

**2. One instrument, a longer horizon**

```text
$ python quickstart.py --symbols ES --horizon 10
       proxy  bars        from          to  timelike %  BH active %  onsets  all days |ret 10d| %  onset signed ret 10d %  onset hit rate %
symbol
ES       SPY  1542  2020-02-10  2026-04-02       43.45         2.85      23                  2.64                   -0.27             56.52
```

**3. Every well on the chart.** The same run saves `examples/output/bh_quickstart.png`. Here it is redrawn, wells shaded, mass under each panel:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/wells-dark.svg">
  <img alt="SPY, QQQ and DIA daily closes over the last 750 days with black-hole active periods shaded and BH mass plotted under each panel against the formation threshold" src="assets/wells-light.svg" width="100%">
</picture>

## Use it in 3 steps

1. Clone and run `python quickstart.py` (above).
2. Change the question: `--symbols ES,NQ`, `--horizon 10`, or edit `PARAMS` (speed of light `cf`, threshold `bh_form`) in `examples/bh_quickstart.py`.
3. Go further: `pip install -r requirements-dev.txt && pytest tests -q` (what CI runs), then the backtests, Monte Carlo and paper trader in [docs/REFERENCE.md](docs/REFERENCE.md#the-full-lab).

## The SRFM family

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/family-dark.svg">
  <img alt="Map of the SRFM repositories: srfm-paper-impl at the origin of a light cone, the C++ core and srfm-python inside it, and srfm-lab at the top" src="assets/family-light.svg" width="100%">
</picture>

| Repository | What it is |
|---|---|
| [srfm-paper-impl](https://gitlab.com/mattbusel/srfm-paper-impl) | The paper (PDF), scripts and a notebook that redraw its figures, and a small Rust version of the core formulas with no dependencies |
| [Special-Relativity-in-Financial-Modeling](https://gitlab.com/mattbusel/Special-Relativity-in-Financial-Modeling) | C++20 core and Windows/Linux command-line tools. Computes price velocity (beta), Lorentz factor (gamma), spacetime interval labels, Christoffel symbols and geodesic deviation on OHLCV bars |
| [srfm-python](https://gitlab.com/mattbusel/srfm-python) | Pure-Python SDK: a pandas `df.srfm` accessor and a Polars wrapper for the Lorentz-factor pipeline |
| **srfm-lab** (this repo) | The multi-language lab: black-hole (BH) signal, idea automation engine, backtests and paper trading |

The Rust crate [fin-stream](https://gitlab.com/mattbusel/fin-stream) also ships a streaming `lorentz` module built on the same transform.

## Documentation

| Document | What is in it |
|---|---|
| [docs/REFERENCE.md](docs/REFERENCE.md) | Everything else: full lab installs, crypto backtest + Monte Carlo, paper trading, analysis pipeline, IAE services, performance notes, module map, primitives by language, latency figures |
| [docs/bh_physics.md](docs/bh_physics.md) | The black-hole physics in depth |
| [docs/iae_architecture.md](docs/iae_architecture.md) | The Idea Automation Engine |
| [docs/wave4_backtest.md](docs/wave4_backtest.md) · [docs/monte_carlo.md](docs/monte_carlo.md) | Backtest and Monte Carlo engines |
| [docs/runbook.md](docs/runbook.md) | Running the live services |
| [RESULTS.md](RESULTS.md) · [HYPOTHESIS.md](HYPOTHESIS.md) · [JOURNAL.md](JOURNAL.md) | Research results, hypotheses and the lab journal |
| [CONTRIBUTING.md](CONTRIBUTING.md) · [CHANGELOG.md](CHANGELOG.md) | Contributing and version history |

## License

MIT, see [LICENSE](LICENSE). Research code, as is, not financial advice.
