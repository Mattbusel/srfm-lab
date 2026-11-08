# SRFM Lab

English | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Quant research lab that runs special relativity math on prices.

9 languages, 1.7M lines.

- Labels each price bar a normal move or a jump
- Flags "black holes" in runs of normal moves
- Backtests, runs Monte Carlo, paper trades
- Hunts for new ideas

Research code, not financial advice.

## Install

```bash
git clone -c core.longpaths=true https://gitlab.com/mattbusel/srfm-lab
```

## Use

Needs Python 3.12+. No API keys.

```bash
cd srfm-lab
python quickstart.py --symbols ES --horizon 10
```

[Project site](https://srfm-lab.vercel.app/) · [Reference](docs/REFERENCE.md)

More: [docs/DETAILS.md](docs/DETAILS.md)

License: MIT
