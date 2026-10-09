<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/hero-dark.svg">
  <img alt="SRFM Lab：把 SPY 日收盘价画成一条时空世界线，每根 K 线都带一个光锥，类时 K 线为青绿色，类空 K 线为红色虚线，黑洞势阱以琥珀色阴影标出" src="assets/hero-light.svg" width="100%">
</picture>

<p align="center">
  <a href="https://srfm-lab.vercel.app/"><b>项目网站</b></a> &nbsp;·&nbsp;
  <a href="#quick-start-one-command">快速开始</a> &nbsp;·&nbsp;
  <a href="#how-it-works">工作原理</a> &nbsp;·&nbsp;
  <a href="#examples">示例</a> &nbsp;·&nbsp;
  <a href="docs/REFERENCE.md">完整参考</a>
  <br><br>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

# SRFM Lab

[English](README.md) | 简体中文 | [日本語](README.ja.md) | [한국어](README.ko.md)

**一个用 Python 编写的量化研究实验室：用狭义相对论的数学把每根行情 K 线标记为普通波动或跳空，并找出“黑洞”，也就是有序波动不断累积的区间。** 它是 SRFM（Special Relativity in Financial Modeling，金融建模中的狭义相对论）系列项目的核心仓库，围绕这个信号提供回测、蒙特卡洛模拟、模拟盘交易和一个策略想法挖掘引擎。

面向想在真实数据上探索一种受物理学启发的交易信号的量化研究员、学生和爱好者。这是一个研究用的 monorepo，不是打包好的产品：核心部分使用仓库自带的数据即可离线运行，而更大范围的实验室（Rust、Go、Julia、R、C/C++、Zig、Elixir、TypeScript）需要 API 密钥和外部服务。

> [!IMPORTANT]
> **这是研究代码，不构成投资建议。** 回测和模拟盘结果只是在历史数据或模拟数据上的实验，不能证明任何策略能够盈利。

<a id="quick-start-one-command"></a>
## 快速开始：一条命令

需要 Python 3.12+。不需要 API 密钥，安装完成后无需联网。

```bash
git clone -c core.longpaths=true https://gitlab.com/mattbusel/srfm-lab
cd srfm-lab
python quickstart.py
```

[`quickstart.py`](quickstart.py) 会创建 `.venv/`，从 `requirements-core.txt` 安装锁定版本的核心依赖，然后用 `tools/data_cache/` 中自带的 SPY、QQQ 和 DIA 小时级 K 线运行 [`examples/bh_quickstart.py`](examples/bh_quickstart.py)。`-c core.longpaths=true` 让克隆在 Windows 上能够完成，因为仓库里提交的部分构建产物超过了 260 个字符的路径长度限制。没有安装程序，也没有 `.exe`：这是研究代码，托管的[项目网站](https://srfm-lab.vercel.app/)无需安装就能看到同样的结果。

<img alt="快速开始的终端输出：一张表格，列出 ES、NQ 和 YM 的 K 线数量、类时占比、黑洞活跃占比、形成次数和远期收益" src="assets/quickstart-terminal.svg" width="100%">

<a id="how-it-works"></a>
## 工作原理

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/physics-explainer-dark.svg">
  <img alt="从一根 K 线到一个黑洞的四个步骤：1，beta 等于绝对收益除以速度上限 c（SPY 为每天 0.5%）；2，beta 小于 1 为类时，即普通波动，beta 大于等于 1 为类空，即跳空；3，类时 K 线增加质量（质量乘以 0.95，加上绝对收益乘以 100 再乘以连续加成），类空 K 线使质量减少 30% 并重置连续计数；4，连续五根类时 K 线之后质量超过 1.5 时，黑洞形成。" src="assets/physics-explainer-light.svg" width="100%">
</picture>

通俗地说：每根 K 线都有一个**速度** β，即它的收益率相对于每个品种各自的“光速” `c` 的比值。慢的 K 线位于自己的**光锥**之内（类时，`ds² = c²dt² − dx²` 为正）；快的 K 线冲出光锥（类空）。一连串有序的 K 线会积累**质量**，在连续五根有序 K 线之后质量足够大，就会形成一个**黑洞**：一个指向价格此前漂移方向的势阱。**洛伦兹因子** γ = 1/√(1 − v²) 把每根 K 线的速度换算成固有时。这一切都在 [`lib/srfm_core.py`](lib/srfm_core.py) 的两个小类里：`MinkowskiClassifier` 和 `BlackHoleDetector`。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/bh-formation-dark.svg">
  <img alt="2025-12-10 至 2026-01-28 的真实 SPY 日线：每根 K 线相对于上限 1 的速度 beta、连续类时 K 线的数量，以及黑洞质量。势阱分别在 2025-12-30 和 2026-01-16 形成，每次都发生在连续五根类时 K 线且质量高于 1.5 之后。" src="assets/bh-formation-light.svg" width="100%">
</picture>

上图是这两个真实的类在仓库自带的真实 K 线上运行的结果（[`scripts/figures/make_physics_explainer.py`](scripts/figures/make_physics_explainer.py)，2026-09-28 重新生成）。实验室的其余部分都接在这两个类的输出之上：

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/pipeline-dark.svg">
  <img alt="流水线：K 线、闵可夫斯基分类器、黑洞检测器、信号栈、回测、模拟盘交易，策略想法引擎再把参数反馈回去" src="assets/pipeline-light.svg" width="100%">
</picture>

深入阅读：[黑洞物理](docs/bh_physics.md) · [IAE 架构](docs/iae_architecture.md) · [执行栈](docs/execution_stack.md) · [蒙特卡洛](docs/monte_carlo.md)

<a id="examples"></a>
## 示例

2026-09-28 在仓库自带数据上的真实输出。

**1. 默认运行**（即上方终端截图中的表格）：ES、NQ 和 YM（以 SPY、QQQ、DIA 作为替代），2020-02-10 至 2026-04-02。黑洞在 1.2% 到 2.9% 的交易日处于活跃状态。每次黑洞形成后的 5 日收益（按势阱方向取符号）在三个品种上都是负的，这就是如实的结果：在这组参数下，单纯的信号无法预测下一周的走势。

**2. 单个品种，更长的持有期**

```text
$ python quickstart.py --symbols ES --horizon 10
       proxy  bars        from          to  timelike %  BH active %  onsets  all days |ret 10d| %  onset signed ret 10d %  onset hit rate %
symbol
ES       SPY  1542  2020-02-10  2026-04-02       43.45         2.85      23                  2.64                   -0.27             56.52
```

**3. 图上的每一个势阱**。同一次运行会保存 `examples/output/bh_quickstart.png`；下面是重新绘制的版本，势阱以阴影标出，每个面板下方是质量曲线：

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/wells-dark.svg">
  <img alt="SPY、QQQ 和 DIA 最近 750 天的日收盘价，黑洞活跃区间以阴影标出，每个面板下方绘制了黑洞质量及其形成阈值" src="assets/wells-light.svg" width="100%">
</picture>

## 三步上手

1. 克隆仓库并运行 `python quickstart.py`（见上文）。
2. 换一个问题：`--symbols ES,NQ`、`--horizon 10`，或者修改 `examples/bh_quickstart.py` 中的 `PARAMS`（光速 `cf`、阈值 `bh_form`）。
3. 更进一步：`pip install -r requirements-dev.txt && pytest tests -q`（CI 运行的就是这个），然后是 [docs/REFERENCE.md](docs/REFERENCE.md#the-full-lab) 中的回测、蒙特卡洛和模拟盘交易。

## SRFM 系列项目

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/family-dark.svg">
  <img alt="SRFM 各仓库的关系图：srfm-paper-impl 位于光锥的原点，C++ 核心和 srfm-python 在光锥之内，srfm-lab 在最上方" src="assets/family-light.svg" width="100%">
</picture>

| 仓库 | 简介 |
|---|---|
| [srfm-paper-impl](https://gitlab.com/mattbusel/srfm-paper-impl) | 论文（PDF）、重新生成论文图表的脚本和一个 notebook，以及一个无依赖的小型 Rust 参考实现，实现了核心公式 |
| [Special-Relativity-in-Financial-Modeling](https://gitlab.com/mattbusel/Special-Relativity-in-Financial-Modeling) | C++20 核心和 Windows/Linux 命令行工具：在 OHLCV K 线上计算价格速度（beta）、洛伦兹因子（gamma）、时空间隔标签、克里斯托费尔符号和测地线偏离 |
| [srfm-python](https://gitlab.com/mattbusel/srfm-python) | 纯 Python SDK：pandas 的 `df.srfm` 访问器，以及用于洛伦兹因子流水线的 Polars 封装 |
| **srfm-lab**（本仓库） | 大型多语言研究实验室，在这一思路之上构建交易研究：黑洞（BH）物理信号、策略想法自动化引擎、回测和模拟盘交易 |

Rust crate [fin-stream](https://gitlab.com/mattbusel/fin-stream) 也提供了一个基于同一变换的流式 `lorentz` 模块。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/REFERENCE.md](docs/REFERENCE.md) | 旧版 README 中的其余所有内容：完整的实验室（安装、加密货币回测 + 蒙特卡洛、模拟盘交易、分析流水线、IAE 服务）、性能说明、模块地图、各语言的基础组件、延迟数据 |
| [docs/bh_physics.md](docs/bh_physics.md) | 黑洞物理的详细说明 |
| [docs/iae_architecture.md](docs/iae_architecture.md) | 策略想法自动化引擎（Idea Automation Engine） |
| [docs/wave4_backtest.md](docs/wave4_backtest.md) · [docs/monte_carlo.md](docs/monte_carlo.md) | 回测和蒙特卡洛引擎 |
| [docs/runbook.md](docs/runbook.md) | 运行实盘服务 |
| [RESULTS.md](RESULTS.md) · [HYPOTHESIS.md](HYPOTHESIS.md) · [JOURNAL.md](JOURNAL.md) | 研究结果、假设和实验室日志 |
| [CONTRIBUTING.md](CONTRIBUTING.md) · [CHANGELOG.md](CHANGELOG.md) | 贡献指南和版本历史 |

## 许可证

MIT，见 [LICENSE](LICENSE)。研究代码，按原样提供，不构成投资建议。
