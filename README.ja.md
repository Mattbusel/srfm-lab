<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/hero-dark.svg">
  <img alt="SRFM Lab：SPY の日足終値を時空の世界線として描いた図。各足に光円錐が付き、時間的な足はティール、空間的な足は赤の破線、ブラックホールの井戸は琥珀色で塗られている" src="assets/hero-light.svg" width="100%">
</picture>

<p align="center">
  <a href="https://srfm-lab.vercel.app/"><b>プロジェクトサイト</b></a> &nbsp;·&nbsp;
  <a href="#quick-start-one-command">クイックスタート</a> &nbsp;·&nbsp;
  <a href="#how-it-works">仕組み</a> &nbsp;·&nbsp;
  <a href="#examples">実行例</a> &nbsp;·&nbsp;
  <a href="docs/REFERENCE.md">詳細リファレンス</a>
  <br><br>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

# SRFM Lab

[English](README.md) | [简体中文](README.zh-CN.md) | 日本語 | [한국어](README.ko.md)

**特殊相対性理論の数学を使って、相場の各ローソク足を「通常の値動き」か「ジャンプ」かに分類し、秩序だった値動きが積み重なる区間「ブラックホール」を見つけ出す、Python 製のクオンツ研究ラボです。** SRFM（Special Relativity in Financial Modeling）ファミリーの中心となるリポジトリで、このシグナルを軸にバックテスト、モンテカルロ・シミュレーション、ペーパートレード、アイデア探索エンジンを備えています。

物理学に着想を得たトレーディングシグナルを実データで試してみたいクオンツ、学生、いじり好きの方に向けたプロジェクトです。パッケージ化された製品ではなく研究用のモノレポです。コア部分は同梱データだけでオフライン実行できますが、ラボ全体（Rust、Go、Julia、R、C/C++、Zig、Elixir、TypeScript）を動かすには API キーや外部サービスが必要です。

> [!IMPORTANT]
> **研究用のコードであり、投資助言ではありません。** バックテストやペーパートレードの結果は過去データまたはシミュレーションデータ上の実験であり、どの戦略についても利益が出ることを示すものではありません。

<a id="quick-start-one-command"></a>
## クイックスタート：コマンドひとつで

Python 3.12 以上が必要です。API キーは不要で、インストール後はネットワークも使いません。

```bash
git clone -c core.longpaths=true https://gitlab.com/mattbusel/srfm-lab
cd srfm-lab
python quickstart.py
```

[`quickstart.py`](quickstart.py) は `.venv/` を作成し、`requirements-core.txt` からバージョン固定されたコア依存をインストールしたうえで、`tools/data_cache/` に同梱されている SPY、QQQ、DIA の 1 時間足で [`examples/bh_quickstart.py`](examples/bh_quickstart.py) を実行します。`-c core.longpaths=true` は Windows でクローンを最後まで完了させるためのものです。コミットされているビルド成果物の一部が 260 文字のパス長制限を超えているためです。インストーラーや `.exe` はありません。これは研究用コードであり、ホストされている[プロジェクトサイト](https://srfm-lab.vercel.app/)なら何もインストールせずに同じ結果を見られます。

<img alt="クイックスタートのターミナル出力：ES、NQ、YM について、足の本数、時間的な足の割合、ブラックホールがアクティブな割合、発生回数、フォワードリターンをまとめた表" src="assets/quickstart-terminal.svg" width="100%">

<a id="how-it-works"></a>
## 仕組み

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/physics-explainer-dark.svg">
  <img alt="1 本のローソク足からブラックホールに至る 4 つのステップ：1、beta は絶対リターンを速度の上限 c（SPY では 1 日 0.5%）で割った値。2、beta が 1 未満なら時間的で通常の値動き、1 以上なら空間的でジャンプ。3、時間的な足は質量を増やし（質量 × 0.95 に、絶対リターン × 100 × 連続ボーナスを加える）、空間的な足は質量を 30% 減らして連続カウントをリセットする。4、時間的な足が 5 本連続したあとに質量が 1.5 を超えるとブラックホールが形成される。" src="assets/physics-explainer-light.svg" width="100%">
</picture>

ざっくり言うと、各ローソク足には**速度** β が与えられます。これはその足のリターンを、銘柄ごとに決めた「光速」`c` と比べた値です。ゆっくりした足は自分の**光円錐**の内側に収まり（時間的、`ds² = c²dt² − dx²` が正）、速い足はその外へ飛び出します（空間的）。秩序だった足が続くと**質量**が溜まり、秩序だった足が 5 本連続したあとに質量が十分大きくなると**ブラックホール**が生まれます。これは価格がそれまで漂ってきた方向を指し示す井戸です。**ローレンツ因子** γ = 1/√(1 − v²) は各足の速度を固有時に変換します。これらはすべて [`lib/srfm_core.py`](lib/srfm_core.py) にある 2 つの小さなクラス、`MinkowskiClassifier` と `BlackHoleDetector` に収まっています。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/bh-formation-dark.svg">
  <img alt="2025-12-10 から 2026-01-28 までの実際の SPY 日足：上限 1 に対する各足の速度 beta、時間的な足の連続本数、ブラックホールの質量。井戸は 2025-12-30 と 2026-01-16 に形成され、いずれも時間的な足が 5 本連続し、質量が 1.5 を超えたあとだった。" src="assets/bh-formation-light.svg" width="100%">
</picture>

上のチャートは、実際の 2 つのクラスを同梱の実データで動かした結果です（[`scripts/figures/make_physics_explainer.py`](scripts/figures/make_physics_explainer.py)、2026-09-28 に再生成）。ラボの他の部分はすべて、この 2 つのクラスの出力につながっています。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/pipeline-dark.svg">
  <img alt="パイプライン：ローソク足、ミンコフスキー分類器、ブラックホール検出器、シグナルスタック、バックテスト、ペーパートレーダー。アイデアエンジンがパラメータをフィードバックする" src="assets/pipeline-light.svg" width="100%">
</picture>

詳しい解説：[BH の物理](docs/bh_physics.md) · [IAE のアーキテクチャ](docs/iae_architecture.md) · [執行スタック](docs/execution_stack.md) · [モンテカルロ](docs/monte_carlo.md)

<a id="examples"></a>
## 実行例

2026-09-28 に、リポジトリ同梱のデータで実際に出力した結果です。

**1. デフォルト実行**（上のターミナル画像の表）：ES、NQ、YM（代わりに SPY、QQQ、DIA を使用）、2020-02-10 から 2026-04-02。ブラックホールがアクティブだったのは全日数の 1.2% から 2.9% でした。各発生後の 5 日リターンを井戸の方向で符号付けすると 3 銘柄すべてでマイナスになります。これが正直な結果で、このパラメータでは素のシグナルは翌週を予測できません。

**2. 1 銘柄、より長いホライズン**

```text
$ python quickstart.py --symbols ES --horizon 10
       proxy  bars        from          to  timelike %  BH active %  onsets  all days |ret 10d| %  onset signed ret 10d %  onset hit rate %
symbol
ES       SPY  1542  2020-02-10  2026-04-02       43.45         2.85      23                  2.64                   -0.27             56.52
```

**3. チャート上のすべての井戸**。同じ実行で `examples/output/bh_quickstart.png` が保存されます。ここでは井戸を塗りつぶし、各パネルの下に質量を描き直しています。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/wells-dark.svg">
  <img alt="SPY、QQQ、DIA の直近 750 日の日足終値。ブラックホールがアクティブな期間を塗りつぶし、各パネルの下に BH の質量を形成しきい値とともにプロットしている" src="assets/wells-light.svg" width="100%">
</picture>

## 3 ステップで使う

1. クローンして `python quickstart.py` を実行します（上記）。
2. 問いを変えてみます：`--symbols ES,NQ`、`--horizon 10`、または `examples/bh_quickstart.py` の `PARAMS`（光速 `cf`、しきい値 `bh_form`）を編集します。
3. さらに先へ：`pip install -r requirements-dev.txt && pytest tests -q`（CI が実行しているもの）を試し、続いて [docs/REFERENCE.md](docs/REFERENCE.md#the-full-lab) にあるバックテスト、モンテカルロ、ペーパートレーダーへ進みます。

## SRFM ファミリー

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/family-dark.svg">
  <img alt="SRFM リポジトリの関係図：光円錐の原点に srfm-paper-impl、その内側に C++ コアと srfm-python、一番上に srfm-lab" src="assets/family-light.svg" width="100%">
</picture>

| リポジトリ | 内容 |
|---|---|
| [srfm-paper-impl](https://gitlab.com/mattbusel/srfm-paper-impl) | 論文（PDF）、その図を再生成するスクリプトとノートブック、コアとなる数式を実装した依存なしの小さな Rust リファレンス実装 |
| [Special-Relativity-in-Financial-Modeling](https://gitlab.com/mattbusel/Special-Relativity-in-Financial-Modeling) | C++20 のコアと Windows/Linux 向けコマンドラインツール：OHLCV の足に対して価格速度（beta）、ローレンツ因子（gamma）、時空間隔のラベル、クリストッフェル記号、測地線偏差を計算 |
| [srfm-python](https://gitlab.com/mattbusel/srfm-python) | ピュア Python の SDK：pandas の `df.srfm` アクセサと、ローレンツ因子パイプライン用の Polars ラッパー |
| **srfm-lab**（このリポジトリ） | このアイデアの上にトレーディング研究を組み立てる大規模な多言語研究ラボ：ブラックホール（BH）物理シグナル、アイデア自動化エンジン、バックテスト、ペーパートレード |

Rust クレート [fin-stream](https://gitlab.com/mattbusel/fin-stream) にも、同じ変換に基づくストリーミング用の `lorentz` モジュールが含まれています。

## ドキュメント

| ドキュメント | 内容 |
|---|---|
| [docs/REFERENCE.md](docs/REFERENCE.md) | 旧 README のその他すべて：ラボ全体（インストール、暗号資産のバックテスト + モンテカルロ、ペーパートレード、分析パイプライン、IAE サービス）、パフォーマンスに関するメモ、モジュールマップ、言語別のプリミティブ、レイテンシの数値 |
| [docs/bh_physics.md](docs/bh_physics.md) | ブラックホール物理の詳細 |
| [docs/iae_architecture.md](docs/iae_architecture.md) | アイデア自動化エンジン（Idea Automation Engine） |
| [docs/wave4_backtest.md](docs/wave4_backtest.md) · [docs/monte_carlo.md](docs/monte_carlo.md) | バックテストエンジンとモンテカルロエンジン |
| [docs/runbook.md](docs/runbook.md) | ライブサービスの運用 |
| [RESULTS.md](RESULTS.md) · [HYPOTHESIS.md](HYPOTHESIS.md) · [JOURNAL.md](JOURNAL.md) | 研究結果、仮説、ラボの日誌 |
| [CONTRIBUTING.md](CONTRIBUTING.md) · [CHANGELOG.md](CHANGELOG.md) | コントリビューションとバージョン履歴 |

## ライセンス

MIT。[LICENSE](LICENSE) を参照してください。研究用のコードを現状のまま提供しており、投資助言ではありません。
