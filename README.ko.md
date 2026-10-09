<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/hero-dark.svg">
  <img alt="SRFM Lab: SPY 일봉 종가를 시공간 세계선으로 그린 그림. 봉마다 광원뿔이 있고, 시간꼴 봉은 청록색, 공간꼴 봉은 빨간 점선, 블랙홀 우물은 호박색으로 칠해져 있다" src="assets/hero-light.svg" width="100%">
</picture>

<p align="center">
  <a href="https://srfm-lab.vercel.app/"><b>프로젝트 사이트</b></a> &nbsp;·&nbsp;
  <a href="#quick-start-one-command">빠른 시작</a> &nbsp;·&nbsp;
  <a href="#how-it-works">작동 원리</a> &nbsp;·&nbsp;
  <a href="#examples">예제</a> &nbsp;·&nbsp;
  <a href="docs/REFERENCE.md">전체 레퍼런스</a>
  <br><br>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

# SRFM Lab

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | 한국어

**특수상대성이론의 수학으로 시장의 가격 봉 하나하나를 평범한 움직임인지 점프인지 분류하고, 질서 있는 움직임이 쌓이는 구간인 "블랙홀"을 찾아내는 Python 퀀트 리서치 랩입니다.** SRFM(Special Relativity in Financial Modeling) 프로젝트 군의 중심 저장소로, 이 신호를 중심으로 백테스트, 몬테카를로 시뮬레이션, 모의투자, 아이디어 발굴 엔진을 갖추고 있습니다.

물리학에서 착안한 트레이딩 신호를 실제 데이터로 탐구해 보고 싶은 퀀트, 학생, 그리고 이것저것 만져 보길 좋아하는 분들을 위한 프로젝트입니다. 패키지로 만든 제품이 아니라 연구용 모노레포입니다. 코어는 함께 들어 있는 데이터로 오프라인 실행되지만, 더 넓은 랩(Rust, Go, Julia, R, C/C++, Zig, Elixir, TypeScript)을 돌리려면 API 키와 외부 서비스가 필요합니다.

> [!IMPORTANT]
> **연구용 코드이며 투자 조언이 아닙니다.** 백테스트와 모의투자 결과는 과거 데이터나 시뮬레이션 데이터에서 한 실험일 뿐, 어떤 전략이 수익을 낸다는 증거가 아닙니다.

<a id="quick-start-one-command"></a>
## 빠른 시작: 명령어 하나로

Python 3.12 이상이 필요합니다. API 키가 필요 없고, 설치가 끝나면 네트워크도 쓰지 않습니다.

```bash
git clone -c core.longpaths=true https://gitlab.com/mattbusel/srfm-lab
cd srfm-lab
python quickstart.py
```

[`quickstart.py`](quickstart.py)는 `.venv/`를 만들고, `requirements-core.txt`에서 버전이 고정된 코어 의존성을 설치한 뒤, `tools/data_cache/`에 들어 있는 SPY, QQQ, DIA 1시간봉으로 [`examples/bh_quickstart.py`](examples/bh_quickstart.py)를 실행합니다. `-c core.longpaths=true`는 Windows에서 클론이 끝까지 완료되도록 해 줍니다. 커밋된 빌드 결과물 일부가 260자 경로 길이 제한을 넘기 때문입니다. 설치 프로그램이나 `.exe`는 없습니다. 연구용 코드이기 때문이며, 호스팅된 [프로젝트 사이트](https://srfm-lab.vercel.app/)에서는 아무것도 설치하지 않고 같은 결과를 볼 수 있습니다.

<img alt="빠른 시작의 터미널 출력: ES, NQ, YM에 대한 봉 개수, 시간꼴 비율, 블랙홀 활성 비율, 발생 횟수, 선행 수익률을 정리한 표" src="assets/quickstart-terminal.svg" width="100%">

<a id="how-it-works"></a>
## 작동 원리

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/physics-explainer-dark.svg">
  <img alt="가격 봉 하나에서 블랙홀까지의 네 단계: 1, beta는 절대 수익률을 속도 한계 c(SPY는 하루 0.5%)로 나눈 값. 2, beta가 1 미만이면 시간꼴로 평범한 움직임, 1 이상이면 공간꼴로 점프. 3, 시간꼴 봉은 질량을 늘리고(질량 곱하기 0.95에 절대 수익률 곱하기 100 곱하기 연속 보너스를 더함), 공간꼴 봉은 질량을 30% 깎고 연속 횟수를 초기화. 4, 시간꼴 봉이 다섯 개 연속된 뒤 질량이 1.5를 넘으면 블랙홀이 형성됨." src="assets/physics-explainer-light.svg" width="100%">
</picture>

쉽게 말하면, 봉마다 **속도** β가 있습니다. 봉의 수익률을 종목별로 정한 "광속" `c`에 견준 값입니다. 느린 봉은 자기 **광원뿔** 안에 머물고(시간꼴, `ds² = c²dt² − dx²`가 양수), 빠른 봉은 그 밖으로 튀어 나갑니다(공간꼴). 질서 있는 봉이 이어지면 **질량**이 쌓이고, 질서 있는 봉이 다섯 개 연속된 뒤 질량이 충분히 커지면 **블랙홀**이 생깁니다. 가격이 그동안 흘러온 방향을 가리키는 우물입니다. **로런츠 인자** γ = 1/√(1 − v²)는 각 봉의 속도를 고유 시간으로 바꿉니다. 이 모든 것이 [`lib/srfm_core.py`](lib/srfm_core.py)에 있는 작은 클래스 두 개, `MinkowskiClassifier`와 `BlackHoleDetector`에 들어 있습니다.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/bh-formation-dark.svg">
  <img alt="2025-12-10부터 2026-01-28까지의 실제 SPY 일봉: 한계값 1에 대한 각 봉의 속도 beta, 시간꼴 봉의 연속 개수, 블랙홀 질량. 우물은 2025-12-30과 2026-01-16에 형성되었고, 두 번 모두 시간꼴 봉이 다섯 개 연속되고 질량이 1.5를 넘은 뒤였다." src="assets/bh-formation-light.svg" width="100%">
</picture>

위 차트는 실제 두 클래스를 함께 들어 있는 실제 봉 데이터로 돌린 결과입니다([`scripts/figures/make_physics_explainer.py`](scripts/figures/make_physics_explainer.py), 2026-09-28 재생성). 랩의 나머지 부분은 모두 이 두 클래스가 내보내는 결과에 연결됩니다.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/pipeline-dark.svg">
  <img alt="파이프라인: 가격 봉, 민코프스키 분류기, 블랙홀 탐지기, 신호 스택, 백테스트, 모의투자 트레이더. 아이디어 엔진이 파라미터를 다시 피드백한다" src="assets/pipeline-light.svg" width="100%">
</picture>

자세히 보기: [BH 물리](docs/bh_physics.md) · [IAE 아키텍처](docs/iae_architecture.md) · [주문 실행 스택](docs/execution_stack.md) · [몬테카를로](docs/monte_carlo.md)

<a id="examples"></a>
## 예제

2026-09-28에 저장소에 들어 있는 데이터로 실제로 출력한 결과입니다.

**1. 기본 실행**(위 터미널 이미지의 표): ES, NQ, YM(대신 SPY, QQQ, DIA 사용), 2020-02-10부터 2026-04-02까지. 블랙홀은 전체 일수의 1.2%에서 2.9% 동안 활성 상태였습니다. 각 발생 후 5일 수익률을 우물 방향으로 부호를 매기면 세 종목 모두 음수인데, 이것이 정직한 결과입니다. 이 파라미터에서는 신호 자체만으로 다음 주를 예측하지 못합니다.

**2. 한 종목, 더 긴 기간**

```text
$ python quickstart.py --symbols ES --horizon 10
       proxy  bars        from          to  timelike %  BH active %  onsets  all days |ret 10d| %  onset signed ret 10d %  onset hit rate %
symbol
ES       SPY  1542  2020-02-10  2026-04-02       43.45         2.85      23                  2.64                   -0.27             56.52
```

**3. 차트 위의 모든 우물.** 같은 실행에서 `examples/output/bh_quickstart.png`가 저장됩니다. 여기서는 우물을 칠하고 각 패널 아래에 질량을 넣어 다시 그렸습니다.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/wells-dark.svg">
  <img alt="최근 750일간 SPY, QQQ, DIA의 일봉 종가. 블랙홀 활성 구간을 칠하고, 각 패널 아래에 형성 임계값과 함께 BH 질량을 그렸다" src="assets/wells-light.svg" width="100%">
</picture>

## 3단계로 사용하기

1. 클론하고 `python quickstart.py`를 실행합니다(위 참고).
2. 질문을 바꿔 봅니다: `--symbols ES,NQ`, `--horizon 10`, 또는 `examples/bh_quickstart.py`의 `PARAMS`(광속 `cf`, 임계값 `bh_form`)를 수정합니다.
3. 더 나아가기: `pip install -r requirements-dev.txt && pytest tests -q`(CI가 실행하는 것)를 돌린 다음, [docs/REFERENCE.md](docs/REFERENCE.md#the-full-lab)에 있는 백테스트, 몬테카를로, 모의투자 트레이더로 넘어갑니다.

## SRFM 프로젝트 군

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/family-dark.svg">
  <img alt="SRFM 저장소 관계도: 광원뿔의 원점에 srfm-paper-impl, 그 안에 C++ 코어와 srfm-python, 맨 위에 srfm-lab" src="assets/family-light.svg" width="100%">
</picture>

| 저장소 | 설명 |
|---|---|
| [srfm-paper-impl](https://gitlab.com/mattbusel/srfm-paper-impl) | 논문(PDF), 논문 그림을 다시 만드는 스크립트와 노트북, 그리고 핵심 수식을 구현한 의존성 없는 작은 Rust 레퍼런스 구현 |
| [Special-Relativity-in-Financial-Modeling](https://gitlab.com/mattbusel/Special-Relativity-in-Financial-Modeling) | C++20 코어와 Windows/Linux 명령줄 도구: OHLCV 봉에서 가격 속도(beta), 로런츠 인자(gamma), 시공간 간격 라벨, 크리스토펠 기호, 측지선 편차를 계산 |
| [srfm-python](https://gitlab.com/mattbusel/srfm-python) | 순수 Python SDK: pandas `df.srfm` 접근자와 로런츠 인자 파이프라인용 Polars 래퍼 |
| **srfm-lab**(이 저장소) | 이 아이디어 위에 트레이딩 연구를 쌓아 올리는 대규모 다중 언어 연구 랩: 블랙홀(BH) 물리 신호, 아이디어 자동화 엔진, 백테스트, 모의투자 |

Rust 크레이트 [fin-stream](https://gitlab.com/mattbusel/fin-stream)에도 같은 변환을 바탕으로 한 스트리밍 `lorentz` 모듈이 들어 있습니다.

## 문서

| 문서 | 내용 |
|---|---|
| [docs/REFERENCE.md](docs/REFERENCE.md) | 예전 README의 나머지 전부: 전체 랩(설치, 암호화폐 백테스트 + 몬테카를로, 모의투자, 분석 파이프라인, IAE 서비스), 성능 메모, 모듈 지도, 언어별 기본 구성 요소, 지연 시간 수치 |
| [docs/bh_physics.md](docs/bh_physics.md) | 블랙홀 물리 심화 설명 |
| [docs/iae_architecture.md](docs/iae_architecture.md) | 아이디어 자동화 엔진(Idea Automation Engine) |
| [docs/wave4_backtest.md](docs/wave4_backtest.md) · [docs/monte_carlo.md](docs/monte_carlo.md) | 백테스트 엔진과 몬테카를로 엔진 |
| [docs/runbook.md](docs/runbook.md) | 라이브 서비스 운영 |
| [RESULTS.md](RESULTS.md) · [HYPOTHESIS.md](HYPOTHESIS.md) · [JOURNAL.md](JOURNAL.md) | 연구 결과, 가설, 랩 일지 |
| [CONTRIBUTING.md](CONTRIBUTING.md) · [CHANGELOG.md](CHANGELOG.md) | 기여 방법과 버전 기록 |

## 라이선스

MIT, [LICENSE](LICENSE) 참고. 연구용 코드이며 있는 그대로 제공되고, 투자 조언이 아닙니다.
