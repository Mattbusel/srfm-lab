# Hyper-Agent — Multi-Agent RL with BH Physics Observations

Hyper-Agent runs five specialized trading agents that observe SRFM BH physics state directly and compete/cooperate to trade ES/NQ/YM.

---

## Core Result (H4)

| Agent | Real Sharpe | Synthetic Sharpe |
|---|---|---|
| Momentum | Best performer | — |
| BH-Follower | -0.009 | 0.234 |
| BH-Contrarian | -0.075 | — |
| MeanReversion | — | — |
| NoiseTrader | — | — |

H4 is not supported: BH-Follower underperforms on real data despite performing well on synthetic. The likely explanation: on synthetic Heston paths, any consistent signal beats random noise. On real ES/NQ/YM, the BH-Follower competes against a much harder market structure where simple heuristic following of BH direction is insufficient without a full execution model.

The Momentum agent outperforms — consistent with momentum being the dominant short-term factor in equity index futures.

---

## Agent Architecture

### BHPolicyNet

```python
class BHPolicyNet(nn.Module):
    """Policy that sees BH mass + direction as observations."""
    def __init__(self, obs_dim=10, act_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),     nn.ReLU(),
            nn.Linear(32, act_dim)            # long / flat / short
        )
    def forward(self, x): return self.net(x)
```

### Observation Space (10-dim)

```
obs = [
    ES_return[-3],  ES_return[-2],  ES_return[-1],   # recent returns (3)
    ES_mass/BH_FORM, NQ_mass/BH_FORM, YM_mass/BH_FORM,  # BH mass (3)
    ES_active, NQ_active, YM_active,                 # BH flags (3)
    convergence_score,                               # sum(active)/3 (1)
]
```

### Action Space

- 0: Short (-1)
- 1: Flat (0)
- 2: Long (+1)

PnL = direction × next_bar_return × $1000 notional per step.

---

## Agent Specializations

Each agent uses the same BHPolicyNet architecture but applies heuristic policy overrides:

| Agent | Override Logic |
|---|---|
| BH-Follower | When ES BH active: force action = BH direction |
| BH-Contrarian | When ES BH active: force action = -BH direction |
| Momentum | Always: sign of last bar's return |
| MeanReversion | No override — pure neural policy |
| NoiseTrader | No override — pure neural policy (baseline) |

---

## ELO Tournament

After all episodes, agents are ranked by episode PnL in head-to-head comparisons using an ELO rating system (K=32, starting ELO=1500). This produces a skill ranking that accounts for the difficulty of each episode independently of absolute PnL level.

---

## Convergence vs Calm Episode Analysis

Episodes are classified by the fraction of bars where convergence is active. Episodes with >20% convergence fraction are "convergence episodes." BH-Follower PnL is compared across episode types to test whether the agent performs better when its defining signal (BH physics) is most active.

---

## Files

- `run_aeternus_real.py` — Module 6 implementation (BHPolicyNet, agent loop, ELO tournament)
- `aeternus/hyper_agent/` — Full Hyper-Agent module (domain randomization, adversarial training, online system identification, transfer learning, robustness evaluation, MARL)
- `lib/math/mean_field_games.py` — MFG Nash execution, multi-agent equilibrium
- `idea-engine/rl/ppo_trader.py` — PPO with GAE, entropy bonus, actor-critic
