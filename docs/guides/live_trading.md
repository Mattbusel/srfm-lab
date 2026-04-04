# Live and Paper Trading Setup

## Architecture

The live trading system uses:
- **Alpaca Markets** as the broker (both stock ETFs and crypto)
- **tools/live_trader_alpaca.py** as the execution engine
- **spacetime/cache/live_state.json** as the state file for dashboard consumption
- **spacetime/api** as the monitoring API (read-only during live trading)

---

## Paper Trading Setup

### 1. Get Alpaca paper credentials

Sign up at https://alpaca.markets and create paper trading API keys.

### 2. Set environment variables

```bash
export ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxxxxxxxx
export ALPACA_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Add these to your `.env` file (never commit this file):

```bash
echo "ALPACA_API_KEY=PK..." >> .env
echo "ALPACA_API_SECRET=..." >> .env
echo "ALPACA_BASE_URL=https://paper-api.alpaca.markets" >> .env
```

### 3. Start the trader

```bash
make run-live
# OR
python tools/live_trader_alpaca.py
```

The trader will:
1. Connect to Alpaca data streams
2. Subscribe to 15-minute bars for all configured instruments
3. Initialize BH engines for all 3 timeframes
4. Begin processing bars and placing paper orders

### 4. Monitor via dashboard

```bash
# In a second terminal:
make run-api          # Start FastAPI on port 8000
make run-terminal     # Start Arena UI on port 5173 (or terminal on 5174)

# OR start all services:
make docker-up        # docker-compose up
```

Open `http://localhost:5173` and navigate to the "Live" tab.

---

## Live Trading Setup

**Warning**: Live trading uses real money. Only proceed after thorough paper trading validation.

### Checklist before going live

- [ ] At least 3 months of paper trading with satisfactory results
- [ ] Live backtest P&L tracks paper trading P&L within 15%
- [ ] MC blowup probability < 0.5% for your account size
- [ ] Risk limits configured in `config/risk_limits.yaml`
- [ ] Drawdown circuit breaker tested (simulate by reducing initial_equity)
- [ ] Alpaca position limits checked for your account tier
- [ ] Instrument caps set for low-liquidity instruments (NQ, NG, VX)

### Switch to live

```bash
export ALPACA_BASE_URL=https://api.alpaca.markets   # Remove "paper-"
make run-live
```

---

## Instrument Configuration

The instruments traded are configured in `INSTRUMENTS` dict in `tools/live_trader_alpaca.py`.

To add or remove instruments, edit that dict. Each instrument needs:
- `ticker`: The Alpaca ticker symbol
- `type`: `"stock"` or `"crypto"`
- `cf_15m`, `cf_1h`, `cf_1d`: Curvature factors (see CF calibration guide)
- `bh_form`: Formation threshold

---

## Monitoring and Alerts

### Log files

```
spacetime/logs/live_trader.log    # Main trader log
spacetime/logs/api.log            # API server log
```

### Live state file

```json
{
  "timestamp": "2024-01-15T14:30:00Z",
  "equity": 125430.50,
  "positions": {
    "ES": {"frac": 0.35, "side": "long", "entry_price": 4750.0, "unrealized_pnl": 1250.0}
  },
  "bh_states": {
    "ES": {"mass_1d": 2.34, "mass_1h": 1.89, "mass_15m": 0.45, "tf_score": 6}
  },
  "regime": {
    "ES": "BULL"
  }
}
```

### Circuit breakers

The live trader has automatic circuit breakers:
1. **Daily loss limit**: Exit all positions if day P&L < -2% of equity
2. **Drawdown limit**: Exit all positions if drawdown > 15%
3. **Max position**: No single instrument > 65% of equity (tf_score 7 cap)
4. **Instrument caps**: Hard $ caps for illiquid instruments (VX, NG)

Configure limits in `config/risk_limits.yaml`.
