-- =============================================================================
-- 06_seed_data.sql
-- SRFM Lab — Seed Data: Instrument Universe, Strategy Metadata
-- PostgreSQL 15+
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Instruments — Equity Indices
-- ---------------------------------------------------------------------------
INSERT INTO instruments (symbol, name, asset_class, base_currency, quote_currency,
    tick_size, cf_15m, cf_1h, cf_1d, bh_form, bh_collapse, bh_decay, corr_group, alpaca_ticker)
VALUES
-- ES: S&P 500 futures proxy via SPY ETF
('ES', 'S&P 500 E-mini Futures (via SPY)', 'equity_index', 'USD', 'USD',
    0.01, 0.00030, 0.00100, 0.00500, 1.5, 1.0, 0.95, 'equity_us', 'SPY'),

-- NQ: Nasdaq futures proxy via QQQ
('NQ', 'Nasdaq 100 E-mini Futures (via QQQ)', 'equity_index', 'USD', 'USD',
    0.01, 0.00040, 0.00120, 0.00600, 1.5, 1.0, 0.95, 'equity_us', 'QQQ'),

-- YM: Dow futures proxy via DIA
('YM', 'Dow Jones E-mini Futures (via DIA)', 'equity_index', 'USD', 'USD',
    0.01, 0.00025, 0.00080, 0.00400, 1.5, 1.0, 0.95, 'equity_us', 'DIA'),

-- RTY: Russell 2000 futures proxy via IWM
('RTY', 'Russell 2000 Futures (via IWM)', 'equity_index', 'USD', 'USD',
    0.01, 0.00035, 0.00110, 0.00550, 1.5, 1.0, 0.95, 'equity_us', 'IWM'),

-- SPY: direct ETF
('SPY', 'SPDR S&P 500 ETF', 'equity_index', 'USD', 'USD',
    0.01, 0.00030, 0.00100, 0.00500, 1.5, 1.0, 0.95, 'equity_us', 'SPY'),

-- QQQ: direct ETF
('QQQ', 'Invesco QQQ Trust ETF', 'equity_index', 'USD', 'USD',
    0.01, 0.00040, 0.00120, 0.00600, 1.5, 1.0, 0.95, 'equity_us', 'QQQ')

ON CONFLICT (symbol) DO UPDATE SET
    name         = EXCLUDED.name,
    cf_15m       = EXCLUDED.cf_15m,
    cf_1h        = EXCLUDED.cf_1h,
    cf_1d        = EXCLUDED.cf_1d,
    bh_form      = EXCLUDED.bh_form,
    updated_at   = CURRENT_TIMESTAMP;

-- ---------------------------------------------------------------------------
-- Instruments — Commodities (Energy)
-- ---------------------------------------------------------------------------
INSERT INTO instruments (symbol, name, asset_class, base_currency, quote_currency,
    tick_size, cf_15m, cf_1h, cf_1d, bh_form, bh_collapse, bh_decay, corr_group, alpaca_ticker)
VALUES
('CL', 'Crude Oil WTI Futures (via USO)', 'commodity', 'USD', 'USD',
    0.01, 0.00150, 0.00400, 0.01500, 1.8, 1.0, 0.95, 'energy', 'USO'),

('NG', 'Natural Gas Futures (via UNG)', 'commodity', 'USD', 'USD',
    0.001, 0.00200, 0.00600, 0.02000, 1.8, 1.0, 0.95, 'energy', 'UNG'),

('USO', 'United States Oil Fund ETF', 'commodity', 'USD', 'USD',
    0.01, 0.00150, 0.00400, 0.01500, 1.8, 1.0, 0.95, 'energy', 'USO')

ON CONFLICT (symbol) DO UPDATE SET
    cf_15m = EXCLUDED.cf_15m, cf_1h = EXCLUDED.cf_1h, cf_1d = EXCLUDED.cf_1d,
    updated_at = CURRENT_TIMESTAMP;

-- ---------------------------------------------------------------------------
-- Instruments — Commodities (Metals)
-- ---------------------------------------------------------------------------
INSERT INTO instruments (symbol, name, asset_class, base_currency, quote_currency,
    tick_size, cf_15m, cf_1h, cf_1d, bh_form, bh_collapse, bh_decay, corr_group, alpaca_ticker)
VALUES
('GC', 'Gold Futures (via GLD)', 'commodity', 'USD', 'USD',
    0.10, 0.00080, 0.00250, 0.00800, 1.5, 1.0, 0.95, 'metals', 'GLD'),

('SI', 'Silver Futures (via SLV)', 'commodity', 'USD', 'USD',
    0.005, 0.00080, 0.00250, 0.00800, 1.5, 1.0, 0.95, 'metals', 'SLV'),

('GLD', 'SPDR Gold Shares ETF', 'commodity', 'USD', 'USD',
    0.01, 0.00080, 0.00250, 0.00800, 1.5, 1.0, 0.95, 'metals', 'GLD'),

('SLV', 'iShares Silver Trust ETF', 'commodity', 'USD', 'USD',
    0.01, 0.00080, 0.00250, 0.00800, 1.5, 1.0, 0.95, 'metals', 'SLV')

ON CONFLICT (symbol) DO UPDATE SET
    cf_15m = EXCLUDED.cf_15m, cf_1h = EXCLUDED.cf_1h, cf_1d = EXCLUDED.cf_1d,
    updated_at = CURRENT_TIMESTAMP;

-- ---------------------------------------------------------------------------
-- Instruments — Bonds
-- ---------------------------------------------------------------------------
INSERT INTO instruments (symbol, name, asset_class, base_currency, quote_currency,
    tick_size, cf_15m, cf_1h, cf_1d, bh_form, bh_collapse, bh_decay, corr_group, alpaca_ticker)
VALUES
('ZB', '30-Year US Treasury Bond Futures (via TLT)', 'bond', 'USD', 'USD',
    0.015625, 0.00050, 0.00150, 0.00500, 1.5, 1.0, 0.95, 'bonds_us', 'TLT'),

('ZN', '10-Year US Treasury Note Futures (via IEF)', 'bond', 'USD', 'USD',
    0.015625, 0.00030, 0.00100, 0.00350, 1.5, 1.0, 0.95, 'bonds_us', 'IEF'),

('TLT', 'iShares 20+ Year Treasury Bond ETF', 'bond', 'USD', 'USD',
    0.01, 0.00050, 0.00150, 0.00500, 1.5, 1.0, 0.95, 'bonds_us', 'TLT'),

('IEF', 'iShares 7-10 Year Treasury Bond ETF', 'bond', 'USD', 'USD',
    0.01, 0.00030, 0.00100, 0.00350, 1.5, 1.0, 0.95, 'bonds_us', 'IEF')

ON CONFLICT (symbol) DO UPDATE SET
    cf_15m = EXCLUDED.cf_15m, cf_1h = EXCLUDED.cf_1h, cf_1d = EXCLUDED.cf_1d,
    updated_at = CURRENT_TIMESTAMP;

-- ---------------------------------------------------------------------------
-- Instruments — Volatility
-- ---------------------------------------------------------------------------
INSERT INTO instruments (symbol, name, asset_class, base_currency, quote_currency,
    tick_size, cf_15m, cf_1h, cf_1d, bh_form, bh_collapse, bh_decay, corr_group, alpaca_ticker)
VALUES
('VX', 'VIX Futures (via VIXY)', 'volatility', 'USD', 'USD',
    0.05, 0.00300, 0.00800, 0.02500, 1.8, 1.0, 0.92, 'volatility', 'VIXY'),

('VIX', 'CBOE Volatility Index (via VIXY)', 'volatility', 'USD', 'USD',
    0.01, 0.00300, 0.00800, 0.02500, 1.8, 1.0, 0.92, 'volatility', 'VIXY'),

('VIXY', 'ProShares VIX Short-Term Futures ETF', 'volatility', 'USD', 'USD',
    0.01, 0.00300, 0.00800, 0.02500, 1.8, 1.0, 0.92, 'volatility', 'VIXY')

ON CONFLICT (symbol) DO UPDATE SET
    cf_15m = EXCLUDED.cf_15m, cf_1h = EXCLUDED.cf_1h, cf_1d = EXCLUDED.cf_1d,
    updated_at = CURRENT_TIMESTAMP;

-- ---------------------------------------------------------------------------
-- Instruments — Crypto (Large Cap)
-- ---------------------------------------------------------------------------
INSERT INTO instruments (symbol, name, asset_class, base_currency, quote_currency,
    tick_size, cf_15m, cf_1h, cf_1d, bh_form, bh_collapse, bh_decay, corr_group, alpaca_ticker)
VALUES
('BTC', 'Bitcoin', 'crypto', 'BTC', 'USD',
    1.0, 0.00500, 0.01500, 0.05000, 1.5, 1.0, 0.95, 'crypto_large', 'BTC/USD'),

('ETH', 'Ethereum', 'crypto', 'ETH', 'USD',
    0.01, 0.00700, 0.02000, 0.07000, 1.5, 1.0, 0.95, 'crypto_large', 'ETH/USD'),

('SOL', 'Solana', 'crypto', 'SOL', 'USD',
    0.01, 0.01000, 0.03000, 0.10000, 1.5, 1.0, 0.95, 'crypto_mid', 'SOL/USD'),

('XRP', 'Ripple', 'crypto', 'XRP', 'USD',
    0.0001, 0.01000, 0.03000, 0.10000, 1.5, 1.0, 0.95, 'crypto_mid', 'XRP/USD'),

('AVAX', 'Avalanche', 'crypto', 'AVAX', 'USD',
    0.001, 0.01200, 0.03500, 0.12000, 1.5, 1.0, 0.95, 'crypto_mid', 'AVAX/USD'),

('LINK', 'Chainlink', 'crypto', 'LINK', 'USD',
    0.001, 0.01200, 0.03500, 0.12000, 1.5, 1.0, 0.95, 'crypto_mid', 'LINK/USD'),

('DOT', 'Polkadot', 'crypto', 'DOT', 'USD',
    0.001, 0.01200, 0.03500, 0.12000, 1.5, 1.0, 0.95, 'crypto_mid', 'DOT/USD'),

('UNI', 'Uniswap', 'crypto', 'UNI', 'USD',
    0.001, 0.01500, 0.04500, 0.15000, 1.5, 1.0, 0.95, 'crypto_defi', 'UNI/USD'),

('AAVE', 'Aave', 'crypto', 'AAVE', 'USD',
    0.01, 0.01500, 0.04500, 0.15000, 1.5, 1.0, 0.95, 'crypto_defi', 'AAVE/USD'),

('LTC', 'Litecoin', 'crypto', 'LTC', 'USD',
    0.01, 0.01000, 0.03000, 0.10000, 1.5, 1.0, 0.95, 'crypto_mid', 'LTC/USD'),

('BCH', 'Bitcoin Cash', 'crypto', 'BCH', 'USD',
    0.01, 0.01200, 0.03500, 0.12000, 1.5, 1.0, 0.95, 'crypto_mid', 'BCH/USD'),

('ADA', 'Cardano', 'crypto', 'ADA', 'USD',
    0.0001, 0.01500, 0.04500, 0.15000, 1.5, 1.0, 0.95, 'crypto_mid', 'ADA/USD'),

('DOGE', 'Dogecoin', 'crypto', 'DOGE', 'USD',
    0.00001, 0.02000, 0.06000, 0.20000, 1.5, 1.0, 0.95, 'crypto_small', 'DOGE/USD'),

('SHIB', 'Shiba Inu', 'crypto', 'SHIB', 'USD',
    0.000000001, 0.02500, 0.07500, 0.25000, 1.5, 1.0, 0.95, 'crypto_small', 'SHIB/USD'),

('FIL', 'Filecoin', 'crypto', 'FIL', 'USD',
    0.001, 0.01800, 0.05500, 0.18000, 1.5, 1.0, 0.95, 'crypto_small', 'FIL/USD'),

('GRT', 'The Graph', 'crypto', 'GRT', 'USD',
    0.0001, 0.02000, 0.06000, 0.20000, 1.5, 1.0, 0.95, 'crypto_small', 'GRT/USD'),

('BAT', 'Basic Attention Token', 'crypto', 'BAT', 'USD',
    0.0001, 0.02000, 0.06000, 0.20000, 1.5, 1.0, 0.95, 'crypto_small', 'BAT/USD'),

('CRV', 'Curve DAO Token', 'crypto', 'CRV', 'USD',
    0.0001, 0.02000, 0.06000, 0.20000, 1.5, 1.0, 0.95, 'crypto_defi', 'CRV/USD'),

('SUSHI', 'SushiSwap', 'crypto', 'SUSHI', 'USD',
    0.0001, 0.02000, 0.06000, 0.20000, 1.5, 1.0, 0.95, 'crypto_defi', 'SUSHI/USD'),

('MATIC', 'Polygon', 'crypto', 'MATIC', 'USD',
    0.0001, 0.01500, 0.04500, 0.15000, 1.5, 1.0, 0.95, 'crypto_mid', 'MATIC/USD'),

('ATOM', 'Cosmos', 'crypto', 'ATOM', 'USD',
    0.001, 0.01500, 0.04500, 0.15000, 1.5, 1.0, 0.95, 'crypto_mid', 'ATOM/USD'),

('ALGO', 'Algorand', 'crypto', 'ALGO', 'USD',
    0.0001, 0.02000, 0.06000, 0.20000, 1.5, 1.0, 0.95, 'crypto_small', 'ALGO/USD')

ON CONFLICT (symbol) DO UPDATE SET
    cf_15m = EXCLUDED.cf_15m, cf_1h = EXCLUDED.cf_1h, cf_1d = EXCLUDED.cf_1d,
    corr_group = EXCLUDED.corr_group, alpaca_ticker = EXCLUDED.alpaca_ticker,
    updated_at = CURRENT_TIMESTAMP;

-- ---------------------------------------------------------------------------
-- Instruments — Forex
-- ---------------------------------------------------------------------------
INSERT INTO instruments (symbol, name, asset_class, base_currency, quote_currency,
    tick_size, cf_15m, cf_1h, cf_1d, bh_form, bh_collapse, bh_decay, corr_group, alpaca_ticker)
VALUES
('EURUSD', 'Euro / US Dollar', 'forex', 'EUR', 'USD',
    0.00001, 0.00050, 0.00150, 0.00500, 1.5, 1.0, 0.95, 'forex_majors', NULL),

('GBPUSD', 'British Pound / US Dollar', 'forex', 'GBP', 'USD',
    0.00001, 0.00060, 0.00180, 0.00600, 1.5, 1.0, 0.95, 'forex_majors', NULL),

('USDJPY', 'US Dollar / Japanese Yen', 'forex', 'USD', 'JPY',
    0.001, 0.00050, 0.00150, 0.00500, 1.5, 1.0, 0.95, 'forex_majors', NULL),

('AUDUSD', 'Australian Dollar / US Dollar', 'forex', 'AUD', 'USD',
    0.00001, 0.00060, 0.00180, 0.00600, 1.5, 1.0, 0.95, 'forex_majors', NULL),

('USDCAD', 'US Dollar / Canadian Dollar', 'forex', 'USD', 'CAD',
    0.00001, 0.00050, 0.00150, 0.00500, 1.5, 1.0, 0.95, 'forex_majors', NULL)

ON CONFLICT (symbol) DO UPDATE SET
    cf_15m = EXCLUDED.cf_15m, cf_1h = EXCLUDED.cf_1h, cf_1d = EXCLUDED.cf_1d,
    updated_at = CURRENT_TIMESTAMP;

-- ---------------------------------------------------------------------------
-- Historical strategy run metadata (representative runs)
-- ---------------------------------------------------------------------------
INSERT INTO strategy_runs (
    run_name, strategy_name, strategy_version, run_type,
    start_date, end_date,
    initial_equity, final_equity,
    total_return_pct, cagr, sharpe, sortino, max_drawdown_pct, calmar,
    win_rate, profit_factor, n_trades,
    parameters, tags
) VALUES
(
    'LARSA-v14-baseline-2020-2024', 'larsa', 'v14', 'backtest',
    '2020-01-01', '2024-01-01',
    100000, 387000,
    2.870, 0.402, 1.84, 2.31, 0.218, 1.84,
    0.58, 2.3, 412,
    '{"bh_form": 1.5, "min_tf_score": 2, "pos_floor": 0.0, "instruments": ["ES","NQ","YM","CL","GC","ZB","NG","VX"]}',
    ARRAY['baseline', 'v14', 'no-harvest']
),
(
    'LARSA-v16-gear1-only-2020-2024', 'larsa', 'v16', 'backtest',
    '2020-01-01', '2024-01-01',
    3000000, 8940000,
    1.980, 0.318, 1.72, 2.18, 0.195, 1.63,
    0.56, 2.1, 381,
    '{"bh_form": 1.5, "min_tf_score": 2, "gear": "tail_only", "instruments": ["ES","NQ","YM","CL","GC","ZB","NG","VX"]}',
    ARRAY['v16', 'gear1', 'tail-only']
),
(
    'LARSA-v16-full-2021-2024', 'larsa', 'v16', 'backtest',
    '2021-01-01', '2024-01-01',
    5000000, 14200000,
    1.840, 0.430, 1.96, 2.44, 0.182, 2.36,
    0.59, 2.6, 628,
    '{"bh_form": 1.5, "min_tf_score": 2, "gear": "full", "harvest_z_entry": 1.5, "instruments": ["ES","NQ","YM","CL","GC","ZB","NG","VX"]}',
    ARRAY['v16', 'full', 'gear1+gear2']
),
(
    'LARSA-sweep-bh_form-1.2', 'larsa', 'v16', 'backtest',
    '2020-01-01', '2024-01-01',
    100000, 321000,
    2.210, 0.346, 1.62, 2.04, 0.251, 1.38,
    0.54, 1.9, 529,
    '{"bh_form": 1.2, "min_tf_score": 2, "instruments": ["ES","NQ","YM","CL","GC","ZB","NG","VX"]}',
    ARRAY['sweep-bh_form']
),
(
    'LARSA-sweep-bh_form-2.0', 'larsa', 'v16', 'backtest',
    '2020-01-01', '2024-01-01',
    100000, 298000,
    1.980, 0.318, 1.78, 2.26, 0.198, 1.61,
    0.61, 2.5, 274,
    '{"bh_form": 2.0, "min_tf_score": 2, "instruments": ["ES","NQ","YM","CL","GC","ZB","NG","VX"]}',
    ARRAY['sweep-bh_form']
),
(
    'LARSA-v16-crypto-expansion', 'larsa', 'v16', 'backtest',
    '2021-01-01', '2024-01-01',
    100000, 448000,
    3.480, 0.530, 1.91, 2.39, 0.312, 1.70,
    0.55, 2.2, 892,
    '{"bh_form": 1.5, "min_tf_score": 2, "instruments": ["ES","NQ","YM","CL","GC","ZB","NG","VX","BTC","ETH","SOL"]}',
    ARRAY['v16', 'crypto-expansion']
)
ON CONFLICT DO NOTHING;

-- ---------------------------------------------------------------------------
-- Sample BH formation events (representative examples for testing)
-- ---------------------------------------------------------------------------
-- Using instrument IDs by symbol lookup
WITH inst AS (
    SELECT id, symbol FROM instruments
    WHERE symbol IN ('ES', 'NQ', 'CL', 'GC', 'BTC')
)
INSERT INTO bh_formations (
    instrument_id, timeframe, formed_at, collapsed_at,
    peak_mass, duration_bars, direction, regime_at_formation,
    price_at_formation, price_at_collapse, was_profitable
)
SELECT
    i.id,
    v.timeframe,
    v.formed_at::TIMESTAMPTZ,
    v.collapsed_at::TIMESTAMPTZ,
    v.peak_mass,
    v.duration_bars,
    v.direction,
    v.regime,
    v.entry_price,
    v.exit_price,
    v.was_profitable
FROM inst i
JOIN (VALUES
    ('ES',  '1d', '2022-06-10', '2022-06-24', 2.84, 14, -1, 'BEAR',   3900.00, 3666.00, TRUE),
    ('ES',  '1d', '2023-03-08', '2023-03-22', 2.31, 14,  1, 'BULL',   3975.00, 4050.00, TRUE),
    ('NQ',  '1d', '2022-09-13', '2022-09-27', 3.12, 14, -1, 'BEAR',  11600.00,10800.00, TRUE),
    ('CL',  '1d', '2022-03-07', '2022-03-21', 2.65, 14,  1, 'BULL',    105.00,  109.00, TRUE),
    ('GC',  '1d', '2023-03-10', '2023-03-24', 2.18, 14,  1, 'BULL',   1840.00, 1990.00, TRUE),
    ('BTC', '1d', '2021-10-20', '2021-11-03', 3.45, 14,  1, 'BULL',  61000.00,68000.00, TRUE),
    ('BTC', '1d', '2022-05-10', '2022-05-24', 4.01, 14, -1, 'BEAR',  31000.00,28000.00, TRUE),
    ('ES',  '1h', '2023-08-24', '2023-08-25', 1.89,  6, -1, 'BEAR',   4400.00, 4350.00, TRUE),
    ('NQ',  '1h', '2024-01-05', '2024-01-06', 2.02,  5,  1, 'BULL',  16200.00,16450.00, TRUE),
    ('CL',  '1h', '2022-09-01', '2022-09-02', 2.44,  8, -1, 'BEAR',     86.00,   82.00, TRUE)
) AS v(symbol, timeframe, formed_at, collapsed_at, peak_mass, duration_bars,
       direction, regime, entry_price, exit_price, was_profitable)
ON i.symbol = v.symbol
ON CONFLICT DO NOTHING;

-- ---------------------------------------------------------------------------
-- Trading calendar: mark weekends as non-trading (partial, 2023-2026)
-- ---------------------------------------------------------------------------
INSERT INTO trading_calendar (calendar_date, is_trading_day, notes)
SELECT
    d::DATE,
    EXTRACT(DOW FROM d) NOT IN (0, 6),   -- 0=Sun, 6=Sat
    CASE EXTRACT(DOW FROM d)
        WHEN 0 THEN 'Sunday'
        WHEN 6 THEN 'Saturday'
    END
FROM GENERATE_SERIES('2023-01-01'::DATE, '2026-12-31'::DATE, '1 day') d
ON CONFLICT (calendar_date) DO NOTHING;

-- Mark known US equity market holidays (partial list)
UPDATE trading_calendar SET is_trading_day = FALSE, notes = 'New Year''s Day'
WHERE calendar_date IN ('2023-01-02', '2024-01-01', '2025-01-01', '2026-01-01');

UPDATE trading_calendar SET is_trading_day = FALSE, notes = 'MLK Day'
WHERE calendar_date IN ('2023-01-16', '2024-01-15', '2025-01-20', '2026-01-19');

UPDATE trading_calendar SET is_trading_day = FALSE, notes = 'Presidents Day'
WHERE calendar_date IN ('2023-02-20', '2024-02-19', '2025-02-17', '2026-02-16');

UPDATE trading_calendar SET is_trading_day = FALSE, notes = 'Good Friday'
WHERE calendar_date IN ('2023-04-07', '2024-03-29', '2025-04-18', '2026-04-03');

UPDATE trading_calendar SET is_trading_day = FALSE, notes = 'Memorial Day'
WHERE calendar_date IN ('2023-05-29', '2024-05-27', '2025-05-26', '2026-05-25');

UPDATE trading_calendar SET is_trading_day = FALSE, notes = 'Juneteenth'
WHERE calendar_date IN ('2023-06-19', '2024-06-19', '2025-06-19', '2026-06-19');

UPDATE trading_calendar SET is_trading_day = FALSE, notes = 'Independence Day'
WHERE calendar_date IN ('2023-07-04', '2024-07-04', '2025-07-04', '2026-07-03');

UPDATE trading_calendar SET is_trading_day = FALSE, notes = 'Labor Day'
WHERE calendar_date IN ('2023-09-04', '2024-09-02', '2025-09-01', '2026-09-07');

UPDATE trading_calendar SET is_trading_day = FALSE, notes = 'Thanksgiving'
WHERE calendar_date IN ('2023-11-23', '2024-11-28', '2025-11-27', '2026-11-26');

UPDATE trading_calendar SET is_trading_day = FALSE, notes = 'Christmas'
WHERE calendar_date IN ('2023-12-25', '2024-12-25', '2025-12-25', '2026-12-25');

-- ---------------------------------------------------------------------------
-- Initialize bh_current_state for all instruments (all-zero default)
-- ---------------------------------------------------------------------------
INSERT INTO bh_current_state (instrument_id, tf_score)
SELECT id, 0 FROM instruments ON CONFLICT DO NOTHING;
