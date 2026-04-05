-- Shadow Runner schema extension for idea_engine.db
-- Applied automatically by ShadowRunner._ensure_schema()

CREATE TABLE IF NOT EXISTS shadow_runs (
    shadow_id           TEXT    NOT NULL,
    genome_id           INTEGER NOT NULL,
    ts                  TEXT    NOT NULL,
    symbol              TEXT    NOT NULL,
    virtual_qty         REAL    NOT NULL DEFAULT 0.0,
    virtual_equity      REAL    NOT NULL DEFAULT 0.0,
    signal              TEXT,
    shadow_state_json   TEXT,   -- serialised ShadowState for checkpoint/restore
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    PRIMARY KEY (shadow_id, ts, symbol)
);

CREATE INDEX IF NOT EXISTS idx_sr_genome  ON shadow_runs (genome_id);
CREATE INDEX IF NOT EXISTS idx_sr_ts      ON shadow_runs (ts);
CREATE INDEX IF NOT EXISTS idx_sr_symbol  ON shadow_runs (symbol);

CREATE TABLE IF NOT EXISTS shadow_comparisons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    shadow_id       TEXT    NOT NULL,
    genome_id       INTEGER NOT NULL,
    period_days     INTEGER NOT NULL,
    shadow_return   REAL,
    live_return     REAL,
    alpha           REAL,
    promoted        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_sc_shadow  ON shadow_comparisons (shadow_id);
CREATE INDEX IF NOT EXISTS idx_sc_alpha   ON shadow_comparisons (alpha DESC);
CREATE INDEX IF NOT EXISTS idx_sc_promo   ON shadow_comparisons (promoted, created_at DESC);
