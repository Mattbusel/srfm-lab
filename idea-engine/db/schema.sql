-- ============================================================
-- idea_engine.db  —  Idea Automation Engine schema
-- SQLite 3.x  |  PRAGMA foreign_keys = ON required at runtime
-- ============================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------
-- patterns
-- Mined statistical patterns from backtests / live data
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS patterns (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    source              TEXT    NOT NULL,           -- e.g. 'backtest', 'live', 'walk_forward'
    miner               TEXT    NOT NULL,           -- miner class that produced this
    pattern_type        TEXT    NOT NULL,           -- 'time_of_day', 'regime_cluster', 'bh_physics', 'drawdown'
    label               TEXT    NOT NULL,           -- human-readable label
    description         TEXT,
    feature_json        TEXT,                       -- JSON dict of feature values that define this pattern
    window_start        TEXT,                       -- ISO datetime or NULL
    window_end          TEXT,
    instruments         TEXT,                       -- JSON list of symbols
    sample_size         INTEGER NOT NULL DEFAULT 0,
    p_value             REAL,
    effect_size         REAL,
    effect_size_type    TEXT,                       -- 'cohens_d', 'cliffs_delta', 'eta_squared'
    win_rate            REAL,
    avg_pnl             REAL,
    avg_pnl_baseline    REAL,
    sharpe              REAL,
    max_dd              REAL,
    profit_factor       REAL,
    confidence          REAL,                       -- 0-1, overall confidence after bootstrap
    status              TEXT    NOT NULL DEFAULT 'new',  -- new | confirmed | rejected | promoted
    tags                TEXT,                       -- comma-separated tags
    genome_id           INTEGER REFERENCES genomes(id) ON DELETE SET NULL,
    hypothesis_id       INTEGER REFERENCES hypotheses(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_patterns_type    ON patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_status  ON patterns(status);
CREATE INDEX IF NOT EXISTS idx_patterns_miner   ON patterns(miner);
CREATE INDEX IF NOT EXISTS idx_patterns_pvalue  ON patterns(p_value);
CREATE INDEX IF NOT EXISTS idx_patterns_created ON patterns(created_at);

-- ------------------------------------------------------------
-- hypotheses
-- Auto-generated or human-authored hypotheses derived from patterns
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hypotheses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    title           TEXT    NOT NULL,
    body            TEXT    NOT NULL,
    rationale       TEXT,
    prior_prob      REAL    NOT NULL DEFAULT 0.5,
    posterior_prob  REAL,
    status          TEXT    NOT NULL DEFAULT 'open',  -- open | testing | confirmed | refuted | parked
    priority        INTEGER NOT NULL DEFAULT 5,       -- 1 (highest) – 10 (lowest)
    source_pattern_ids TEXT,                          -- JSON list of pattern ids
    experiment_ids  TEXT,                             -- JSON list of experiment ids
    tags            TEXT,
    genome_id       INTEGER REFERENCES genomes(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_hypotheses_status   ON hypotheses(status);
CREATE INDEX IF NOT EXISTS idx_hypotheses_priority ON hypotheses(priority);

-- ------------------------------------------------------------
-- genomes
-- Parameter sets (strategy genomes) that encode a trading strategy
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS genomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    name            TEXT    NOT NULL,
    parent_id       INTEGER REFERENCES genomes(id) ON DELETE SET NULL,
    generation      INTEGER NOT NULL DEFAULT 0,
    params_json     TEXT    NOT NULL,               -- full parameter dict as JSON
    fitness_sharpe  REAL,
    fitness_cagr    REAL,
    fitness_dd      REAL,
    fitness_pf      REAL,
    composite_score REAL,
    is_oos_sharpe   REAL,
    oos_degradation REAL,
    source          TEXT    NOT NULL DEFAULT 'manual',  -- 'manual', 'evolved', 'imported'
    status          TEXT    NOT NULL DEFAULT 'candidate',  -- candidate | active | archived
    tags            TEXT
);

CREATE INDEX IF NOT EXISTS idx_genomes_score  ON genomes(composite_score);
CREATE INDEX IF NOT EXISTS idx_genomes_status ON genomes(status);

-- ------------------------------------------------------------
-- shadow_variants
-- Shadow-test variants of a genome running in paper-mode
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS shadow_variants (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    genome_id       INTEGER NOT NULL REFERENCES genomes(id) ON DELETE CASCADE,
    label           TEXT    NOT NULL,
    params_delta    TEXT,       -- JSON of params that differ from parent genome
    start_ts        TEXT,
    end_ts          TEXT,
    realized_pnl    REAL,
    realized_sharpe REAL,
    realized_dd     REAL,
    trade_count     INTEGER NOT NULL DEFAULT 0,
    status          TEXT    NOT NULL DEFAULT 'running',  -- running | completed | abandoned
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_shadow_genome  ON shadow_variants(genome_id);
CREATE INDEX IF NOT EXISTS idx_shadow_status  ON shadow_variants(status);

-- ------------------------------------------------------------
-- experiments
-- A/B and controlled experiments linking hypotheses to outcomes
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    hypothesis_id   INTEGER REFERENCES hypotheses(id) ON DELETE SET NULL,
    genome_id       INTEGER REFERENCES genomes(id) ON DELETE SET NULL,
    name            TEXT    NOT NULL,
    design          TEXT    NOT NULL,           -- 'backtest', 'paper', 'live', 'shadow_ab'
    start_ts        TEXT,
    end_ts          TEXT,
    config_json     TEXT,                       -- full config snapshot
    result_json     TEXT,                       -- result summary as JSON
    p_value         REAL,
    effect_size     REAL,
    outcome         TEXT,                       -- 'confirmed', 'refuted', 'inconclusive', 'pending'
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_experiments_hypothesis ON experiments(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_experiments_outcome    ON experiments(outcome);

-- ------------------------------------------------------------
-- counterfactuals
-- "What if" scenarios evaluated against a base experiment
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS counterfactuals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    base_experiment_id  INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
    base_pattern_id     INTEGER REFERENCES patterns(id) ON DELETE SET NULL,
    description         TEXT    NOT NULL,
    intervention_json   TEXT    NOT NULL,       -- what was changed
    result_json         TEXT,                   -- outcome of the counterfactual
    delta_sharpe        REAL,
    delta_cagr          REAL,
    delta_dd            REAL,
    notes               TEXT
);

CREATE INDEX IF NOT EXISTS idx_cf_experiment ON counterfactuals(base_experiment_id);

-- ------------------------------------------------------------
-- causal_edges
-- Directed causal graph edges between patterns / factors
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS causal_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    cause_type      TEXT    NOT NULL,   -- 'pattern', 'hypothesis', 'factor'
    cause_id        INTEGER NOT NULL,
    effect_type     TEXT    NOT NULL,
    effect_id       INTEGER NOT NULL,
    method          TEXT    NOT NULL DEFAULT 'granger',  -- 'granger', 'pc', 'manual', 'llm'
    strength        REAL,               -- 0-1
    p_value         REAL,
    lag_bars        INTEGER,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_causal_cause  ON causal_edges(cause_type, cause_id);
CREATE INDEX IF NOT EXISTS idx_causal_effect ON causal_edges(effect_type, effect_id);

-- ------------------------------------------------------------
-- academic_papers
-- Papers ingested from arXiv / PDFs for idea seeding
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS academic_papers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ingested_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    arxiv_id        TEXT    UNIQUE,
    doi             TEXT,
    title           TEXT    NOT NULL,
    authors         TEXT,
    abstract        TEXT,
    year            INTEGER,
    venue           TEXT,
    url             TEXT,
    local_path      TEXT,
    embedding_json  TEXT,               -- stored embedding vector as JSON array
    relevance_score REAL,
    tags            TEXT,
    processed       INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_papers_arxiv     ON academic_papers(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_papers_relevance ON academic_papers(relevance_score);

-- ------------------------------------------------------------
-- academic_ideas
-- Specific implementable ideas extracted from papers
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS academic_ideas (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    paper_id        INTEGER NOT NULL REFERENCES academic_papers(id) ON DELETE CASCADE,
    title           TEXT    NOT NULL,
    description     TEXT    NOT NULL,
    applicable_to   TEXT,               -- comma-separated: 'bh_physics', 'regime', 'portfolio', etc.
    implementation_notes TEXT,
    priority        INTEGER NOT NULL DEFAULT 5,
    status          TEXT    NOT NULL DEFAULT 'backlog',  -- backlog | in_progress | implemented | rejected
    hypothesis_id   INTEGER REFERENCES hypotheses(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_aideas_paper  ON academic_ideas(paper_id);
CREATE INDEX IF NOT EXISTS idx_aideas_status ON academic_ideas(status);

-- ------------------------------------------------------------
-- genealogy_nodes
-- Nodes in the idea genealogy graph (patterns, hypotheses, genomes, experiments)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS genealogy_nodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    node_type       TEXT    NOT NULL,   -- 'pattern', 'hypothesis', 'genome', 'experiment', 'paper_idea'
    ref_id          INTEGER NOT NULL,   -- FK to the type's table (by convention)
    label           TEXT    NOT NULL,
    metadata_json   TEXT,
    x_pos           REAL,               -- layout hint for visualization
    y_pos           REAL
);

CREATE INDEX IF NOT EXISTS idx_gnodes_type ON genealogy_nodes(node_type, ref_id);

-- ------------------------------------------------------------
-- genealogy_edges
-- Directed edges in the genealogy graph
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS genealogy_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    src_node_id     INTEGER NOT NULL REFERENCES genealogy_nodes(id) ON DELETE CASCADE,
    dst_node_id     INTEGER NOT NULL REFERENCES genealogy_nodes(id) ON DELETE CASCADE,
    edge_type       TEXT    NOT NULL DEFAULT 'derived_from',  -- derived_from | tested_by | evolved_to | inspired_by
    weight          REAL    NOT NULL DEFAULT 1.0,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_gedges_src ON genealogy_edges(src_node_id);
CREATE INDEX IF NOT EXISTS idx_gedges_dst ON genealogy_edges(dst_node_id);

-- ------------------------------------------------------------
-- narrative_notes
-- Free-form notes attached to any entity
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS narrative_notes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    entity_type     TEXT    NOT NULL,
    entity_id       INTEGER NOT NULL,
    author          TEXT    NOT NULL DEFAULT 'system',
    body            TEXT    NOT NULL,
    tags            TEXT
);

CREATE INDEX IF NOT EXISTS idx_notes_entity ON narrative_notes(entity_type, entity_id);

-- ------------------------------------------------------------
-- event_log
-- Append-only audit / event stream for pipeline runs and state transitions
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS event_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    event_type      TEXT    NOT NULL,   -- 'pipeline_run', 'pattern_created', 'hypothesis_promoted', etc.
    entity_type     TEXT,
    entity_id       INTEGER,
    actor           TEXT    NOT NULL DEFAULT 'system',
    payload_json    TEXT,
    severity        TEXT    NOT NULL DEFAULT 'info'   -- debug | info | warning | error
);

CREATE INDEX IF NOT EXISTS idx_events_type   ON event_log(event_type);
CREATE INDEX IF NOT EXISTS idx_events_ts     ON event_log(ts);
CREATE INDEX IF NOT EXISTS idx_events_entity ON event_log(entity_type, entity_id);
