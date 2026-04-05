-- Genealogy schema extension for idea_engine.db
-- Run once: sqlite3 idea_engine.db < schema_extension.sql

CREATE TABLE IF NOT EXISTS genealogy_nodes (
    genome_id       INTEGER PRIMARY KEY,
    island          TEXT    NOT NULL,
    generation      INTEGER NOT NULL,
    params_json     TEXT    NOT NULL,
    fitness         REAL,
    mutation_ops    TEXT,  -- JSON list of mutation operation names
    is_hall_of_fame INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_genealogy_nodes_island
    ON genealogy_nodes (island);

CREATE INDEX IF NOT EXISTS idx_genealogy_nodes_fitness
    ON genealogy_nodes (fitness DESC);

CREATE TABLE IF NOT EXISTS genealogy_edges (
    child_id        INTEGER NOT NULL REFERENCES genealogy_nodes(genome_id),
    parent_id       INTEGER NOT NULL REFERENCES genealogy_nodes(genome_id),
    PRIMARY KEY (child_id, parent_id)
);

CREATE INDEX IF NOT EXISTS idx_genealogy_edges_child
    ON genealogy_edges (child_id);

CREATE INDEX IF NOT EXISTS idx_genealogy_edges_parent
    ON genealogy_edges (parent_id);

CREATE TABLE IF NOT EXISTS evolution_tracking (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    island          TEXT    NOT NULL,
    generation      INTEGER NOT NULL,
    pop_size        INTEGER NOT NULL,
    best_fitness    REAL,
    mean_fitness    REAL,
    diversity_index REAL,
    event           TEXT,   -- 'migration', 'stagnation', 'convergence', etc.
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_evolution_tracking_island_gen
    ON evolution_tracking (island, generation);
