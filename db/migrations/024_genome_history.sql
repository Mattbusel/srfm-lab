-- Migration 024: IAE genome evolution history
-- Tracks every genome evaluated during IAE optimization cycles

-- UP

CREATE TABLE IF NOT EXISTS genome_history (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  genome_id         TEXT    NOT NULL UNIQUE,
  -- Lineage
  generation        INTEGER NOT NULL,
  parent_genome_id  TEXT,
  parent2_genome_id TEXT,   -- second parent for crossover
  creation_method   TEXT    NOT NULL CHECK(creation_method IN (
                      'RANDOM','MUTATION','CROSSOVER','ELITE','MANUAL','INJECTION'
                    )),
  mutation_type     TEXT,   -- POINT, SWAP, GAUSSIAN, etc.
  -- Genome encoding (JSON)
  genome_json       TEXT    NOT NULL,
  -- Decoded parameters (JSON)
  params_json       TEXT    NOT NULL,
  -- Fitness evaluation
  fitness           REAL,
  fitness_components_json TEXT,  -- JSON: component -> value
  -- In-sample performance
  is_sharpe         REAL,
  is_cagr           REAL,
  is_max_drawdown   REAL,
  is_win_rate       REAL,
  is_num_trades     INTEGER,
  is_period_start   TEXT,
  is_period_end     TEXT,
  -- Out-of-sample performance (if evaluated)
  oos_sharpe        REAL,
  oos_cagr          REAL,
  oos_max_drawdown  REAL,
  oos_win_rate      REAL,
  oos_num_trades    INTEGER,
  oos_period_start  TEXT,
  oos_period_end    TEXT,
  -- Overfitting penalty
  overfit_penalty   REAL,
  adjusted_fitness  REAL,
  -- Population management
  is_elite          INTEGER NOT NULL DEFAULT 0 CHECK(is_elite IN (0,1)),
  was_deployed      INTEGER NOT NULL DEFAULT 0 CHECK(was_deployed IN (0,1)),
  deployed_at       TEXT,
  -- Evaluation metadata
  eval_duration_sec REAL,
  eval_error        TEXT,
  eval_node         TEXT,  -- which compute node ran this eval
  -- IAE cycle context
  cycle_id          TEXT,
  population_size   INTEGER,
  -- Constraints passed
  constraints_passed INTEGER NOT NULL DEFAULT 1 CHECK(constraints_passed IN (0,1)),
  constraint_violations TEXT,
  created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_genome_history_id         ON genome_history(genome_id);
CREATE INDEX IF NOT EXISTS idx_genome_history_generation ON genome_history(generation);
CREATE INDEX IF NOT EXISTS idx_genome_history_fitness    ON genome_history(fitness);
CREATE INDEX IF NOT EXISTS idx_genome_history_cycle      ON genome_history(cycle_id);
CREATE INDEX IF NOT EXISTS idx_genome_history_elite      ON genome_history(is_elite);

-- DOWN

DROP INDEX IF EXISTS idx_genome_history_elite;
DROP INDEX IF EXISTS idx_genome_history_cycle;
DROP INDEX IF EXISTS idx_genome_history_fitness;
DROP INDEX IF EXISTS idx_genome_history_generation;
DROP INDEX IF EXISTS idx_genome_history_id;
DROP TABLE IF EXISTS genome_history;
