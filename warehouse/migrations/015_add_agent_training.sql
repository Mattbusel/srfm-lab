-- =============================================================================
-- Migration 015: Add agent training tables
-- Applied: 2026-04-04
-- =============================================================================
-- UP

-- training_runs: Metadata for each RL/ML agent training run.
CREATE TABLE IF NOT EXISTS training_runs (
    id              SERIAL          PRIMARY KEY,
    run_id          INTEGER         REFERENCES strategy_runs (id) ON DELETE SET NULL,
    agent_type      VARCHAR(40)     NOT NULL CHECK (agent_type IN (
                        'ppo', 'sac', 'td3', 'ddpg', 'dqn', 'recurrent_ppo',
                        'transformer', 'bc', 'iql', 'cql', 'custom'
                    )),
    experiment_name VARCHAR(100)    NOT NULL,
    started_at      TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at     TIMESTAMPTZ,
    status          VARCHAR(20)     NOT NULL DEFAULT 'running'
                                    CHECK (status IN ('running', 'complete', 'failed', 'cancelled')),
    total_timesteps BIGINT,
    trained_steps   BIGINT          NOT NULL DEFAULT 0,
    n_envs          INTEGER         NOT NULL DEFAULT 1,
    seed            INTEGER,
    instrument_id   INTEGER         REFERENCES instruments (id),
    universe        VARCHAR(100),
    obs_space       JSONB           NOT NULL DEFAULT '{}',   -- observation space spec
    act_space       JSONB           NOT NULL DEFAULT '{}',   -- action space spec
    hyperparams     JSONB           NOT NULL DEFAULT '{}',
    framework       VARCHAR(20)     DEFAULT 'stable_baselines3'
                                    CHECK (framework IN (
                                        'stable_baselines3', 'rllib', 'cleanrl',
                                        'tianshou', 'custom'
                                    )),
    git_hash        VARCHAR(40),
    checkpoint_dir  TEXT,           -- path to saved weights directory
    best_eval_reward DECIMAL(18,8),
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_training_runs_status
    ON training_runs (status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_runs_agent
    ON training_runs (agent_type, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_runs_experiment
    ON training_runs (experiment_name, started_at DESC);

-- episode_results: Per-episode return, steps, and loss during training.
CREATE TABLE IF NOT EXISTS episode_results (
    id              SERIAL          PRIMARY KEY,
    training_run_id INTEGER         NOT NULL REFERENCES training_runs (id) ON DELETE CASCADE,
    episode         INTEGER         NOT NULL,
    timestep        BIGINT          NOT NULL,
    env_index       INTEGER         NOT NULL DEFAULT 0,
    episode_return  DECIMAL(18,8),
    episode_length  INTEGER,
    policy_loss     DECIMAL(18,8),
    value_loss      DECIMAL(18,8),
    entropy_loss    DECIMAL(18,8),
    approx_kl       DECIMAL(18,8),
    clip_fraction   DECIMAL(10,6),
    explained_var   DECIMAL(10,6),
    learning_rate   DECIMAL(18,10),
    sharpe_ep       DECIMAL(10,4),  -- rolling Sharpe for this episode
    max_drawdown_ep DECIMAL(10,6),
    trade_count_ep  INTEGER,
    recorded_at     TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_episode_results_run
    ON episode_results (training_run_id, episode DESC);
CREATE INDEX IF NOT EXISTS idx_episode_results_timestep
    ON episode_results (training_run_id, timestep DESC);

-- eval_results: Evaluation metrics per checkpoint on held-out eval environment.
CREATE TABLE IF NOT EXISTS eval_results (
    id              SERIAL          PRIMARY KEY,
    training_run_id INTEGER         NOT NULL REFERENCES training_runs (id) ON DELETE CASCADE,
    checkpoint_step BIGINT          NOT NULL,
    evaluated_at    TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    n_eval_episodes INTEGER         NOT NULL DEFAULT 10,
    mean_reward     DECIMAL(18,8),
    std_reward      DECIMAL(18,8),
    min_reward      DECIMAL(18,8),
    max_reward      DECIMAL(18,8),
    mean_ep_length  DECIMAL(10,2),
    sharpe          DECIMAL(10,4),
    cagr            DECIMAL(10,6),
    max_drawdown    DECIMAL(10,6),
    sortino         DECIMAL(10,4),
    win_rate        DECIMAL(10,6),
    mean_trade_count DECIMAL(10,2),
    is_best         BOOLEAN         NOT NULL DEFAULT FALSE,
    eval_env        VARCHAR(40),    -- eval environment identifier
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_eval_results_run
    ON eval_results (training_run_id, checkpoint_step DESC);
CREATE INDEX IF NOT EXISTS idx_eval_results_best
    ON eval_results (training_run_id, is_best) WHERE is_best = TRUE;

-- agent_weights: Pointer to saved weight files / checkpoints.
CREATE TABLE IF NOT EXISTS agent_weights (
    id              SERIAL          PRIMARY KEY,
    training_run_id INTEGER         NOT NULL REFERENCES training_runs (id) ON DELETE CASCADE,
    eval_result_id  INTEGER         REFERENCES eval_results (id) ON DELETE SET NULL,
    checkpoint_step BIGINT          NOT NULL,
    saved_at        TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    file_path       TEXT            NOT NULL,   -- relative path from SRFM_ROOT
    file_size_bytes BIGINT,
    checksum_sha256 VARCHAR(64),
    framework       VARCHAR(20),
    is_best         BOOLEAN         NOT NULL DEFAULT FALSE,
    is_deployed     BOOLEAN         NOT NULL DEFAULT FALSE,
    deployed_at     TIMESTAMPTZ,
    tags            JSONB           NOT NULL DEFAULT '[]',
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_agent_weights_run
    ON agent_weights (training_run_id, checkpoint_step DESC);
CREATE INDEX IF NOT EXISTS idx_agent_weights_best
    ON agent_weights (training_run_id, is_best) WHERE is_best = TRUE;
CREATE INDEX IF NOT EXISTS idx_agent_weights_deployed
    ON agent_weights (is_deployed, deployed_at DESC) WHERE is_deployed = TRUE;

INSERT INTO _schema_migrations (version, name) VALUES (15, 'add_agent_training')
ON CONFLICT (version) DO NOTHING;

-- DOWN
-- DROP TABLE IF EXISTS agent_weights;
-- DROP TABLE IF EXISTS eval_results;
-- DROP TABLE IF EXISTS episode_results;
-- DROP TABLE IF EXISTS training_runs;
-- DELETE FROM _schema_migrations WHERE version = 15;
