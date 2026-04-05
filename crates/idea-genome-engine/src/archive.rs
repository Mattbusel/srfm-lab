/// Hall-of-fame archive: keeps the best unique genomes across all generations.

use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{Connection, params};

use crate::genome::Genome;

// ---------------------------------------------------------------------------
// Archive
// ---------------------------------------------------------------------------

/// Maintains a fixed-capacity collection of elite genomes discovered across
/// the entire evolutionary run.  The archive is updated each generation and
/// can be persisted to / reloaded from an SQLite database.
#[derive(Debug)]
pub struct Archive {
    /// Maximum number of genomes stored in the archive.
    pub capacity: usize,
    /// Archived genomes, always sorted by Sharpe descending.
    pub genomes: Vec<Genome>,
}

impl Archive {
    /// Create an empty archive with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            genomes: Vec::with_capacity(capacity),
        }
    }

    // ------------------------------------------------------------------
    // Update
    // ------------------------------------------------------------------

    /// Attempt to add `genome` to the archive.
    ///
    /// The genome is accepted if any of the following hold:
    ///   (a) The archive has fewer genomes than `capacity`.
    ///   (b) `genome` dominates at least one existing archive member.
    ///   (c) `genome` has a higher Sharpe than the worst member.
    ///
    /// After insertion the archive is kept at most `capacity` genomes by
    /// removing the worst genome (by Sharpe).
    pub fn update(&mut self, genome: &Genome) {
        if genome.fitness.is_none() {
            return;
        }

        // Skip exact duplicates by genome id.
        if self.genomes.iter().any(|g| g.id == genome.id) {
            return;
        }

        let should_add = if self.genomes.len() < self.capacity {
            true
        } else {
            // Add if it dominates any member or beats the worst by Sharpe.
            let dominates_any = self.genomes.iter().any(|g| genome.dominates(g));
            let better_than_worst = self
                .genomes
                .last()
                .map(|w| genome.sharpe() > w.sharpe())
                .unwrap_or(true);
            dominates_any || better_than_worst
        };

        if !should_add {
            return;
        }

        self.genomes.push(genome.clone());

        // Re-sort descending by Sharpe.
        self.genomes.sort_by(|a, b| {
            b.sharpe()
                .partial_cmp(&a.sharpe())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Trim to capacity.
        self.genomes.truncate(self.capacity);
    }

    /// Bulk-update from a slice of genomes (convenience wrapper).
    pub fn update_from_slice(&mut self, genomes: &[Genome]) {
        for g in genomes {
            self.update(g);
        }
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Best genome by Sharpe, or `None` when the archive is empty.
    pub fn best_by_sharpe(&self) -> Option<&Genome> {
        self.genomes.first()
    }

    /// Number of stored genomes.
    pub fn len(&self) -> usize {
        self.genomes.len()
    }

    /// Returns `true` when the archive contains no genomes.
    pub fn is_empty(&self) -> bool {
        self.genomes.is_empty()
    }

    // ------------------------------------------------------------------
    // Persistence
    // ------------------------------------------------------------------

    /// Write all archived genomes to the `genomes` table in `db_path`.
    ///
    /// The table is created if it does not exist.  Existing rows with the
    /// same `genome_id` are replaced.
    pub fn save_to_db(&self, db_path: &str) -> Result<()> {
        let conn = Connection::open(db_path)
            .with_context(|| format!("opening SQLite database at {}", db_path))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS genomes (
                genome_id       TEXT PRIMARY KEY,
                generation      INTEGER NOT NULL,
                parameters_json TEXT NOT NULL,
                fitness_json    TEXT,
                parent_ids_json TEXT NOT NULL,
                sharpe          REAL,
                calmar          REAL,
                max_dd          REAL,
                win_rate        REAL,
                profit_factor   REAL,
                n_trades        INTEGER,
                is_oos_spread   REAL,
                created_at      TEXT NOT NULL
            );",
        )
        .context("creating genomes table")?;

        let now = Utc::now().to_rfc3339();

        for genome in &self.genomes {
            let params_json = serde_json::to_string(&genome.parameters)
                .context("serialising parameters")?;
            let fitness_json = genome
                .fitness
                .as_ref()
                .map(|f| serde_json::to_string(f))
                .transpose()
                .context("serialising fitness")?;
            let parent_ids_json = serde_json::to_string(&genome.parent_ids)
                .context("serialising parent_ids")?;

            let (sharpe, calmar, max_dd, win_rate, pf, n_trades, spread) =
                if let Some(ref f) = genome.fitness {
                    (
                        Some(f.sharpe),
                        Some(f.calmar),
                        Some(f.max_dd),
                        Some(f.win_rate),
                        Some(f.profit_factor),
                        Some(f.n_trades as i64),
                        Some(f.is_oos_spread),
                    )
                } else {
                    (None, None, None, None, None, None, None)
                };

            conn.execute(
                "INSERT OR REPLACE INTO genomes (
                    genome_id, generation, parameters_json, fitness_json,
                    parent_ids_json, sharpe, calmar, max_dd, win_rate,
                    profit_factor, n_trades, is_oos_spread, created_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                params![
                    genome.id,
                    genome.generation as i64,
                    params_json,
                    fitness_json,
                    parent_ids_json,
                    sharpe,
                    calmar,
                    max_dd,
                    win_rate,
                    pf,
                    n_trades,
                    spread,
                    now,
                ],
            )
            .context("inserting genome row")?;
        }

        Ok(())
    }

    /// Load archived genomes from the `genomes` table in `db_path`.
    ///
    /// Returns an archive with `capacity` filled from the top-`capacity` rows
    /// ordered by Sharpe descending.
    pub fn load_from_db(db_path: &str, capacity: usize) -> Result<Self> {
        let conn = Connection::open(db_path)
            .with_context(|| format!("opening SQLite database at {}", db_path))?;

        // Return an empty archive if the table does not yet exist.
        let table_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='genomes'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0)
            > 0;

        if !table_exists {
            return Ok(Archive::new(capacity));
        }

        let mut stmt = conn
            .prepare(
                "SELECT genome_id, generation, parameters_json, fitness_json, parent_ids_json
                 FROM genomes
                 ORDER BY sharpe DESC
                 LIMIT ?1",
            )
            .context("preparing SELECT statement")?;

        let rows = stmt
            .query_map(params![capacity as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,   // genome_id
                    row.get::<_, i64>(1)?,       // generation
                    row.get::<_, String>(2)?,    // parameters_json
                    row.get::<_, Option<String>>(3)?, // fitness_json
                    row.get::<_, String>(4)?,    // parent_ids_json
                ))
            })
            .context("querying genomes table")?;

        let mut genomes: Vec<Genome> = Vec::new();
        for row in rows {
            let (id, generation, params_json, fitness_json, parent_ids_json) =
                row.context("reading genome row")?;

            let parameters: Vec<f64> = serde_json::from_str(&params_json)
                .context("deserialising parameters")?;
            let fitness = fitness_json
                .as_deref()
                .map(serde_json::from_str)
                .transpose()
                .context("deserialising fitness")?;
            let parent_ids: Vec<String> = serde_json::from_str(&parent_ids_json)
                .context("deserialising parent_ids")?;

            let mut genome = Genome::from_parameters(parameters, generation as u32, parent_ids);
            genome.id = id;
            genome.fitness = fitness;
            genomes.push(genome);
        }

        Ok(Archive { capacity, genomes })
    }

    /// Export the archive to a pretty-printed JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.genomes).expect("Archive JSON serialisation is infallible")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use crate::fitness::{EvaluatorConfig, FitnessEvaluator};
    use crate::population::Population;

    fn make_evaluated_genomes(n: usize, seed: u64) -> Vec<Genome> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut pop = Population::new(n, &mut rng);
        let eval = FitnessEvaluator::new(EvaluatorConfig {
            dry_run: true,
            ..Default::default()
        });
        pop.evaluate_all_parallel(&eval);
        pop.genomes
    }

    #[test]
    fn archive_respects_capacity() {
        let genomes = make_evaluated_genomes(100, 1);
        let mut archive = Archive::new(50);
        for g in &genomes {
            archive.update(g);
        }
        assert!(archive.len() <= 50);
    }

    #[test]
    fn archive_sorted_descending_sharpe() {
        let genomes = make_evaluated_genomes(30, 2);
        let mut archive = Archive::new(20);
        for g in &genomes {
            archive.update(g);
        }
        for i in 1..archive.len() {
            assert!(
                archive.genomes[i - 1].sharpe() >= archive.genomes[i].sharpe(),
                "Archive not sorted by Sharpe"
            );
        }
    }

    #[test]
    fn archive_db_round_trip() {
        let genomes = make_evaluated_genomes(10, 3);
        let mut archive = Archive::new(10);
        for g in &genomes {
            archive.update(g);
        }

        let tmp = std::env::temp_dir().join("test_archive.db");
        let db_path = tmp.to_str().unwrap();

        archive.save_to_db(db_path).expect("save_to_db failed");
        let loaded = Archive::load_from_db(db_path, 10).expect("load_from_db failed");

        assert_eq!(loaded.len(), archive.len(), "loaded archive has wrong size");

        // Verify first genome IDs match.
        if let (Some(a), Some(b)) = (archive.best_by_sharpe(), loaded.best_by_sharpe()) {
            assert_eq!(a.id, b.id, "best genome id should match after round-trip");
        }

        let _ = std::fs::remove_file(db_path);
    }
}
