"""Genetic Algorithm Strategy Optimizer."""
from .genome import (
    StrategyGenome, GenomeFactory, Chromosome, Gene,
    ParamRange, ParamType,
    momentum_strategy_params, mean_reversion_strategy_params,
    pairs_trading_params, ml_hyperparameter_params, portfolio_weights_params,
    MutationOperator, CrossoverOperator,
)
from .population import (
    Population, PopulationConfig, HallOfFame,
    SelectionOperator, FitnessSharing,
    DiversityMetrics, PopulationStats,
)
from .fitness import (
    FitnessEvaluator, FitnessConfig, MultiObjectiveFitnessEvaluator,
    BacktestResult, ParetoAnalysis, OverfittingAnalyzer,
    compute_sharpe, compute_sortino, compute_calmar, compute_max_drawdown,
    compute_profit_factor, compute_win_rate,
)
from .operators import (
    AdaptiveMutationRate, AdaptiveMutationConfig,
    AdaptiveOperatorSelection, NichePreservation,
    IslandModel, IslandConfig, MigrationConfig,
    RestartStrategy, ReproductionOperator, OperatorConfig,
)
from .evolution import (
    GeneticAlgorithm, EvolutionConfig, EvolutionResult,
    evolve, multi_run_evolution, MultiRunStats,
)
from .coevolution import (
    CompetitiveCoevolution, CoevolutionConfig, CoevolutionResult,
    CooperativeCoevolution, StrategyEnsembleCoevolution,
    MarketMakerStrategy, MarketTakerStrategy,
)
from .visualization import (
    PlotConfig, EvolutionDashboard, FitnessLandscapeVisualizer,
    ParetoFrontVisualizer, ParameterConvergenceVisualizer,
    GenealogyTracker, ASCIIPlotter,
)
