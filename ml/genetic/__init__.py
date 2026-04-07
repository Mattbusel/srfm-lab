"""
Genetic Algorithm Strategy Optimizer + GP-based Alpha Signal Discovery.

Original GA module: parameter optimization for trading strategies.
New GP sub-system: symbolic expression trees for discovering alpha signals.
"""

# -- GP sub-system (symbolic expression trees / signal discovery)
from .expression_tree import (
    NodeType,
    ExpressionNode,
    ExpressionTree,
    TreeGenerator,
    FunctionArity,
    ALL_FUNCTION_NAMES,
    BINARY_FUNCTION_NAMES,
    UNARY_FUNCTION_NAMES,
    ALL_TERMINALS,
    SIGNAL_NAMES,
    RAW_FEATURES,
    FUNCTION_ARITY,
)
from .gp_operators import (
    fitness_proportional_select,
    tournament_select,
    lexicase_select,
    subtree_crossover,
    point_mutation,
    subtree_mutation,
    hoist_mutation,
    apply_random_mutation,
)
from .gp_engine import (
    Individual,
    GPConfig,
    GPEngine,
)
from .signal_validator import (
    SignalValidator,
    ValidationResult,
)
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
