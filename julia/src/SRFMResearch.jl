"""
SRFMResearch — Julia quantitative research suite for the SRFM trading lab.

Submodules:
  BHPhysics        — Black-Hole physics engine (mass dynamics, multi-TF backtest)
  Stochastic       — Stochastic processes (GBM, GARCH, Heston, OU, Hawkes, MJD)
  SRFMStats        — Statistical toolkit (performance metrics, hypothesis tests, distributions)
  SRFMOptimization — Portfolio & parameter optimization (MVO, HRP, BL, walk-forward)
  SRFMViz          — Research-grade visualization (equity curves, heatmaps, MC fans)
  Bayesian         — Bayesian inference and probabilistic models (Turing.jl)
  FactorModel      — Factor model construction and analysis
  OptimalExecution — Almgren-Chriss, Obizhaeva-Wang, IS minimization, TCA
  InterestRates    — Vasicek, CIR, Hull-White, HJM, LMM, yield curve bootstrap
  VolatilitySurface — SVI, SABR, local vol, surface interpolation, arbitrage checks
  NetworkAnalysis  — Correlation networks, MST, PMFG, community detection, systemic risk
  MachineLearning  — LSTM, gradient boosting, GP regression, online learning, CV
"""
module SRFMResearch

using Reexport

include("BHPhysics.jl")
include("Stochastic.jl")
include("Statistics.jl")
include("Optimization.jl")
include("Visualization.jl")
include("Bayesian.jl")
include("FactorModel.jl")
include("OptimalExecution.jl")
include("InterestRates.jl")
include("VolatilitySurface.jl")
include("NetworkAnalysis.jl")
include("MachineLearning.jl")

@reexport using .BHPhysics
@reexport using .Stochastic
@reexport using .SRFMStats
@reexport using .SRFMOptimization
@reexport using .SRFMViz
@reexport using .Bayesian
@reexport using .FactorModel
@reexport using .OptimalExecution
@reexport using .InterestRates
@reexport using .VolatilitySurface
@reexport using .NetworkAnalysis
@reexport using .MachineLearning

end # module SRFMResearch
