"""
SRFMResearch — Julia quantitative research suite for the SRFM trading lab.

Submodules:
  BHPhysics     — Black-Hole physics engine (mass dynamics, multi-TF backtest)
  Stochastic    — Stochastic processes (GBM, GARCH, Heston, OU, Hawkes, MJD)
  SRFMStats     — Statistical toolkit (performance metrics, hypothesis tests, distributions)
  SRFMOptimization — Portfolio & parameter optimization (MVO, HRP, BL, walk-forward)
  SRFMViz       — Research-grade visualization (equity curves, heatmaps, MC fans)
  Bayesian      — Bayesian inference and probabilistic models (Turing.jl)
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

@reexport using .BHPhysics
@reexport using .Stochastic
@reexport using .SRFMStats
@reexport using .SRFMOptimization
@reexport using .SRFMViz
@reexport using .Bayesian
@reexport using .FactorModel

end # module SRFMResearch
