"""
NeuroSDE.jl — Neural Stochastic Differential Equations Framework

Module 2 of Project AETERNUS (SRFM-Lab).

This package implements neural SDE models for financial volatility modelling,
latent state inference, and market calibration. The core philosophy is to
combine the interpretability of classical SDE models (Heston, SABR, rough vol)
with the expressiveness of neural networks parameterising the drift and
diffusion functions.

Architecture:
  - neural_networks.jl   : MLP drift/diffusion nets, weight init, serialisation
  - sde_solvers.jl       : EM, Milstein, RK4.5 solvers, batch simulation
  - adjoint_sensitivity.jl : continuous/discrete adjoints, variance reduction
  - volatility_models.jl : NeuralHeston, NeuralSABR, RoughVol, JumpDiffusion
  - calibration.jl       : MLE, GMM, Carr-Madan FFT, training loops
  - latent_dynamics.jl   : GRU encoder, variational ELBO, latent SDE
  - regime_detection.jl  : particle filter, Kalman-Bucy, Viterbi decoding
  - market_calibration.jl: end-to-end calibration pipeline on LOB data
"""
module NeuroSDE

using LinearAlgebra
using Statistics
using Random
using Distributions
using Flux
using Zygote
using Optimisers
using DataFrames
using CSV

# Conditional GPU support
const CUDA_AVAILABLE = try
    using CUDA
    CUDA.functional()
catch
    false
end

# ── Sub-modules ────────────────────────────────────────────────────────────────

include("neural_networks.jl")
include("sde_solvers.jl")
include("adjoint_sensitivity.jl")
include("volatility_models.jl")
include("calibration.jl")
include("latent_dynamics.jl")
include("regime_detection.jl")
include("market_calibration.jl")

# ── Exports ────────────────────────────────────────────────────────────────────

# Neural network types
export DriftNet, DiffusionNet, LatentSDE
export build_drift_net, build_diffusion_net
export sinusoidal_embedding, count_params
export xavier_init!, he_init!
export save_model, load_model

# SDE solvers
export EulerMaruyama, Milstein, RungeKutta45SDE
export SDEProblem, solve_sde
export batch_simulate, simulate_paths
export ItoInterpretation, StratonovichInterpretation

# Adjoint / sensitivity
export continuous_adjoint, discrete_adjoint
export control_variate_gradient, importance_sampling_gradient
export adjoint_interpolate

# Volatility models
export NeuralHeston, NeuralSABR, RoughVol, JumpDiffusion, RegimeSwitching
export simulate_model, log_likelihood_model
export calibrate_heston_params, heston_characteristic_fn

# Calibration
export mle_calibrate, gmm_calibrate, carr_madan_calibrate
export TrainingConfig, train_model!
export ValidationTracker, bootstrap_ci

# Latent dynamics
export LatentDynamicsModel, GRUEncoder, ReturnDecoder
export elbo_loss, train_latent_model!
export encode_returns, decode_latent

# Regime detection
export RegimeDetector, ParticleFilter, KalmanBucy
export detect_regimes, viterbi_decode, regime_probabilities
export systematic_resample

# Market calibration
export MarketCalibrator, RollingCalibrator
export calibrate_to_market, rolling_recalibrate
export oos_prediction_error, compare_models

end # module NeuroSDE
