"""
neural_networks.jl — Neural network architectures for Neural SDE

Implements:
  - DriftNet: MLP (state, time) → drift vector with time embedding
  - DiffusionNet: MLP → diffusion matrix (positive-definite via Cholesky)
  - LatentSDE: combined struct
  - Weight initialisers: Xavier (Glorot), He (Kaiming)
  - Utilities: count_params, save_model, load_model
"""

using Flux
using LinearAlgebra
using Random
using Statistics
using BSON: @save, @load

# ─────────────────────────────────────────────────────────────────────────────
# TIME EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

"""
    sinusoidal_embedding(t, dim)

Sinusoidal positional embedding for time t ∈ ℝ, producing a `dim`-dimensional
feature vector. Follows the transformer positional encoding scheme:

    emb[2k]   = sin(t / 10000^(2k/dim))
    emb[2k+1] = cos(t / 10000^(2k/dim))

This allows the network to distinguish close time values and generalise.
"""
function sinusoidal_embedding(t::Real, dim::Int)
    @assert iseven(dim) "Embedding dimension must be even"
    emb = zeros(Float32, dim)
    half = dim ÷ 2
    for k in 1:half
        freq = Float32(1.0 / 10000.0^(2(k-1) / dim))
        emb[2k-1] = sin(t * freq)
        emb[2k]   = cos(t * freq)
    end
    return emb
end

"""
    sinusoidal_embedding(t_vec, dim)

Batch version: t_vec is a vector of times, returns (dim × batch) matrix.
"""
function sinusoidal_embedding(t_vec::AbstractVector, dim::Int)
    batch = length(t_vec)
    emb = zeros(Float32, dim, batch)
    half = dim ÷ 2
    for (b, t) in enumerate(t_vec)
        for k in 1:half
            freq = Float32(1.0 / 10000.0^(2(k-1) / dim))
            emb[2k-1, b] = sin(t * freq)
            emb[2k,   b] = cos(t * freq)
        end
    end
    return emb
end

# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

"""
    xavier_init!(layer)

Xavier (Glorot) uniform initialisation for Dense layers.
Scales weights by sqrt(6 / (fan_in + fan_out)).
"""
function xavier_init!(layer::Dense)
    fan_in  = size(layer.weight, 2)
    fan_out = size(layer.weight, 1)
    limit   = sqrt(6.0f0 / (fan_in + fan_out))
    layer.weight .= Float32.(rand(Uniform(-limit, limit), size(layer.weight)))
    if layer.bias !== false
        layer.bias .= 0f0
    end
    return layer
end

"""
    he_init!(layer)

He (Kaiming) normal initialisation, recommended for ReLU networks.
std = sqrt(2 / fan_in).
"""
function he_init!(layer::Dense)
    fan_in = size(layer.weight, 2)
    std    = sqrt(2.0f0 / fan_in)
    layer.weight .= Float32.(randn(size(layer.weight)) .* std)
    if layer.bias !== false
        layer.bias .= 0f0
    end
    return layer
end

"""
    init_network!(model; scheme=:xavier)

Apply weight initialisation to all Dense layers in a Flux model.
"""
function init_network!(model; scheme::Symbol=:xavier)
    for layer in Flux.trainable(model)
        if layer isa Dense
            if scheme == :xavier
                xavier_init!(layer)
            elseif scheme == :he
                he_init!(layer)
            end
        end
    end
    return model
end

# ─────────────────────────────────────────────────────────────────────────────
# DRIFT NETWORK
# ─────────────────────────────────────────────────────────────────────────────

"""
    DriftNet

Multi-layer perceptron mapping (state ∈ ℝᵈ, time t ∈ ℝ) → drift ∈ ℝᵈ.

Features:
  - Sinusoidal time embedding of dimension `time_emb_dim`
  - Configurable depth (`n_layers`) and width (`hidden_dim`)
  - BatchNorm between hidden layers for training stability
  - Residual connections if `use_residual=true` and dims match
  - Final linear layer (no activation) for unconstrained drift

Fields:
  - `state_dim`    : dimension of SDE state
  - `time_emb_dim` : dimension of sinusoidal time embedding
  - `hidden_dim`   : width of hidden layers
  - `n_layers`     : number of hidden layers
  - `net`          : the Flux Chain
"""
struct DriftNet
    state_dim    :: Int
    time_emb_dim :: Int
    hidden_dim   :: Int
    n_layers     :: Int
    use_residual :: Bool
    net          :: Chain
end

"""
    build_drift_net(state_dim; time_emb_dim=16, hidden_dim=64, n_layers=3,
                    activation=tanh, use_batchnorm=true, use_residual=false,
                    init_scheme=:xavier)

Construct a DriftNet with the given architecture.
"""
function build_drift_net(state_dim::Int;
                          time_emb_dim::Int = 16,
                          hidden_dim::Int   = 64,
                          n_layers::Int     = 3,
                          activation        = tanh,
                          use_batchnorm::Bool = true,
                          use_residual::Bool  = false,
                          init_scheme::Symbol = :xavier)

    input_dim = state_dim + time_emb_dim
    layers = Any[]

    # First hidden layer
    push!(layers, Dense(input_dim => hidden_dim, activation))
    use_batchnorm && push!(layers, BatchNorm(hidden_dim))

    # Additional hidden layers
    for _ in 2:n_layers
        push!(layers, Dense(hidden_dim => hidden_dim, activation))
        use_batchnorm && push!(layers, BatchNorm(hidden_dim))
    end

    # Output layer: hidden_dim → state_dim, linear
    push!(layers, Dense(hidden_dim => state_dim))

    net = Chain(layers...)
    model = DriftNet(state_dim, time_emb_dim, hidden_dim, n_layers, use_residual, net)

    # Initialise weights
    for layer in model.net.layers
        if layer isa Dense
            init_scheme == :xavier ? xavier_init!(layer) : he_init!(layer)
        end
    end

    return model
end

"""
    (m::DriftNet)(x, t)

Forward pass: x is state (state_dim × batch) or (state_dim,), t is scalar or
vector of times. Returns drift of same shape as x.
"""
function (m::DriftNet)(x::AbstractMatrix, t::AbstractVector)
    # x: (state_dim, batch), t: (batch,)
    emb = sinusoidal_embedding(t, m.time_emb_dim)   # (time_emb_dim, batch)
    inp = vcat(x, emb)                               # (state_dim + time_emb_dim, batch)
    return m.net(inp)                                # (state_dim, batch)
end

function (m::DriftNet)(x::AbstractVector, t::Real)
    emb = sinusoidal_embedding(t, m.time_emb_dim)   # (time_emb_dim,)
    inp = vcat(x, emb)                               # (input_dim,)
    return m.net(inp)                                # (state_dim,)
end

function (m::DriftNet)(x::AbstractMatrix, t::Real)
    batch = size(x, 2)
    t_vec = fill(Float32(t), batch)
    return m(x, t_vec)
end

# Make DriftNet a Flux model
Flux.@functor DriftNet (net,)

# ─────────────────────────────────────────────────────────────────────────────
# DIFFUSION NETWORK
# ─────────────────────────────────────────────────────────────────────────────

"""
    DiffusionNet

MLP parameterising the diffusion coefficient σ(x, t).

Two modes:
  - `diagonal=true` : outputs d positive scalars → diagonal diffusion matrix.
    Uses softplus to enforce positivity: σᵢ = softplus(netᵢ(x,t)) + ε
  - `diagonal=false`: Cholesky parameterisation of full d×d PSD matrix.
    Net outputs d*(d+1)/2 entries of lower-triangular L; Σ = LLᵀ.

The Cholesky diagonal entries are exponentiated to ensure positivity:
  Lᵢᵢ = exp(raw_ii), Lᵢⱼ = raw_ij for i > j.
"""
struct DiffusionNet
    state_dim    :: Int
    time_emb_dim :: Int
    hidden_dim   :: Int
    diagonal     :: Bool
    min_sigma    :: Float32
    net          :: Chain
end

"""
    build_diffusion_net(state_dim; diagonal=true, time_emb_dim=16,
                        hidden_dim=64, n_layers=3, min_sigma=1e-4)
"""
function build_diffusion_net(state_dim::Int;
                              diagonal::Bool     = true,
                              time_emb_dim::Int  = 16,
                              hidden_dim::Int    = 64,
                              n_layers::Int      = 3,
                              min_sigma::Float32 = 1f-4,
                              init_scheme::Symbol = :xavier)

    input_dim  = state_dim + time_emb_dim
    output_dim = diagonal ? state_dim : (state_dim * (state_dim + 1)) ÷ 2

    layers = Any[]
    push!(layers, Dense(input_dim => hidden_dim, tanh))
    for _ in 2:n_layers
        push!(layers, Dense(hidden_dim => hidden_dim, tanh))
    end
    push!(layers, Dense(hidden_dim => output_dim))  # linear output

    net = Chain(layers...)

    model = DiffusionNet(state_dim, time_emb_dim, hidden_dim, diagonal, min_sigma, net)

    for layer in model.net.layers
        if layer isa Dense
            init_scheme == :xavier ? xavier_init!(layer) : he_init!(layer)
        end
    end

    return model
end

"""
    diffusion_matrix(m::DiffusionNet, x, t) → σ

Returns the diffusion coefficient(s):
  - diagonal=true : vector of length state_dim
  - diagonal=false: (state_dim × state_dim) lower-Cholesky factor L
"""
function diffusion_matrix(m::DiffusionNet, x::AbstractVector, t::Real)
    emb = sinusoidal_embedding(t, m.time_emb_dim)
    inp = vcat(x, emb)
    raw = m.net(inp)

    if m.diagonal
        # Softplus + min_sigma for diagonal case
        return Flux.softplus.(raw) .+ m.min_sigma
    else
        d = m.state_dim
        L = zeros(Float32, d, d)
        idx = 1
        for j in 1:d
            for i in j:d
                if i == j
                    L[i, j] = exp(raw[idx]) + m.min_sigma
                else
                    L[i, j] = raw[idx]
                end
                idx += 1
            end
        end
        return L  # lower-triangular Cholesky factor
    end
end

function diffusion_matrix(m::DiffusionNet, x::AbstractMatrix, t::Real)
    # Batched diagonal case only
    @assert m.diagonal "Batched diffusion only supported for diagonal=true"
    batch = size(x, 2)
    t_vec = fill(Float32(t), batch)
    emb   = sinusoidal_embedding(t_vec, m.time_emb_dim)
    inp   = vcat(x, emb)
    raw   = m.net(inp)
    return Flux.softplus.(raw) .+ m.min_sigma
end

function (m::DiffusionNet)(x, t)
    diffusion_matrix(m, x, t)
end

Flux.@functor DiffusionNet (net,)

# ─────────────────────────────────────────────────────────────────────────────
# LATENT SDE
# ─────────────────────────────────────────────────────────────────────────────

"""
    LatentSDE

Combined struct holding a DriftNet and DiffusionNet.
Represents the SDE:  dX = f(X,t)dt + g(X,t)dW

Fields:
  - `drift`     : DriftNet
  - `diffusion` : DiffusionNet
  - `state_dim` : latent state dimension
  - `name`      : optional model name for bookkeeping
"""
struct LatentSDE
    drift     :: DriftNet
    diffusion :: DiffusionNet
    state_dim :: Int
    name      :: String
end

"""
    LatentSDE(state_dim; drift_hidden=64, drift_layers=3,
              diff_hidden=64, diff_layers=2, time_emb_dim=16,
              diagonal_diffusion=true, name="LatentSDE")
"""
function LatentSDE(state_dim::Int;
                   drift_hidden::Int   = 64,
                   drift_layers::Int   = 3,
                   diff_hidden::Int    = 64,
                   diff_layers::Int    = 2,
                   time_emb_dim::Int   = 16,
                   diagonal_diffusion::Bool = true,
                   name::String        = "LatentSDE")

    drift = build_drift_net(state_dim;
                             hidden_dim   = drift_hidden,
                             n_layers     = drift_layers,
                             time_emb_dim = time_emb_dim)

    diffusion = build_diffusion_net(state_dim;
                                    diagonal     = diagonal_diffusion,
                                    hidden_dim   = diff_hidden,
                                    n_layers     = diff_layers,
                                    time_emb_dim = time_emb_dim)

    return LatentSDE(drift, diffusion, state_dim, name)
end

"""
    drift_at(model::LatentSDE, x, t) → f(x,t)
"""
drift_at(m::LatentSDE, x, t) = m.drift(x, t)

"""
    diffusion_at(model::LatentSDE, x, t) → g(x,t)
"""
diffusion_at(m::LatentSDE, x, t) = m.diffusion(x, t)

Flux.@functor LatentSDE (drift, diffusion)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

"""
    count_params(model) → Int

Count total number of trainable parameters in a Flux model.
"""
function count_params(model)
    total = 0
    for p in Flux.params(model)
        total += length(p)
    end
    return total
end

"""
    save_model(model, path::String)

Serialise a Flux model (or any Julia struct) to a BSON file.
"""
function save_model(model, path::String)
    model_cpu = Flux.cpu(model)
    @save path model_cpu
    @info "Model saved to $path ($(count_params(model)) params)"
end

"""
    load_model(path::String)

Load a Flux model from BSON file. Returns the loaded model.
"""
function load_model(path::String)
    @load path model_cpu
    @info "Model loaded from $path"
    return model_cpu
end

"""
    param_summary(model)

Print a table of layer names and parameter counts.
"""
function param_summary(model)
    println("─" ^ 50)
    println("Model: $(typeof(model))")
    println("─" ^ 50)
    total = 0
    for (i, p) in enumerate(Flux.params(model))
        n = length(p)
        total += n
        println("  param[$i]: shape=$(size(p)), n=$n")
    end
    println("─" ^ 50)
    println("  TOTAL: $total parameters")
    println("─" ^ 50)
end

# ─────────────────────────────────────────────────────────────────────────────
# BATCH NORM WITH CUSTOM FORWARD (for SDE paths)
# ─────────────────────────────────────────────────────────────────────────────

"""
    running_mean_update!(running_mean, x_mean, momentum)

Exponential moving average update for BatchNorm running statistics.
"""
function running_mean_update!(running_mean::AbstractVector,
                               x_mean::AbstractVector,
                               momentum::Float32=0.1f0)
    running_mean .= (1f0 - momentum) .* running_mean .+ momentum .* x_mean
    return running_mean
end

# ─────────────────────────────────────────────────────────────────────────────
# SPECTRAL NORMALISATION (optional layer wrapper)
# ─────────────────────────────────────────────────────────────────────────────

"""
    SpectralNorm

Wraps a Dense layer with spectral normalisation to enforce Lipschitz continuity.
The weight matrix is normalised by its largest singular value σ₁(W).

This is useful for ensuring the drift network doesn't explode gradients.
"""
mutable struct SpectralNorm
    layer :: Dense
    u     :: Vector{Float32}
    v     :: Vector{Float32}
    n_power_iter :: Int
end

function SpectralNorm(layer::Dense; n_power_iter::Int=1)
    m, n = size(layer.weight)
    u = normalize(randn(Float32, m))
    v = normalize(randn(Float32, n))
    return SpectralNorm(layer, u, v, n_power_iter)
end

function (sn::SpectralNorm)(x)
    W = sn.layer.weight
    # Power iteration to estimate largest singular value
    û = sn.u
    v̂ = sn.v
    for _ in 1:sn.n_power_iter
        v̂ = normalize(W' * û)
        û = normalize(W  * v̂)
    end
    σ = dot(û, W * v̂)
    # Update stored vectors (in-place, not tracked by AD)
    sn.u .= û
    sn.v .= v̂
    # Normalised forward pass
    W_sn = W ./ σ
    return sn.layer.σ.(W_sn * x .+ sn.layer.bias)
end

Flux.@functor SpectralNorm (layer,)

# ─────────────────────────────────────────────────────────────────────────────
# RESIDUAL BLOCK FOR DEEP DRIFT NETWORKS
# ─────────────────────────────────────────────────────────────────────────────

"""
    ResidualBlock(dim, activation)

A two-layer residual block: y = x + W₂·σ(W₁·x + b₁) + b₂.
Helps train very deep drift networks (≥ 5 layers).
"""
struct ResidualBlock
    fc1 :: Dense
    fc2 :: Dense
    act
end

function ResidualBlock(dim::Int, activation=tanh)
    fc1 = Dense(dim => dim, activation)
    fc2 = Dense(dim => dim)
    xavier_init!(fc1)
    xavier_init!(fc2)
    ResidualBlock(fc1, fc2, activation)
end

function (rb::ResidualBlock)(x)
    return x .+ rb.fc2(rb.fc1(x))
end

Flux.@functor ResidualBlock (fc1, fc2)

"""
    build_deep_drift_net(state_dim, n_res_blocks; hidden_dim=128, time_emb_dim=16)

Build a very deep drift network using residual blocks. Recommended for state_dim ≥ 4.
"""
function build_deep_drift_net(state_dim::Int, n_res_blocks::Int;
                               hidden_dim::Int   = 128,
                               time_emb_dim::Int = 16)
    input_dim = state_dim + time_emb_dim
    # Input projection
    proj = Dense(input_dim => hidden_dim, tanh)
    xavier_init!(proj)

    # Residual blocks
    blocks = [ResidualBlock(hidden_dim) for _ in 1:n_res_blocks]

    # Output projection
    out = Dense(hidden_dim => state_dim)
    xavier_init!(out)

    net = Chain(proj, blocks..., out)

    # Wrap in DriftNet (no BatchNorm, uses residuals instead)
    return DriftNet(state_dim, time_emb_dim, hidden_dim, n_res_blocks, true, net)
end

# ─────────────────────────────────────────────────────────────────────────────
# FOURIER FEATURE NETWORKS (for periodic / rough vol dynamics)
# ─────────────────────────────────────────────────────────────────────────────

"""
    FourierFeatureLayer(input_dim, n_features; sigma=1.0)

Random Fourier feature layer: maps x → [sin(Bx), cos(Bx)] where B ~ N(0, σ²).
Approximates the RBF kernel and gives the network inductive bias for smooth dynamics.
"""
struct FourierFeatureLayer
    B     :: Matrix{Float32}   # (n_features, input_dim)
    sigma :: Float32
end

function FourierFeatureLayer(input_dim::Int, n_features::Int; sigma::Float32=1.0f0)
    B = Float32.(randn(n_features ÷ 2, input_dim) .* sigma)
    return FourierFeatureLayer(B, sigma)
end

function (ff::FourierFeatureLayer)(x::AbstractVector)
    z = ff.B * x   # (n_features/2,)
    return vcat(sin.(z), cos.(z))
end

function (ff::FourierFeatureLayer)(x::AbstractMatrix)
    z = ff.B * x   # (n_features/2, batch)
    return vcat(sin.(z), cos.(z))
end

Flux.@functor FourierFeatureLayer (B,)

"""
    build_fourier_drift_net(state_dim; n_fourier=32, hidden_dim=64,
                             n_layers=2, time_emb_dim=16, sigma=1.0f0)

Drift network with Fourier feature front-end, suitable for rough/periodic dynamics.
"""
function build_fourier_drift_net(state_dim::Int;
                                  n_fourier::Int    = 32,
                                  hidden_dim::Int   = 64,
                                  n_layers::Int     = 2,
                                  time_emb_dim::Int = 16,
                                  sigma::Float32    = 1.0f0)

    input_dim    = state_dim + time_emb_dim
    fourier_dim  = n_fourier

    ff    = FourierFeatureLayer(input_dim, fourier_dim; sigma=sigma)
    fc1   = Dense(fourier_dim => hidden_dim, tanh)
    inner = [Dense(hidden_dim => hidden_dim, tanh) for _ in 1:n_layers-1]
    out   = Dense(hidden_dim => state_dim)

    net = Chain(ff, fc1, inner..., out)

    return DriftNet(state_dim, time_emb_dim, hidden_dim, n_layers+1, false, net)
end
