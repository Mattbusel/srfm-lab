"""
latent_dynamics.jl — Latent SDE dynamics with variational inference

Implements a VAE-style model where:
  - An RNN (GRU) encoder reads a window of observed returns
    and outputs initial latent state μ₀, σ₀ (mean + std of q(z₀|x_{1:T}))
  - A LatentSDE evolves the latent state z(t) forward
  - A decoder maps z(t) → observed return distribution (μ_r, σ_r)
  - ELBO objective for end-to-end training

Based on:
  - Rubanova et al. (2019) "Latent ODEs for irregularly sampled time series"
  - Li et al. (2020) "Scalable gradients for SDEs"
  - Kidger et al. (2021) "Neural SDEs as infinite-dimensional GANs"
"""

using Flux
using LinearAlgebra
using Statistics
using Random
using Distributions
using Zygote

# ─────────────────────────────────────────────────────────────────────────────
# GRU ENCODER
# ─────────────────────────────────────────────────────────────────────────────

"""
    GRUEncoder

Bidirectional GRU that reads a time series of observations x_{1:T} ∈ ℝ^{d_obs}
and outputs a distribution q(z₀|x) = N(μ₀, diag(σ₀²)) for the initial latent state.

Architecture:
  - GRU processes sequence backwards (more informative for initial state)
  - Final hidden state → two linear layers → (μ₀, log σ₀) ∈ ℝ^{d_latent}

Fields:
  - `d_obs`    : observation dimension (e.g. 1 for scalar returns)
  - `d_latent` : latent SDE dimension
  - `d_hidden` : GRU hidden dimension
  - `gru`      : Flux.GRUCell
  - `fc_mu`    : linear layer h → μ₀
  - `fc_log_sigma` : linear layer h → log σ₀
"""
struct GRUEncoder
    d_obs        :: Int
    d_latent     :: Int
    d_hidden     :: Int
    gru          :: Flux.Recur   # wrapped GRUCell
    fc_mu        :: Dense
    fc_log_sigma :: Dense
    fc_init      :: Dense        # project observation to GRU input
end

"""
    GRUEncoder(d_obs, d_latent; d_hidden=64, n_layers=1)
"""
function GRUEncoder(d_obs::Int, d_latent::Int;
                     d_hidden::Int = 64)

    gru          = Flux.RNN(d_obs => d_hidden, tanh; init=Flux.glorot_uniform)
    # Actually use GRU:
    gru_cell     = Flux.GRU(d_obs => d_hidden)
    fc_mu        = Dense(d_hidden => d_latent)
    fc_log_sigma = Dense(d_hidden => d_latent)
    fc_init      = Dense(d_obs    => d_obs)    # identity projection (learnable)

    return GRUEncoder(d_obs, d_latent, d_hidden, gru_cell, fc_mu, fc_log_sigma, fc_init)
end

Flux.@functor GRUEncoder (gru, fc_mu, fc_log_sigma, fc_init)

"""
    encode_returns(enc::GRUEncoder, returns_window) → (μ₀, σ₀)

Encode a window of returns into initial latent state distribution.

returns_window: (d_obs × T_window) matrix (each column is one time step)
Returns:
  - μ₀        : (d_latent,) mean of initial latent state
  - σ₀        : (d_latent,) std of initial latent state (always positive)
"""
function encode_returns(enc::GRUEncoder,
                         returns_window::AbstractMatrix)
    # returns_window: (d_obs, T_window)
    T_window = size(returns_window, 2)

    # Process sequence backwards (GRU reads T → 1)
    Flux.reset!(enc.gru)
    h = nothing
    for t in T_window:-1:1
        x_t = returns_window[:, t]  # (d_obs,)
        h   = enc.gru(x_t)          # (d_hidden,)
    end

    if h === nothing
        h = zeros(Float32, enc.d_hidden)
    end

    μ₀       = enc.fc_mu(h)
    log_σ₀   = enc.fc_log_sigma(h)
    σ₀       = Flux.softplus.(log_σ₀) .+ 1f-4

    return μ₀, σ₀
end

"""
    encode_returns(enc::GRUEncoder, returns_batch) → (μ₀_batch, σ₀_batch)

Batch encoding: returns_batch is (d_obs, T_window, batch_size).
"""
function encode_returns(enc::GRUEncoder, returns_batch::Array{<:Real,3})
    d_obs, T_window, batch = size(returns_batch)

    μ_batch  = zeros(Float32, enc.d_latent, batch)
    σ_batch  = zeros(Float32, enc.d_latent, batch)

    for b in 1:batch
        μ, σ = encode_returns(enc, returns_batch[:, :, b])
        μ_batch[:, b] = μ
        σ_batch[:, b] = σ
    end

    return μ_batch, σ_batch
end

# ─────────────────────────────────────────────────────────────────────────────
# RETURN DECODER
# ─────────────────────────────────────────────────────────────────────────────

"""
    ReturnDecoder

Maps latent state z(t) ∈ ℝ^{d_latent} to observed return distribution:
  p(r_t | z_t) = N(μ_r(z_t), σ_r²(z_t))

Architecture: MLP with 2 output heads (mean and log-std).
"""
struct ReturnDecoder
    d_latent :: Int
    d_obs    :: Int
    net      :: Chain
    fc_mu    :: Dense
    fc_logσ  :: Dense
end

function ReturnDecoder(d_latent::Int, d_obs::Int;
                        hidden_dim::Int = 64,
                        n_layers::Int   = 2)

    layers = Any[Dense(d_latent => hidden_dim, tanh)]
    for _ in 2:n_layers
        push!(layers, Dense(hidden_dim => hidden_dim, tanh))
    end
    net    = Chain(layers...)
    fc_mu  = Dense(hidden_dim => d_obs)
    fc_logσ = Dense(hidden_dim => d_obs)

    return ReturnDecoder(d_latent, d_obs, net, fc_mu, fc_logσ)
end

Flux.@functor ReturnDecoder (net, fc_mu, fc_logσ)

"""
    decode_latent(dec::ReturnDecoder, z) → (μ_r, σ_r)

Map latent state z → return distribution parameters.
z: (d_latent,) or (d_latent, batch)
"""
function decode_latent(dec::ReturnDecoder, z::AbstractVector)
    h    = dec.net(z)
    μ_r  = dec.fc_mu(h)
    σ_r  = Flux.softplus.(dec.fc_logσ(h)) .+ 1f-4
    return μ_r, σ_r
end

function decode_latent(dec::ReturnDecoder, z::AbstractMatrix)
    h    = dec.net(z)
    μ_r  = dec.fc_mu(h)
    σ_r  = Flux.softplus.(dec.fc_logσ(h)) .+ 1f-4
    return μ_r, σ_r
end

# ─────────────────────────────────────────────────────────────────────────────
# LATENT DYNAMICS MODEL
# ─────────────────────────────────────────────────────────────────────────────

"""
    LatentDynamicsModel

Full model combining encoder, latent SDE, and decoder.

Generative model:
  z₀ ~ p(z₀) = N(0, I)
  dz = f_θ(z, t)dt + g_θ(z, t)dW      (LatentSDE)
  r_t | z_t ~ N(μ_dec(z_t), σ²_dec(z_t))  (ReturnDecoder)

Recognition (variational) model:
  z₀ | x_{1:T} ~ N(μ_enc(x), σ²_enc(x))  (GRUEncoder)

Fields:
  - `encoder`    : GRUEncoder
  - `sde_model`  : LatentSDE
  - `decoder`    : ReturnDecoder
  - `d_latent`   : latent dimension
  - `d_obs`      : observation dimension
"""
struct LatentDynamicsModel
    encoder   :: GRUEncoder
    sde_model :: LatentSDE
    decoder   :: ReturnDecoder
    d_latent  :: Int
    d_obs     :: Int
end

"""
    LatentDynamicsModel(d_obs, d_latent; encoder_hidden=64, sde_hidden=64,
                         decoder_hidden=64, sde_layers=3)
"""
function LatentDynamicsModel(d_obs::Int, d_latent::Int;
                               encoder_hidden::Int = 64,
                               sde_hidden::Int     = 64,
                               decoder_hidden::Int = 64,
                               sde_layers::Int     = 3,
                               time_emb_dim::Int   = 16)

    encoder   = GRUEncoder(d_obs, d_latent; d_hidden=encoder_hidden)
    sde_model = LatentSDE(d_latent;
                           drift_hidden=sde_hidden, drift_layers=sde_layers,
                           diff_hidden=sde_hidden÷2, diff_layers=2,
                           time_emb_dim=time_emb_dim,
                           diagonal_diffusion=true)
    decoder   = ReturnDecoder(d_latent, d_obs; hidden_dim=decoder_hidden)

    return LatentDynamicsModel(encoder, sde_model, decoder, d_latent, d_obs)
end

Flux.@functor LatentDynamicsModel (encoder, sde_model, decoder)

# ─────────────────────────────────────────────────────────────────────────────
# ELBO OBJECTIVE
# ─────────────────────────────────────────────────────────────────────────────

"""
    reparameterise(μ, σ; rng) → z

Reparameterisation trick: z = μ + σ ⊙ ε, ε ~ N(0, I).
Allows gradients to flow through the sampling operation.
"""
function reparameterise(μ::AbstractArray, σ::AbstractArray;
                         rng = Random.GLOBAL_RNG)
    ε = Float32.(randn(rng, size(μ)...))
    return μ .+ σ .* ε
end

"""
    kl_gaussian(μ, σ) → KL divergence

KL[q(z) || p(z)] where q = N(μ, σ²I) and p = N(0, I).
  KL = (1/2) Σᵢ [μᵢ² + σᵢ² - log σᵢ² - 1]
"""
function kl_gaussian(μ::AbstractArray, σ::AbstractArray)
    return 0.5f0 .* sum(μ.^2 .+ σ.^2 .- log.(σ.^2 .+ 1f-8) .- 1f0)
end

"""
    gaussian_nll(μ_pred, σ_pred, x_obs) → NLL

Negative log-likelihood under Gaussian decoder:
  NLL = Σ_t [log(σ_t) + (x_t - μ_t)² / (2σ_t²)] + n/2 log(2π)
"""
function gaussian_nll(μ_pred::AbstractArray, σ_pred::AbstractArray,
                       x_obs::AbstractArray)
    return sum(log.(σ_pred .+ 1f-8) .+
                (x_obs .- μ_pred).^2 ./ (2f0 .* σ_pred.^2 .+ 1f-8)) .+
           0.5f0 * length(x_obs) * log(2π)
end

"""
    elbo_loss(ldm::LatentDynamicsModel, returns_window, future_returns;
               dt, n_latent_samples, rng) → ELBO

Compute the negative ELBO (Evidence Lower BOund):

  -ELBO = E_q[-log p(x|z)] + KL[q(z₀|x) || p(z₀)]
        = reconstruction_loss + kl_loss

Arguments:
  - returns_window  : (d_obs, T_enc) tensor of past returns (context window)
  - future_returns  : (d_obs, T_pred) tensor of future returns to reconstruct
  - dt              : time step
  - n_latent_samples: number of latent samples for MC estimate of ELBO
"""
function elbo_loss(ldm::LatentDynamicsModel,
                    returns_window::AbstractMatrix,
                    future_returns::AbstractMatrix;
                    dt::Float64 = 1.0/252,
                    n_latent_samples::Int = 5,
                    rng = Random.GLOBAL_RNG)

    T_pred = size(future_returns, 2)
    tspan  = (0.0, T_pred * dt)

    # Encode: q(z₀|x) = N(μ_q, σ_q)
    μ_q, σ_q = encode_returns(ldm.encoder, returns_window)

    # KL divergence (closed form since p(z₀) = N(0,I))
    kl = kl_gaussian(μ_q, σ_q)

    # Reconstruction via MC estimate
    recon_ll = 0.0f0

    for _ in 1:n_latent_samples
        # Sample z₀ via reparameterisation
        z = reparameterise(μ_q, σ_q; rng=rng)

        # Integrate SDE forward from z₀
        t = Float32(0.0)
        recon_ll_sample = 0.0f0

        for k in 1:T_pred
            dt_k = Float32(dt)

            # Decode current latent state
            μ_r, σ_r = decode_latent(ldm.decoder, z)

            # Reconstruction log-likelihood
            r_obs = future_returns[:, k]
            recon_ll_sample -= gaussian_nll(μ_r, σ_r, Float32.(r_obs))

            # Euler-Maruyama step for latent SDE
            μ_sde = drift_at(ldm.sde_model, z, t)
            σ_sde = diffusion_at(ldm.sde_model, z, t)
            dW    = Float32.(sqrt(dt_k) .* randn(rng, ldm.d_latent))
            z     = z .+ μ_sde .* dt_k .+ σ_sde .* dW
            t    += dt_k
        end

        recon_ll += recon_ll_sample
    end

    recon_ll /= n_latent_samples

    # -ELBO = -recon_ll + kl
    return -recon_ll + kl
end

"""
    elbo_loss_batch(ldm, batch_window, batch_future; dt, n_latent_samples, rng)

Batched ELBO computation.
batch_window  : (d_obs, T_enc,  batch_size)
batch_future  : (d_obs, T_pred, batch_size)
"""
function elbo_loss_batch(ldm::LatentDynamicsModel,
                          batch_window::Array{<:Real,3},
                          batch_future::Array{<:Real,3};
                          dt::Float64           = 1.0/252,
                          n_latent_samples::Int = 5,
                          rng                   = Random.GLOBAL_RNG)

    batch_size = size(batch_window, 3)
    total_elbo = 0.0f0

    for b in 1:batch_size
        elbo_b = elbo_loss(ldm,
                            batch_window[:, :, b],
                            batch_future[:, :, b];
                            dt=dt, n_latent_samples=n_latent_samples, rng=rng)
        total_elbo += elbo_b
    end

    return total_elbo / batch_size
end

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

"""
    prepare_windows(returns, enc_len, pred_len; stride=1)

Split a return series into overlapping (context, prediction) window pairs.

returns   : (n,) or (d_obs, n) matrix
enc_len   : length of encoder context window
pred_len  : length of prediction horizon
stride    : step between windows

Returns (windows, futures) each of shape (d_obs, window_len, n_windows).
"""
function prepare_windows(returns::AbstractVector, enc_len::Int, pred_len::Int;
                          stride::Int = 1)
    returns = reshape(Float32.(returns), 1, :)
    return prepare_windows(returns, enc_len, pred_len; stride=stride)
end

function prepare_windows(returns::AbstractMatrix, enc_len::Int, pred_len::Int;
                          stride::Int = 1)
    d_obs, n = size(returns)
    total    = enc_len + pred_len
    n_win    = floor(Int, (n - total) / stride) + 1

    windows = zeros(Float32, d_obs, enc_len,  n_win)
    futures = zeros(Float32, d_obs, pred_len, n_win)

    for (i, start) in enumerate(1:stride:n-total+1)
        windows[:, :, i] = returns[:, start:start+enc_len-1]
        futures[:, :, i] = returns[:, start+enc_len:start+total-1]
    end

    return windows, futures
end

"""
    train_latent_model!(ldm, returns; enc_len, pred_len, config, rng)

Train LatentDynamicsModel on a return series.

Steps:
  1. Prepare sliding windows from returns
  2. Shuffle and batch
  3. Minimise ELBO with Adam
  4. Validate on held-out windows
  5. Return trained model and loss history
"""
function train_latent_model!(ldm::LatentDynamicsModel,
                              returns::AbstractVector;
                              enc_len::Int       = 60,
                              pred_len::Int      = 21,
                              n_epochs::Int      = 200,
                              batch_size::Int    = 32,
                              lr::Float64        = 1e-3,
                              val_frac::Float64  = 0.2,
                              dt::Float64        = 1.0/252,
                              n_latent_samples::Int = 5,
                              rng                = Random.GLOBAL_RNG,
                              print_every::Int   = 20)

    # Prepare windows
    windows, futures = prepare_windows(returns, enc_len, pred_len; stride=5)
    n_win   = size(windows, 3)
    n_val   = round(Int, n_win * val_frac)
    n_train = n_win - n_val

    # Shuffle indices
    perm   = randperm(rng, n_win)
    train_idx = perm[1:n_train]
    val_idx   = perm[n_train+1:end]

    ps_flat   = Flux.params(ldm)
    optim     = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(optim, ldm)

    train_losses = Float64[]
    val_losses   = Float64[]

    best_val    = Inf
    best_params = deepcopy(Flux.params(ldm))
    no_improve  = 0
    patience    = 30

    for epoch in 1:n_epochs
        # Shuffle training indices
        shuffle!(rng, train_idx)

        epoch_losses = Float64[]
        n_batches = ceil(Int, n_train / batch_size)

        for b in 1:n_batches
            idx_start = (b-1)*batch_size + 1
            idx_end   = min(b*batch_size, n_train)
            batch_idx = train_idx[idx_start:idx_end]
            bsz       = length(batch_idx)

            bw = windows[:, :, batch_idx]
            bf = futures[:, :, batch_idx]

            loss_val, grads = Zygote.withgradient(ps_flat) do
                elbo_loss_batch(ldm, bw, bf;
                                 dt=dt, n_latent_samples=n_latent_samples, rng=rng)
            end

            isfinite(loss_val) && Optimisers.update!(opt_state, ldm, grads)
            push!(epoch_losses, loss_val)
        end

        train_loss = mean(filter(isfinite, epoch_losses))

        # Validation
        val_loss = 0.0f0
        for i in val_idx
            vl = elbo_loss(ldm, windows[:,:,i], futures[:,:,i];
                            dt=dt, n_latent_samples=2, rng=rng)
            isfinite(vl) && (val_loss += vl)
        end
        val_loss /= max(length(val_idx), 1)

        push!(train_losses, train_loss)
        push!(val_losses,   val_loss)

        if val_loss < best_val - 1e-4
            best_val    = val_loss
            best_params = deepcopy(Flux.params(ldm))
            no_improve  = 0
        else
            no_improve += 1
        end

        if epoch % print_every == 0
            @info "Epoch $epoch: train ELBO=$(round(train_loss,digits=3)), val ELBO=$(round(val_loss,digits=3))"
        end

        if no_improve >= patience
            @info "Early stopping at epoch $epoch"
            Flux.loadparams!(ldm, best_params)
            break
        end
    end

    return ldm, train_losses, val_losses
end

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION / INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

"""
    predict_returns(ldm, context_returns, pred_len; dt, n_samples, rng)

Generate return predictions from a trained LatentDynamicsModel.

context_returns : (d_obs, T_context) historical returns window
pred_len        : prediction horizon

Returns:
  - pred_mu    : (d_obs, pred_len) posterior predictive mean
  - pred_std   : (d_obs, pred_len) posterior predictive std
  - samples    : (d_obs, pred_len, n_samples) individual samples
"""
function predict_returns(ldm::LatentDynamicsModel,
                          context_returns::AbstractMatrix,
                          pred_len::Int;
                          dt::Float64      = 1.0/252,
                          n_samples::Int   = 100,
                          rng              = Random.GLOBAL_RNG)

    d_obs   = size(context_returns, 1)
    samples = zeros(Float32, d_obs, pred_len, n_samples)

    # Encode context
    μ_q, σ_q = encode_returns(ldm.encoder, Float32.(context_returns))

    for s in 1:n_samples
        z   = reparameterise(μ_q, σ_q; rng=rng)
        t   = Float32(0.0)

        for k in 1:pred_len
            dt_k    = Float32(dt)
            μ_r, σ_r = decode_latent(ldm.decoder, z)

            # Sample from decoder
            ε_r = Float32.(randn(rng, d_obs))
            samples[:, k, s] = μ_r .+ σ_r .* ε_r

            # Advance latent state
            μ_sde = drift_at(ldm.sde_model, z, t)
            σ_sde = diffusion_at(ldm.sde_model, z, t)
            dW    = Float32.(sqrt(dt_k) .* randn(rng, ldm.d_latent))
            z     = z .+ μ_sde .* dt_k .+ σ_sde .* dW
            t    += dt_k
        end
    end

    pred_mu  = dropdims(mean(samples, dims=3), dims=3)
    pred_std = dropdims(std(samples,  dims=3), dims=3)

    return pred_mu, pred_std, samples
end

"""
    latent_trajectory(ldm, context_returns; dt, n_samples, rng)

Extract the posterior latent trajectory from a trained model.
Returns (z_mean, z_std) each of shape (d_latent, T_context).
"""
function latent_trajectory(ldm::LatentDynamicsModel,
                             context_returns::AbstractMatrix;
                             dt::Float64    = 1.0/252,
                             n_samples::Int = 50,
                             rng            = Random.GLOBAL_RNG)

    T    = size(context_returns, 2)
    d_lat = ldm.d_latent

    all_z = zeros(Float32, d_lat, T, n_samples)

    μ_q, σ_q = encode_returns(ldm.encoder, Float32.(context_returns))

    for s in 1:n_samples
        z = reparameterise(μ_q, σ_q; rng=rng)
        t = Float32(0.0)
        for k in 1:T
            all_z[:, k, s] = z
            dt_k = Float32(dt)
            μ_sde = drift_at(ldm.sde_model, z, t)
            σ_sde = diffusion_at(ldm.sde_model, z, t)
            dW    = Float32.(sqrt(dt_k) .* randn(rng, d_lat))
            z     = z .+ μ_sde .* dt_k .+ σ_sde .* dW
            t    += dt_k
        end
    end

    z_mean = dropdims(mean(all_z, dims=3), dims=3)
    z_std  = dropdims(std(all_z,  dims=3), dims=3)

    return z_mean, z_std
end
