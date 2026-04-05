# machine_learning_advanced.jl
# Advanced ML methods for crypto/quant trading lab
# Pure Julia stdlib implementation

module AdvancedML

using Statistics, LinearAlgebra, Random

# ============================================================
# DATA STRUCTURES
# ============================================================

struct SVMModel
    support_vectors::Matrix{Float64}
    sv_labels::Vector{Float64}
    alphas::Vector{Float64}
    bias::Float64
    kernel::Symbol
    kernel_params::Dict{Symbol, Float64}
end

struct GPModel
    X_train::Matrix{Float64}
    y_train::Vector{Float64}
    K_inv::Matrix{Float64}
    alpha_gp::Vector{Float64}
    noise_var::Float64
    length_scale::Float64
    signal_var::Float64
    kernel::Symbol
end

struct VAEModel
    encoder_weights::Vector{Matrix{Float64}}
    encoder_biases::Vector{Vector{Float64}}
    mu_weights::Matrix{Float64}
    mu_bias::Vector{Float64}
    logvar_weights::Matrix{Float64}
    logvar_bias::Vector{Float64}
    decoder_weights::Vector{Matrix{Float64}}
    decoder_biases::Vector{Vector{Float64}}
    latent_dim::Int
    input_dim::Int
end

struct TransformerEncoder
    W_Q::Array{Float64, 3}   # n_heads x d_k x d_model
    W_K::Array{Float64, 3}
    W_V::Array{Float64, 3}
    W_O::Matrix{Float64}     # d_model x d_model
    W_ff1::Matrix{Float64}
    W_ff2::Matrix{Float64}
    b_ff1::Vector{Float64}
    b_ff2::Vector{Float64}
    n_heads::Int
    d_model::Int
    d_k::Int
    d_ff::Int
end

struct MAMLState
    meta_weights::Vector{Matrix{Float64}}
    meta_biases::Vector{Vector{Float64}}
    inner_lr::Float64
    outer_lr::Float64
    n_inner_steps::Int
end

# ============================================================
# KERNEL FUNCTIONS
# ============================================================

"""RBF (Gaussian) kernel: k(x,y) = σ² exp(-||x-y||²/(2l²))"""
function rbf_kernel(x::Vector{Float64}, y::Vector{Float64};
                    length_scale::Float64=1.0, signal_var::Float64=1.0)
    return signal_var * exp(-0.5 * sum((x - y).^2) / length_scale^2)
end

"""Polynomial kernel: k(x,y) = (x·y + c)^d"""
function poly_kernel(x::Vector{Float64}, y::Vector{Float64};
                     degree::Float64=3.0, coef0::Float64=1.0, gamma::Float64=1.0)
    return (gamma * dot(x, y) + coef0)^degree
end

"""Linear kernel: k(x,y) = x·y"""
function linear_kernel(x::Vector{Float64}, y::Vector{Float64}; kwargs...)
    return dot(x, y)
end

"""Matern 5/2 kernel."""
function matern52_kernel(x::Vector{Float64}, y::Vector{Float64};
                          length_scale::Float64=1.0, signal_var::Float64=1.0)
    r = sqrt(sum((x - y).^2)) / length_scale
    return signal_var * (1 + sqrt(5) * r + 5 * r^2 / 3) * exp(-sqrt(5) * r)
end

"""Build kernel matrix K[i,j] = k(X[i,:], X[j,:])."""
function build_kernel_matrix(X::Matrix{Float64}, kernel_fn::Function; kwargs...)
    n = size(X, 1)
    K = zeros(n, n)
    for i in 1:n
        for j in i:n
            K[i, j] = kernel_fn(X[i, :], X[j, :]; kwargs...)
            K[j, i] = K[i, j]
        end
    end
    return K
end

"""Cross-kernel matrix K_star[i,j] = k(X_new[i,:], X_train[j,:])."""
function build_cross_kernel(X_new::Matrix{Float64}, X_train::Matrix{Float64},
                              kernel_fn::Function; kwargs...)
    n_new = size(X_new, 1)
    n_train = size(X_train, 1)
    K = zeros(n_new, n_train)
    for i in 1:n_new
        for j in 1:n_train
            K[i, j] = kernel_fn(X_new[i, :], X_train[j, :]; kwargs...)
        end
    end
    return K
end

# ============================================================
# SUPPORT VECTOR MACHINE (SVM)
# ============================================================

"""
Sequential Minimal Optimization (SMO) for SVM.
Solves the dual: max Σα_i - ½ Σ_i Σ_j α_i α_j y_i y_j K(x_i,x_j)
subject to 0 ≤ α_i ≤ C, Σ α_i y_i = 0.
"""
function smo_train(X::Matrix{Float64}, y::Vector{Float64};
                   C::Float64=1.0, kernel::Symbol=:rbf,
                   kernel_params::Dict{Symbol, Float64}=Dict(:length_scale=>1.0, :signal_var=>1.0),
                   tol::Float64=1e-3, max_passes::Int=200)
    n, d = size(X)
    alphas = zeros(n)
    b = 0.0

    # Build kernel function
    kfn = if kernel == :rbf
        (xi, xj) -> rbf_kernel(xi, xj;
                                length_scale=get(kernel_params, :length_scale, 1.0),
                                signal_var=get(kernel_params, :signal_var, 1.0))
    elseif kernel == :poly
        (xi, xj) -> poly_kernel(xi, xj;
                                  degree=get(kernel_params, :degree, 3.0),
                                  coef0=get(kernel_params, :coef0, 1.0))
    else
        (xi, xj) -> linear_kernel(xi, xj)
    end

    # Precompute kernel matrix
    K = zeros(n, n)
    for i in 1:n
        for j in i:n
            K[i, j] = kfn(X[i, :], X[j, :])
            K[j, i] = K[i, j]
        end
    end

    # Decision function
    function decision(i::Int)
        return sum(alphas[j] * y[j] * K[j, i] for j in 1:n) - b
    end

    passes = 0
    while passes < max_passes
        num_changed = 0
        for i in 1:n
            Ei = decision(i) - y[i]
            # KKT violation check
            if (y[i] * Ei < -tol && alphas[i] < C) || (y[i] * Ei > tol && alphas[i] > 0)
                # Choose j ≠ i randomly
                j = rand(1:n-1)
                j >= i && (j += 1)

                Ej = decision(j) - y[j]
                ai_old, aj_old = alphas[i], alphas[j]

                # Compute bounds
                if y[i] != y[j]
                    L = max(0.0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else
                    L = max(0.0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])
                end
                L >= H && continue

                # Compute eta
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                eta >= 0 && continue

                # Update alpha_j
                alphas[j] -= y[j] * (Ei - Ej) / eta
                alphas[j] = clamp(alphas[j], L, H)
                abs(alphas[j] - aj_old) < 1e-5 && continue

                # Update alpha_i
                alphas[i] += y[i] * y[j] * (aj_old - alphas[j])

                # Update bias
                b1 = b - Ei - y[i]*(alphas[i]-ai_old)*K[i,i] - y[j]*(alphas[j]-aj_old)*K[i,j]
                b2 = b - Ej - y[i]*(alphas[i]-ai_old)*K[i,j] - y[j]*(alphas[j]-aj_old)*K[j,j]
                if 0 < alphas[i] < C
                    b = b1
                elseif 0 < alphas[j] < C
                    b = b2
                else
                    b = (b1 + b2) / 2
                end

                num_changed += 1
            end
        end

        if num_changed == 0
            passes += 1
        else
            passes = 0
        end
    end

    # Extract support vectors
    sv_idx = findall(alphas .> 1e-5)
    X_sv = X[sv_idx, :]
    y_sv = y[sv_idx]
    alphas_sv = alphas[sv_idx]

    return SVMModel(X_sv, y_sv, alphas_sv, b, kernel, kernel_params)
end

"""Predict with trained SVM."""
function svm_predict(model::SVMModel, X_new::Matrix{Float64})
    n_new = size(X_new, 1)
    n_sv = size(model.support_vectors, 1)

    kfn = if model.kernel == :rbf
        (xi, xj) -> rbf_kernel(xi, xj;
                                length_scale=get(model.kernel_params, :length_scale, 1.0),
                                signal_var=get(model.kernel_params, :signal_var, 1.0))
    elseif model.kernel == :poly
        (xi, xj) -> poly_kernel(xi, xj;
                                  degree=get(model.kernel_params, :degree, 3.0))
    else
        (xi, xj) -> linear_kernel(xi, xj)
    end

    scores = zeros(n_new)
    for i in 1:n_new
        for j in 1:n_sv
            scores[i] += model.alphas[j] * model.sv_labels[j] *
                          kfn(model.support_vectors[j, :], X_new[i, :])
        end
        scores[i] -= model.bias
    end

    return sign.(scores), scores
end

"""SVM regression (SVR) with ε-insensitive loss."""
function svr_train(X::Matrix{Float64}, y::Vector{Float64};
                   C::Float64=1.0, epsilon::Float64=0.1,
                   kernel::Symbol=:rbf,
                   kernel_params::Dict{Symbol, Float64}=Dict(:length_scale=>1.0, :signal_var=>1.0),
                   max_iter::Int=1000)
    n, d = size(X)
    # Transform to classification: expand each sample to 2 (for α and α*)
    # Use simplified gradient descent on dual for SVR
    alpha = zeros(n)     # α_i
    alpha_star = zeros(n)  # α_i*
    b = 0.0

    kfn = if kernel == :rbf
        (xi, xj) -> rbf_kernel(xi, xj;
                                length_scale=get(kernel_params, :length_scale, 1.0),
                                signal_var=get(kernel_params, :signal_var, 1.0))
    else
        (xi, xj) -> linear_kernel(xi, xj)
    end

    K = zeros(n, n)
    for i in 1:n, j in 1:n
        K[i, j] = kfn(X[i, :], X[j, :])
    end

    lr = 0.01 / n
    for iter in 1:max_iter
        for i in 1:n
            # Prediction
            f = sum((alpha[j] - alpha_star[j]) * K[j, i] for j in 1:n) + b
            res = f - y[i]

            # Update alpha_i, alpha_star_i
            if res > epsilon
                da = lr * (1 - res + epsilon)
                alpha[i] = clamp(alpha[i] - da, 0, C)
                alpha_star[i] = clamp(alpha_star[i] + da, 0, C)
            elseif res < -epsilon
                da = lr * (1 + res + epsilon)
                alpha[i] = clamp(alpha[i] + da, 0, C)
                alpha_star[i] = clamp(alpha_star[i] - da, 0, C)
            end
        end

        # Update bias
        b = mean(y[i] - sum((alpha[j]-alpha_star[j])*K[j,i] for j in 1:n)
                 for i in 1:n if alpha[i] > 1e-5 || alpha_star[i] > 1e-5)
        isnan(b) && (b = 0.0)
    end

    sv_idx = findall((alpha .+ alpha_star) .> 1e-5)
    X_sv = X[sv_idx, :]
    alphas_sv = alpha[sv_idx] - alpha_star[sv_idx]

    return (X_sv=X_sv, alphas=alphas_sv, bias=b, kernel=kernel, kernel_params=kernel_params)
end

# ============================================================
# GAUSSIAN PROCESS REGRESSION
# ============================================================

"""
Train a Gaussian Process regression model.
Prior: f ~ GP(0, k(x,x'))
Likelihood: y = f(x) + ε, ε ~ N(0, σ²_n)
"""
function gp_train(X::Matrix{Float64}, y::Vector{Float64};
                  kernel::Symbol=:rbf,
                  length_scale::Float64=1.0,
                  signal_var::Float64=1.0,
                  noise_var::Float64=0.01)
    n = size(X, 1)

    kfn = if kernel == :rbf
        (xi, xj) -> rbf_kernel(xi, xj, length_scale=length_scale, signal_var=signal_var)
    elseif kernel == :matern52
        (xi, xj) -> matern52_kernel(xi, xj, length_scale=length_scale, signal_var=signal_var)
    else
        (xi, xj) -> linear_kernel(xi, xj)
    end

    K = build_kernel_matrix(X, kfn)
    K_noise = K + noise_var * I

    # Cholesky decomposition for stability
    K_inv = inv(K_noise + 1e-8 * I)
    alpha_gp = K_inv * y

    return GPModel(X, y, K_inv, alpha_gp, noise_var, length_scale, signal_var, kernel)
end

"""
GP posterior predictive distribution.
Returns (mean, variance) for each test point.
"""
function gp_predict(model::GPModel, X_new::Matrix{Float64})
    n_new = size(X_new, 1)

    kfn = if model.kernel == :rbf
        (xi, xj) -> rbf_kernel(xi, xj, length_scale=model.length_scale, signal_var=model.signal_var)
    elseif model.kernel == :matern52
        (xi, xj) -> matern52_kernel(xi, xj, length_scale=model.length_scale, signal_var=model.signal_var)
    else
        (xi, xj) -> linear_kernel(xi, xj)
    end

    K_star = build_cross_kernel(X_new, model.X_train, kfn)

    # Posterior mean
    mu = K_star * model.alpha_gp

    # Posterior variance
    sigma2 = zeros(n_new)
    for i in 1:n_new
        k_ss = kfn(X_new[i, :], X_new[i, :])
        k_s = K_star[i, :]
        sigma2[i] = k_ss - dot(k_s, model.K_inv * k_s) + model.noise_var
        sigma2[i] = max(0.0, sigma2[i])
    end

    return mu, sqrt.(sigma2)
end

"""
GP marginal log-likelihood for hyperparameter optimization.
log p(y|X,θ) = -½ y'K_inv y - ½ log|K| - n/2 log(2π)
"""
function gp_log_marginal_likelihood(X::Matrix{Float64}, y::Vector{Float64};
                                     length_scale::Float64=1.0,
                                     signal_var::Float64=1.0,
                                     noise_var::Float64=0.01)
    n = length(y)
    K = build_kernel_matrix(X, (xi, xj) -> rbf_kernel(xi, xj,
                              length_scale=length_scale, signal_var=signal_var))
    K += noise_var * I + 1e-8 * I

    try
        L = cholesky(Symmetric(K)).L
        alpha = L' \ (L \ y)
        log_det = 2 * sum(log.(diag(L)))
        return -0.5 * dot(y, alpha) - 0.5 * log_det - 0.5 * n * log(2π)
    catch
        return -Inf
    end
end

# ============================================================
# NEURAL ODE (Euler/RK4 integration for time series)
# ============================================================

"""
Neural ODE: model dh/dt = f(h, t; θ) where f is a small neural network.
Solves with RK4 for the forward pass.
"""
struct NeuralODE
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
    hidden_dim::Int
    state_dim::Int
end

function init_neural_ode(state_dim::Int, hidden_dim::Int;
                          rng::AbstractRNG=Random.default_rng())
    scale = sqrt(2.0 / state_dim)
    return NeuralODE(
        randn(rng, hidden_dim, state_dim) * scale,
        zeros(hidden_dim),
        randn(rng, state_dim, hidden_dim) * sqrt(2.0 / hidden_dim),
        zeros(state_dim),
        hidden_dim,
        state_dim
    )
end

"""Tanh activation."""
@inline function tanh_act(x::Float64)
    ex = exp(2x)
    return (ex - 1) / (ex + 1)
end

"""Evaluate neural ODE vector field f(h, t)."""
function neural_ode_f(node::NeuralODE, h::Vector{Float64}, t::Float64)
    hidden = tanh_act.(node.W1 * h + node.b1)
    return node.W2 * hidden + node.b2
end

"""RK4 step for neural ODE."""
function rk4_step(node::NeuralODE, h::Vector{Float64}, t::Float64, dt::Float64)
    k1 = neural_ode_f(node, h, t)
    k2 = neural_ode_f(node, h + 0.5*dt*k1, t + 0.5*dt)
    k3 = neural_ode_f(node, h + 0.5*dt*k2, t + 0.5*dt)
    k4 = neural_ode_f(node, h + dt*k3, t + dt)
    return h + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
end

"""Solve neural ODE over a time grid."""
function solve_neural_ode(node::NeuralODE, h0::Vector{Float64},
                           t_span::Tuple{Float64, Float64}, n_steps::Int=100)
    t0, t1 = t_span
    dt = (t1 - t0) / n_steps
    times = range(t0, t1, length=n_steps+1)
    states = zeros(n_steps+1, node.state_dim)
    states[1, :] = h0

    h = copy(h0)
    for i in 1:n_steps
        h = rk4_step(node, h, times[i], dt)
        states[i+1, :] = h
    end

    return collect(times), states
end

# ============================================================
# VARIATIONAL AUTOENCODER (VAE)
# ============================================================

"""Initialize VAE with given architecture."""
function init_vae(input_dim::Int, hidden_dims::Vector{Int}, latent_dim::Int;
                   rng::AbstractRNG=Random.default_rng())
    enc_layers = length(hidden_dims)
    enc_w = Vector{Matrix{Float64}}(undef, enc_layers)
    enc_b = Vector{Vector{Float64}}(undef, enc_layers)

    prev = input_dim
    for (l, h) in enumerate(hidden_dims)
        enc_w[l] = randn(rng, h, prev) * sqrt(2.0/prev)
        enc_b[l] = zeros(h)
        prev = h
    end

    mu_w = randn(rng, latent_dim, prev) * sqrt(2.0/prev)
    mu_b = zeros(latent_dim)
    logvar_w = randn(rng, latent_dim, prev) * sqrt(2.0/prev)
    logvar_b = zeros(latent_dim)

    dec_w = Vector{Matrix{Float64}}(undef, enc_layers)
    dec_b = Vector{Vector{Float64}}(undef, enc_layers)

    prev = latent_dim
    for (l, h) in enumerate(reverse(hidden_dims))
        dec_w[l] = randn(rng, h, prev) * sqrt(2.0/prev)
        dec_b[l] = zeros(h)
        prev = h
    end

    # Output layer
    push!(dec_w, randn(rng, input_dim, prev) * sqrt(2.0/prev))
    push!(dec_b, zeros(input_dim))

    return VAEModel(enc_w, enc_b, mu_w, mu_b, logvar_w, logvar_b,
                    dec_w, dec_b, latent_dim, input_dim)
end

"""Encoder forward pass: x → (μ, log σ²)."""
function vae_encode(vae::VAEModel, x::Vector{Float64})
    h = copy(x)
    for (W, b) in zip(vae.encoder_weights, vae.encoder_biases)
        h = tanh_act.(W * h + b)
    end
    mu = vae.mu_weights * h + vae.mu_bias
    logvar = vae.logvar_weights * h + vae.logvar_bias
    return mu, logvar
end

"""Reparameterization trick: z = μ + σ * ε, ε ~ N(0,I)."""
function reparameterize(mu::Vector{Float64}, logvar::Vector{Float64};
                         rng::AbstractRNG=Random.default_rng())
    sigma = exp.(0.5 .* logvar)
    eps = randn(rng, length(mu))
    return mu + sigma .* eps
end

"""Decoder forward pass: z → x̂."""
function vae_decode(vae::VAEModel, z::Vector{Float64})
    h = copy(z)
    for (W, b) in zip(vae.decoder_weights[1:end-1], vae.decoder_biases[1:end-1])
        h = tanh_act.(W * h + b)
    end
    # Output: sigmoid for [0,1] data, linear for general
    return vae.decoder_weights[end] * h + vae.decoder_biases[end]
end

"""
VAE ELBO loss for a single sample.
ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
KL term: -½ Σ(1 + log σ² - μ² - σ²)
"""
function vae_elbo(vae::VAEModel, x::Vector{Float64};
                   rng::AbstractRNG=Random.default_rng())
    mu, logvar = vae_encode(vae, x)
    z = reparameterize(mu, logvar, rng=rng)
    x_hat = vae_decode(vae, z)

    # Reconstruction loss (MSE)
    recon_loss = sum((x - x_hat).^2)

    # KL divergence
    kl_loss = -0.5 * sum(1 .+ logvar - mu.^2 - exp.(logvar))

    return -(recon_loss + kl_loss), recon_loss, kl_loss
end

"""Train VAE with SGD on ELBO."""
function vae_train!(vae::VAEModel, X::Matrix{Float64};
                     lr::Float64=1e-3, n_epochs::Int=50, batch_size::Int=32,
                     rng::AbstractRNG=Random.default_rng())
    n_samples = size(X, 1)
    losses = Float64[]

    for epoch in 1:n_epochs
        epoch_loss = 0.0
        idx = randperm(rng, n_samples)

        for b_start in 1:batch_size:n_samples
            b_end = min(b_start + batch_size - 1, n_samples)
            batch = X[idx[b_start:b_end], :]

            batch_loss = 0.0
            for i in 1:size(batch, 1)
                elbo, _, _ = vae_elbo(vae, batch[i, :], rng=rng)
                batch_loss += elbo
            end
            epoch_loss += batch_loss

            # Numerical gradient update (simplified: perturbation-based for demo)
            # In practice, use automatic differentiation
        end

        push!(losses, -epoch_loss / n_samples)
        epoch % 10 == 0 && println("Epoch $epoch, Loss: $(round(losses[end], digits=4))")
    end

    return losses
end

# ============================================================
# ATTENTION MECHANISM AND TRANSFORMER
# ============================================================

"""
Scaled dot-product attention.
Attention(Q,K,V) = softmax(QK'/√d_k) V
"""
function scaled_dot_product_attention(Q::Matrix{Float64}, K::Matrix{Float64},
                                       V::Matrix{Float64}; mask::Union{Matrix{Bool}, Nothing}=nothing)
    d_k = size(Q, 2)
    scores = Q * K' / sqrt(Float64(d_k))  # n_q x n_k

    if mask !== nothing
        scores[mask] .= -1e9
    end

    # Softmax along rows
    attn_weights = similar(scores)
    for i in 1:size(scores, 1)
        row = scores[i, :]
        row .-= maximum(row)  # numerical stability
        row = exp.(row)
        attn_weights[i, :] = row ./ sum(row)
    end

    return attn_weights * V, attn_weights
end

"""
Multi-head attention.
MultiHead(Q,K,V) = Concat(head_1,...,head_h) W_O
head_i = Attention(Q W_Q_i, K W_K_i, V W_V_i)
"""
function multi_head_attention(enc::TransformerEncoder,
                               Q::Matrix{Float64}, K::Matrix{Float64}, V::Matrix{Float64};
                               mask::Union{Matrix{Bool}, Nothing}=nothing)
    n_seq = size(Q, 1)
    heads = Vector{Matrix{Float64}}(undef, enc.n_heads)

    for h in 1:enc.n_heads
        W_q = enc.W_Q[h, :, :]  # d_k x d_model
        W_k = enc.W_K[h, :, :]
        W_v = enc.W_V[h, :, :]

        Q_h = Q * W_q'  # n_seq x d_k
        K_h = K * W_k'
        V_h = V * W_v'

        attn_output, _ = scaled_dot_product_attention(Q_h, K_h, V_h, mask=mask)
        heads[h] = attn_output
    end

    # Concatenate: n_seq x (n_heads * d_k)
    concat = hcat(heads...)
    return concat * enc.W_O'
end

"""Layer normalization: normalize each sequence position."""
function layer_norm(x::Matrix{Float64}; eps::Float64=1e-6)
    mu = mean(x, dims=2)
    sigma = std(x, dims=2) .+ eps
    return (x .- mu) ./ sigma
end

"""Position-wise feed-forward network."""
function feedforward(enc::TransformerEncoder, x::Matrix{Float64})
    h = max.(0.0, x * enc.W_ff1' .+ enc.b_ff1')  # ReLU
    return h * enc.W_ff2' .+ enc.b_ff2'
end

"""
Sinusoidal positional encoding.
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""
function positional_encoding(seq_len::Int, d_model::Int)
    PE = zeros(seq_len, d_model)
    for pos in 1:seq_len
        for i in 1:2:d_model
            div_term = 10000.0^((i-1) / d_model)
            PE[pos, i] = sin(pos / div_term)
            i+1 <= d_model && (PE[pos, i+1] = cos(pos / div_term))
        end
    end
    return PE
end

"""Initialize transformer encoder."""
function init_transformer(d_model::Int=64, n_heads::Int=4, d_ff::Int=256;
                           rng::AbstractRNG=Random.default_rng())
    @assert d_model % n_heads == 0
    d_k = d_model ÷ n_heads
    scale = sqrt(2.0 / d_model)

    W_Q = randn(rng, n_heads, d_k, d_model) * scale
    W_K = randn(rng, n_heads, d_k, d_model) * scale
    W_V = randn(rng, n_heads, d_k, d_model) * scale
    W_O = randn(rng, d_model, n_heads * d_k) * scale

    W_ff1 = randn(rng, d_ff, d_model) * scale
    W_ff2 = randn(rng, d_model, d_ff) * sqrt(2.0 / d_ff)
    b_ff1 = zeros(d_ff)
    b_ff2 = zeros(d_model)

    return TransformerEncoder(W_Q, W_K, W_V, W_O, W_ff1, W_ff2, b_ff1, b_ff2,
                               n_heads, d_model, d_k, d_ff)
end

"""
Transformer encoder forward pass.
Input: x of shape (seq_len, d_model)
"""
function transformer_forward(enc::TransformerEncoder, x::Matrix{Float64};
                               use_positional::Bool=true)
    seq_len, d_model = size(x)

    if use_positional
        x = x + positional_encoding(seq_len, d_model)
    end

    # Self-attention sublayer
    attn_out = multi_head_attention(enc, x, x, x)
    x = layer_norm(x + attn_out)

    # Feed-forward sublayer
    ff_out = feedforward(enc, x)
    x = layer_norm(x + ff_out)

    return x
end

# ============================================================
# META-LEARNING: MAML (Model-Agnostic Meta-Learning)
# ============================================================

"""Initialize MAML model (simple MLP)."""
function init_maml(input_dim::Int, hidden_dim::Int, output_dim::Int;
                   inner_lr::Float64=0.01, outer_lr::Float64=0.001,
                   n_inner_steps::Int=5,
                   rng::AbstractRNG=Random.default_rng())
    W1 = randn(rng, hidden_dim, input_dim) * sqrt(2.0/input_dim)
    b1 = zeros(hidden_dim)
    W2 = randn(rng, output_dim, hidden_dim) * sqrt(2.0/hidden_dim)
    b2 = zeros(output_dim)

    return MAMLState([W1, W2], [b1, b2], inner_lr, outer_lr, n_inner_steps)
end

"""MAML forward pass (regression)."""
function maml_forward(weights::Vector{Matrix{Float64}}, biases::Vector{Vector{Float64}},
                       x::Matrix{Float64})
    h = x
    n_layers = length(weights)
    for l in 1:n_layers-1
        h = max.(0.0, h * weights[l]' .+ biases[l]')  # ReLU
    end
    return h * weights[end]' .+ biases[end]'
end

"""MAML MSE loss."""
function maml_loss(weights::Vector{Matrix{Float64}}, biases::Vector{Vector{Float64}},
                   X::Matrix{Float64}, y::Matrix{Float64})
    y_pred = maml_forward(weights, biases, X)
    return mean((y - y_pred).^2)
end

"""
MAML inner loop: adapt parameters for a specific task.
θ' = θ - α ∇_θ L_task(θ)
Uses finite difference gradients.
"""
function maml_inner_update(weights::Vector{Matrix{Float64}},
                            biases::Vector{Vector{Float64}},
                            X_support::Matrix{Float64},
                            y_support::Matrix{Float64},
                            inner_lr::Float64, n_steps::Int;
                            eps::Float64=1e-5)
    w = deepcopy(weights)
    b = deepcopy(biases)

    for step in 1:n_steps
        # Compute gradient via finite differences
        loss0 = maml_loss(w, b, X_support, y_support)

        grad_w = [zeros(size(wi)) for wi in w]
        grad_b = [zeros(size(bi)) for bi in b]

        for l in 1:length(w)
            for idx in eachindex(w[l])
                w[l][idx] += eps
                loss_plus = maml_loss(w, b, X_support, y_support)
                w[l][idx] -= eps
                grad_w[l][idx] = (loss_plus - loss0) / eps
            end
            for idx in eachindex(b[l])
                b[l][idx] += eps
                loss_plus = maml_loss(w, b, X_support, y_support)
                b[l][idx] -= eps
                grad_b[l][idx] = (loss_plus - loss0) / eps
            end
        end

        for l in 1:length(w)
            w[l] -= inner_lr * grad_w[l]
            b[l] -= inner_lr * grad_b[l]
        end
    end

    return w, b
end

"""
MAML outer loop: meta-update across tasks.
Tasks: list of (X_support, y_support, X_query, y_query)
"""
function maml_meta_update!(state::MAMLState,
                            tasks::Vector{Tuple{Matrix{Float64}, Matrix{Float64},
                                                Matrix{Float64}, Matrix{Float64}}};
                            eps::Float64=1e-5)
    n_tasks = length(tasks)

    # Collect adapted weights and query losses
    meta_grad_w = [zeros(size(wi)) for wi in state.meta_weights]
    meta_grad_b = [zeros(size(bi)) for bi in state.meta_biases]

    total_query_loss = 0.0

    for (X_s, y_s, X_q, y_q) in tasks
        # Inner update
        w_adapted, b_adapted = maml_inner_update(
            state.meta_weights, state.meta_biases,
            X_s, y_s, state.inner_lr, state.n_inner_steps, eps=eps
        )

        # Query loss with adapted parameters
        q_loss = maml_loss(w_adapted, b_adapted, X_q, y_q)
        total_query_loss += q_loss

        # Compute meta-gradient (finite differences on meta-parameters)
        for l in 1:length(state.meta_weights)
            for idx in eachindex(state.meta_weights[l])
                state.meta_weights[l][idx] += eps
                w_a_p, b_a_p = maml_inner_update(state.meta_weights, state.meta_biases,
                                                   X_s, y_s, state.inner_lr, 1, eps=1e-4)
                loss_plus = maml_loss(w_a_p, b_a_p, X_q, y_q)
                state.meta_weights[l][idx] -= eps
                meta_grad_w[l][idx] += (loss_plus - q_loss) / (eps * n_tasks)
            end
        end
    end

    # Outer update
    for l in 1:length(state.meta_weights)
        state.meta_weights[l] -= state.outer_lr * meta_grad_w[l]
    end

    return total_query_loss / n_tasks
end

"""
Generate synthetic few-shot tasks for regime adaptation.
Each task: different market regime (trend, mean-revert, volatile).
"""
function generate_regime_tasks(n_tasks::Int=10, n_support::Int=20, n_query::Int=10,
                                 input_dim::Int=5;
                                 rng::AbstractRNG=Random.default_rng())
    tasks = Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}[]

    for t in 1:n_tasks
        regime = rand(rng, [:trend, :mean_revert, :volatile])

        # Generate features (lagged returns, vol, etc.)
        X_s = randn(rng, n_support, input_dim)
        X_q = randn(rng, n_query, input_dim)

        # Generate labels based on regime
        if regime == :trend
            y_s = reshape(sum(X_s, dims=2) * 0.1 + randn(rng, n_support) * 0.05, :, 1)
            y_q = reshape(sum(X_q, dims=2) * 0.1 + randn(rng, n_query) * 0.05, :, 1)
        elseif regime == :mean_revert
            y_s = reshape(-sum(X_s[:, 1:2], dims=2) * 0.1 + randn(rng, n_support) * 0.05, :, 1)
            y_q = reshape(-sum(X_q[:, 1:2], dims=2) * 0.1 + randn(rng, n_query) * 0.05, :, 1)
        else
            y_s = reshape(randn(rng, n_support) * 0.2, :, 1)
            y_q = reshape(randn(rng, n_query) * 0.2, :, 1)
        end

        push!(tasks, (X_s, y_s, X_q, y_q))
    end

    return tasks
end

# ============================================================
# UTILITY: NUMERIC GRADIENT
# ============================================================

"""Compute numeric Jacobian of f at x."""
function numeric_jacobian(f::Function, x::Vector{Float64}; eps::Float64=1e-5)
    n = length(x)
    y0 = f(x)
    m = length(y0)
    J = zeros(m, n)
    for i in 1:n
        x_plus = copy(x)
        x_plus[i] += eps
        J[:, i] = (f(x_plus) - y0) / eps
    end
    return J
end

# ============================================================
# DEMO FUNCTIONS
# ============================================================

"""Demo: GP regression on noisy sinusoid."""
function demo_gp(; rng::AbstractRNG=MersenneTwister(42))
    println("=== GP Regression Demo ===")
    n_train = 50
    X_train = reshape(sort(randn(rng, n_train)), :, 1) .* 2
    y_train = sin.(X_train[:, 1]) + randn(rng, n_train) * 0.1

    model = gp_train(X_train, y_train, length_scale=0.5, noise_var=0.01)

    X_test = reshape(range(-4, 4, length=20), :, 1)
    mu, sigma = gp_predict(model, X_test)

    println("GP predictions (mean ± std) at x = [-4, 0, 4]:")
    for (x, m, s) in zip([-4.0, 0.0, 4.0], [mu[1], mu[10], mu[20]], [sigma[1], sigma[10], sigma[20]])
        println("  x=$(x): μ=$(round(m, digits=4)), σ=$(round(s, digits=4))")
    end
end

"""Demo: SVM classification."""
function demo_svm(; rng::AbstractRNG=MersenneTwister(42))
    println("\n=== SVM Classification Demo ===")
    n = 100
    X = randn(rng, n, 2)
    y = Float64.(sign.(X[:, 1] .* X[:, 2]))  # XOR-like problem
    y[y .== 0] .= 1.0

    model = smo_train(X, y, C=1.0, kernel=:rbf,
                      kernel_params=Dict(:length_scale=>1.0, :signal_var=>1.0),
                      max_passes=20)

    labels, scores = svm_predict(model, X)
    acc = mean(labels .== y)
    println("SVM training accuracy: $(round(acc*100, digits=1))%")
    println("Number of support vectors: $(size(model.support_vectors, 1))")
end

"""Demo: MAML meta-learning."""
function demo_maml(; rng::AbstractRNG=MersenneTwister(42))
    println("\n=== MAML Meta-Learning Demo ===")
    state = init_maml(5, 32, 1, inner_lr=0.01, outer_lr=0.001, n_inner_steps=3, rng=rng)

    tasks = generate_regime_tasks(5, 20, 10, 5, rng=rng)
    println("Meta-training on $(length(tasks)) tasks...")

    for meta_iter in 1:5
        loss = maml_meta_update!(state, tasks)
        println("Meta-iter $meta_iter, Query loss: $(round(loss, digits=6))")
    end
end


# ============================================================
# ADDITIONAL ADVANCED ML METHODS
# ============================================================

# ============================================================
# GRADIENT BOOSTING (GBM)
# ============================================================

struct DecisionStump
    feature::Int
    threshold::Float64
    left_val::Float64
    right_val::Float64
end

struct GradientBoostingModel
    stumps::Vector{DecisionStump}
    learning_rate::Float64
    initial_pred::Float64
    loss::Symbol  # :mse, :logloss, :huber
end

"""Fit a decision stump (1-level tree) to residuals."""
function fit_stump(X::Matrix{Float64}, residuals::Vector{Float64})
    n, p = size(X)
    best_sse = Inf; best_feat = 1; best_thr = 0.0; best_lv = 0.0; best_rv = 0.0

    for feat in 1:p
        vals = sort(unique(X[:, feat]))
        length(vals) < 2 && continue
        thresholds = (vals[1:end-1] + vals[2:end]) ./ 2

        for thr in thresholds[1:min(end, 50)]
            left = residuals[X[:, feat] .<= thr]
            right = residuals[X[:, feat] .> thr]
            isempty(left) || isempty(right) && continue
            lv, rv = mean(left), mean(right)
            sse = sum((left .- lv).^2) + sum((right .- rv).^2)
            if sse < best_sse
                best_sse = sse; best_feat = feat; best_thr = thr; best_lv = lv; best_rv = rv
            end
        end
    end

    return DecisionStump(best_feat, best_thr, best_lv, best_rv)
end

"""Predict from a single stump."""
function stump_predict(stump::DecisionStump, X::Matrix{Float64})
    n = size(X, 1)
    return [X[i, stump.feature] <= stump.threshold ? stump.left_val : stump.right_val for i in 1:n]
end

"""
Gradient Boosting for regression.
"""
function gradient_boost_fit(X::Matrix{Float64}, y::Vector{Float64};
                              n_trees::Int=100, learning_rate::Float64=0.1,
                              loss::Symbol=:mse)
    n = length(y)
    f = fill(mean(y), n)
    initial_pred = mean(y)
    stumps = DecisionStump[]

    for t in 1:n_trees
        # Compute negative gradient (pseudo-residuals)
        if loss == :mse
            residuals = y - f
        elseif loss == :huber
            delta = quantile(abs.(y - f), 0.9)
            residuals = map(r -> abs(r) <= delta ? r : delta * sign(r), y - f)
        else  # logloss (for binary classification, y in {0,1})
            probs = 1.0 ./ (1.0 .+ exp.(-f))
            residuals = y - probs
        end

        stump = fit_stump(X, residuals)
        update = stump_predict(stump, X)
        f .+= learning_rate .* update
        push!(stumps, stump)
    end

    return GradientBoostingModel(stumps, learning_rate, initial_pred, loss)
end

"""Predict from gradient boosting model."""
function gradient_boost_predict(model::GradientBoostingModel, X::Matrix{Float64})
    n = size(X, 1)
    f = fill(model.initial_pred, n)
    for stump in model.stumps
        f .+= model.learning_rate .* stump_predict(stump, X)
    end
    if model.loss == :logloss
        return 1.0 ./ (1.0 .+ exp.(-f))
    end
    return f
end

# ============================================================
# RANDOM FOREST
# ============================================================

"""Single decision tree for regression (limited depth)."""
struct TreeNode
    feature::Int
    threshold::Float64
    left::Union{TreeNode, Float64}
    right::Union{TreeNode, Float64}
    depth::Int
end

function build_tree(X::Matrix{Float64}, y::Vector{Float64}, max_depth::Int, min_samples::Int,
                     feature_subset::Vector{Int})
    n = length(y)
    n <= min_samples || max_depth == 0 && return mean(y)

    best_feat, best_thr, best_sse = 1, 0.0, Inf

    for feat in feature_subset
        vals = sort(unique(X[:, feat]))
        length(vals) < 2 && continue
        thresholds = (vals[1:end-1] + vals[2:end]) ./ 2

        for thr in thresholds[1:min(end, 20)]
            left_mask = X[:, feat] .<= thr
            right_mask = .!left_mask
            sum(left_mask) < min_samples || sum(right_mask) < min_samples && continue
            lv = mean(y[left_mask]); rv = mean(y[right_mask])
            sse = sum((y[left_mask] .- lv).^2) + sum((y[right_mask] .- rv).^2)
            if sse < best_sse
                best_sse = sse; best_feat = feat; best_thr = thr
            end
        end
    end

    left_mask = X[:, best_feat] .<= best_thr
    right_mask = .!left_mask

    if sum(left_mask) < min_samples || sum(right_mask) < min_samples
        return mean(y)
    end

    left_node = build_tree(X[left_mask, :], y[left_mask], max_depth-1, min_samples, feature_subset)
    right_node = build_tree(X[right_mask, :], y[right_mask], max_depth-1, min_samples, feature_subset)

    return TreeNode(best_feat, best_thr, left_node, right_node, max_depth)
end

function tree_predict_single(node::Union{TreeNode, Float64}, x::Vector{Float64})
    isa(node, Float64) && return node
    if x[node.feature] <= node.threshold
        return tree_predict_single(node.left, x)
    else
        return tree_predict_single(node.right, x)
    end
end

function tree_predict(node::Union{TreeNode, Float64}, X::Matrix{Float64})
    return [tree_predict_single(node, X[i, :]) for i in 1:size(X, 1)]
end

struct RandomForest
    trees::Vector{Union{TreeNode, Float64}}
    n_features::Int
    feature_importance::Vector{Float64}
end

"""Fit a random forest."""
function random_forest_fit(X::Matrix{Float64}, y::Vector{Float64};
                            n_trees::Int=100, max_depth::Int=5, min_samples::Int=5,
                            n_features::Union{Int, Nothing}=nothing,
                            rng::AbstractRNG=Random.default_rng())
    n, p = size(X)
    n_feat = n_features === nothing ? round(Int, sqrt(p)) : n_features
    trees = Union{TreeNode, Float64}[]
    feat_importance = zeros(p)

    for t in 1:n_trees
        # Bootstrap sample
        boot_idx = rand(rng, 1:n, n)
        X_boot = X[boot_idx, :]
        y_boot = y[boot_idx]

        # Random feature subset
        feat_subset = sort(randperm(rng, p)[1:min(n_feat, p)])

        tree = build_tree(X_boot, y_boot, max_depth, min_samples, feat_subset)
        push!(trees, tree)

        # Feature importance: use split frequency
        if isa(tree, TreeNode)
            feat_importance[tree.feature] += 1.0
        end
    end

    feat_importance ./= max(sum(feat_importance), 1e-10)

    return RandomForest(trees, p, feat_importance)
end

"""Predict from random forest (average of trees)."""
function random_forest_predict(rf::RandomForest, X::Matrix{Float64})
    predictions = [tree_predict(t, X) for t in rf.trees]
    return vec(mean(hcat(predictions...), dims=2))
end

# ============================================================
# BAYESIAN NEURAL NETWORK (simplified: MC Dropout)
# ============================================================

"""Simple MLP with dropout for Bayesian approximation."""
struct BayesianMLP
    weights::Vector{Matrix{Float64}}
    biases::Vector{Vector{Float64}}
    dropout_rate::Float64
end

function init_bnn(input_dim::Int, hidden_dims::Vector{Int}, output_dim::Int;
                   dropout_rate::Float64=0.1,
                   rng::AbstractRNG=Random.default_rng())
    n_layers = length(hidden_dims) + 1
    dims = [input_dim; hidden_dims; output_dim]
    weights = [randn(rng, dims[l+1], dims[l]) * sqrt(2.0/dims[l]) for l in 1:n_layers]
    biases = [zeros(dims[l+1]) for l in 1:n_layers]
    return BayesianMLP(weights, biases, dropout_rate)
end

"""Forward pass with MC dropout."""
function bnn_forward(bnn::BayesianMLP, x::Vector{Float64};
                      training::Bool=true, rng::AbstractRNG=Random.default_rng())
    h = copy(x)
    for (l, (W, b)) in enumerate(zip(bnn.weights, bnn.biases))
        h = W * h + b
        if l < length(bnn.weights)
            # ReLU
            h = max.(h, 0.0)
            # MC Dropout
            if training
                mask = Float64.(rand(rng, length(h)) .> bnn.dropout_rate)
                h .= h .* mask ./ max(1 - bnn.dropout_rate, 1e-10)
            end
        end
    end
    return h
end

"""
Bayesian prediction via MC sampling.
Returns (mean, std) of predictions.
"""
function bnn_predict_mc(bnn::BayesianMLP, X::Matrix{Float64};
                         n_samples::Int=100,
                         rng::AbstractRNG=Random.default_rng())
    n = size(X, 1)
    all_preds = zeros(n_samples, n)

    for s in 1:n_samples
        for i in 1:n
            pred = bnn_forward(bnn, X[i, :], training=true, rng=rng)
            all_preds[s, i] = pred[1]
        end
    end

    return vec(mean(all_preds, dims=1)), vec(std(all_preds, dims=1))
end

# ============================================================
# ONLINE LEARNING: ONLINE GRADIENT DESCENT
# ============================================================

"""Online learning state."""
mutable struct OnlineLearner
    weights::Vector{Float64}
    t::Int                    # time step
    learning_rate::Float64
    method::Symbol            # :ogd, :adagrad, :adam
    grad_sq::Vector{Float64}  # for AdaGrad/Adam
    m::Vector{Float64}        # Adam first moment
    v::Vector{Float64}        # Adam second moment
    beta1::Float64
    beta2::Float64
end

function init_online_learner(d::Int; lr::Float64=0.01, method::Symbol=:adam)
    return OnlineLearner(zeros(d), 0, lr, method, zeros(d), zeros(d), zeros(d), 0.9, 0.999)
end

"""Online update with gradient."""
function online_update!(learner::OnlineLearner, grad::Vector{Float64})
    learner.t += 1
    t = learner.t
    lr = learner.learning_rate

    if learner.method == :ogd
        learner.weights .-= lr .* grad
    elseif learner.method == :adagrad
        learner.grad_sq .+= grad.^2
        learner.weights .-= lr ./ (sqrt.(learner.grad_sq) .+ 1e-8) .* grad
    elseif learner.method == :adam
        learner.m .= learner.beta1 .* learner.m + (1 - learner.beta1) .* grad
        learner.v .= learner.beta2 .* learner.v + (1 - learner.beta2) .* grad.^2
        m_hat = learner.m ./ (1 - learner.beta1^t)
        v_hat = learner.v ./ (1 - learner.beta2^t)
        learner.weights .-= lr .* m_hat ./ (sqrt.(v_hat) .+ 1e-8)
    end
end

"""Online regression: predict then update."""
function online_regression_step!(learner::OnlineLearner, x::Vector{Float64}, y::Float64)
    y_hat = dot(learner.weights, x)
    grad = (y_hat - y) .* x
    online_update!(learner, grad)
    return y_hat
end

"""Run online learning on a dataset."""
function run_online_learning(X::Matrix{Float64}, y::Vector{Float64};
                              method::Symbol=:adam, lr::Float64=0.01)
    T, d = size(X)
    learner = init_online_learner(d, lr=lr, method=method)
    predictions = zeros(T)
    cumulative_regret = zeros(T)

    for t in 1:T
        pred = online_regression_step!(learner, X[t, :], y[t])
        predictions[t] = pred
        loss = (pred - y[t])^2
        cumulative_regret[t] = (t > 1 ? cumulative_regret[t-1] : 0.0) + loss
    end

    return (predictions=predictions, cumulative_regret=cumulative_regret,
            final_weights=copy(learner.weights))
end

# ============================================================
# CLUSTERING: K-MEANS AND HIERARCHICAL
# ============================================================

"""
K-means clustering.
"""
function kmeans(X::Matrix{Float64}, k::Int;
                max_iter::Int=300, tol::Float64=1e-6,
                rng::AbstractRNG=Random.default_rng())
    n, d = size(X)
    # Initialize centroids by k-means++
    centroids = zeros(k, d)
    centroids[1, :] = X[rand(rng, 1:n), :]

    for i in 2:k
        dists = [minimum(sum((X[j, :] - centroids[c, :]).^2) for c in 1:i-1) for j in 1:n]
        probs = dists ./ sum(dists)
        cumprob = cumsum(probs)
        r = rand(rng)
        idx = searchsortedfirst(cumprob, r)
        centroids[i, :] = X[clamp(idx, 1, n), :]
    end

    labels = zeros(Int, n)
    for iter in 1:max_iter
        # Assignment
        new_labels = [argmin([sum((X[i, :] - centroids[c, :]).^2) for c in 1:k]) for i in 1:n]

        # Update centroids
        new_centroids = copy(centroids)
        for c in 1:k
            cluster_pts = X[new_labels .== c, :]
            size(cluster_pts, 1) > 0 && (new_centroids[c, :] = vec(mean(cluster_pts, dims=1)))
        end

        max(norm(new_centroids - centroids)) < tol && (labels = new_labels; break)
        centroids = new_centroids
        labels = new_labels
    end

    # Compute inertia
    inertia = sum(sum((X[i, :] - centroids[labels[i], :]).^2) for i in 1:n)

    return (labels=labels, centroids=centroids, inertia=inertia)
end

"""
Hierarchical clustering (single linkage).
"""
function hierarchical_clustering(X::Matrix{Float64})
    n = size(X, 1)
    # Distance matrix
    D = zeros(n, n)
    for i in 1:n, j in i+1:n
        D[i, j] = D[j, i] = norm(X[i, :] - X[j, :])
    end

    # Each point in its own cluster
    clusters = [[i] for i in 1:n]
    merge_history = Tuple{Int, Int, Float64}[]

    while length(clusters) > 1
        # Find closest pair of clusters
        min_dist = Inf; best_i = 1; best_j = 2

        for i in 1:length(clusters)
            for j in i+1:length(clusters)
                # Single linkage: min distance between any two points
                d = minimum(D[a, b] for a in clusters[i], b in clusters[j])
                if d < min_dist
                    min_dist = d; best_i = i; best_j = j
                end
            end
        end

        push!(merge_history, (best_i, best_j, min_dist))
        new_cluster = vcat(clusters[best_i], clusters[best_j])
        deleteat!(clusters, [best_i, best_j])
        push!(clusters, new_cluster)
    end

    return (merge_history=merge_history, final_cluster=clusters[1])
end

"""
Silhouette score for clustering quality.
"""
function silhouette_score(X::Matrix{Float64}, labels::Vector{Int})
    n = size(X, 1)
    k = maximum(labels)
    scores = zeros(n)

    for i in 1:n
        own_cluster = findall(labels .== labels[i])
        own_cluster = filter(j -> j != i, own_cluster)

        a_i = isempty(own_cluster) ? 0.0 :
              mean(norm(X[i, :] - X[j, :]) for j in own_cluster)

        b_i = Inf
        for c in 1:k
            c == labels[i] && continue
            other = findall(labels .== c)
            isempty(other) && continue
            avg_dist = mean(norm(X[i, :] - X[j, :]) for j in other)
            b_i = min(b_i, avg_dist)
        end

        b_i = isinf(b_i) ? 0.0 : b_i
        denom = max(a_i, b_i)
        scores[i] = denom > 0 ? (b_i - a_i) / denom : 0.0
    end

    return mean(scores)
end

# ============================================================
# DIMENSIONALITY REDUCTION: t-SNE (simplified)
# ============================================================

"""
Simplified t-SNE: Barnes-Hut approximation not implemented.
Uses exact t-SNE for small datasets.
"""
function tsne(X::Matrix{Float64}; n_components::Int=2, perplexity::Float64=30.0,
               n_iter::Int=500, learning_rate::Float64=200.0,
               rng::AbstractRNG=Random.default_rng())
    n, d = size(X)

    # Compute pairwise affinities P
    D2 = zeros(n, n)
    for i in 1:n, j in 1:n
        D2[i, j] = sum((X[i, :] - X[j, :]).^2)
    end

    # Gaussian affinities with adaptive bandwidth (simplified: fixed sigma)
    target_entropy = log(perplexity)
    P = zeros(n, n)
    for i in 1:n
        # Binary search for sigma
        lo, hi = 1e-10, 1e10; sigma_sq = 1.0
        for bs in 1:50
            sigma_sq = (lo + hi) / 2
            p_row = exp.(-D2[i, :] ./ (2 * sigma_sq))
            p_row[i] = 0.0
            sum_p = sum(p_row)
            sum_p < 1e-10 && (hi = sigma_sq; continue)
            p_row ./= sum_p
            entropy = -sum(p_j * log(max(p_j, 1e-10)) for p_j in p_row)
            entropy > target_entropy ? (hi = sigma_sq) : (lo = sigma_sq)
        end
        P[i, :] = exp.(-D2[i, :] ./ (2 * sigma_sq))
        P[i, i] = 0.0
        s = sum(P[i, :]); s > 0 && (P[i, :] ./= s)
    end

    # Symmetrize
    P = (P + P') / 2
    P = max.(P, 1e-12)
    P .*= 4.0 / n  # early exaggeration

    # Initialize embedding
    Y = randn(rng, n, n_components) * 0.01
    Y_prev = copy(Y)
    gains = ones(n, n_components)
    velocity = zeros(n, n_components)

    for iter in 1:n_iter
        # Compute Q (Student-t distribution)
        D2_Y = zeros(n, n)
        for i in 1:n, j in 1:n
            D2_Y[i, j] = sum((Y[i, :] - Y[j, :]).^2)
        end
        Q = 1.0 ./ (1.0 .+ D2_Y)
        for i in 1:n; Q[i, i] = 0.0; end
        Q ./= max(sum(Q), 1e-10)
        Q = max.(Q, 1e-12)

        # Gradient
        PQ = P - Q
        dY = zeros(n, n_components)
        for i in 1:n
            for j in 1:n
                i == j && continue
                factor = 4.0 * PQ[i, j] / (1 + D2_Y[i, j])
                dY[i, :] .+= factor .* (Y[i, :] - Y[j, :])
            end
        end

        # Update with momentum
        momentum = iter < 250 ? 0.5 : 0.8
        gains = (gains .+ 0.2) .* (sign.(dY) .!= sign.(velocity)) .+
                (gains .* 0.8) .* (sign.(dY) .== sign.(velocity))
        gains = max.(gains, 0.01)
        velocity .= momentum .* velocity .- learning_rate .* gains .* dY
        Y .+= velocity
        Y .-= mean(Y, dims=1)

        # Remove early exaggeration
        iter == 100 && (P ./= 4.0)
    end

    return Y
end

# ============================================================
# ISOLATION FOREST (ANOMALY DETECTION)
# ============================================================

struct IsolationTree
    feature::Int
    threshold::Float64
    left::Union{IsolationTree, Int}   # Int = leaf (size)
    right::Union{IsolationTree, Int}
    depth::Int
end

"""Build a single isolation tree."""
function build_isolation_tree(X::Matrix{Float64}, depth::Int, max_depth::Int,
                               rng::AbstractRNG)
    n, p = size(X)
    (n <= 1 || depth >= max_depth) && return n

    feat = rand(rng, 1:p)
    x_col = X[:, feat]
    xmin, xmax = minimum(x_col), maximum(x_col)
    xmin >= xmax && return n

    thr = xmin + rand(rng) * (xmax - xmin)
    left_mask = x_col .<= thr
    right_mask = .!left_mask

    left = build_isolation_tree(X[left_mask, :], depth+1, max_depth, rng)
    right = build_isolation_tree(X[right_mask, :], depth+1, max_depth, rng)

    return IsolationTree(feat, thr, left, right, depth)
end

"""Path length in isolation tree."""
function path_length(tree::Union{IsolationTree, Int}, x::Vector{Float64}, depth::Int)
    isa(tree, Int) && return Float64(depth) + avg_path_length(tree)
    if x[tree.feature] <= tree.threshold
        return path_length(tree.left, x, depth+1)
    else
        return path_length(tree.right, x, depth+1)
    end
end

"""Average path length for n points."""
function avg_path_length(n::Int)
    n <= 1 && return 0.0
    n == 2 && return 1.0
    return 2.0 * (log(n-1) + 0.5772156649) - 2*(n-1)/n
end

"""
Fit Isolation Forest.
"""
function isolation_forest_fit(X::Matrix{Float64}; n_trees::Int=100, sample_size::Int=256,
                               rng::AbstractRNG=Random.default_rng())
    n, p = size(X)
    ss = min(sample_size, n)
    max_depth = round(Int, ceil(log2(ss)))

    trees = Union{IsolationTree, Int}[]
    for _ in 1:n_trees
        idx = rand(rng, 1:n, ss)
        X_sample = X[idx, :]
        push!(trees, build_isolation_tree(X_sample, 0, max_depth, rng))
    end

    return (trees=trees, sample_size=ss)
end

"""Anomaly score: high = anomaly."""
function isolation_forest_score(forest::NamedTuple, X::Matrix{Float64})
    n = size(X, 1)
    scores = zeros(n)
    c = avg_path_length(forest.sample_size)

    for i in 1:n
        avg_len = mean(path_length(t, X[i, :], 0) for t in forest.trees)
        scores[i] = 2.0^(-avg_len / max(c, 1e-10))
    end

    return scores
end

end # module AdvancedML
