"""
SentimentAnalysis.jl
====================
Crypto text sentiment analysis module.

Exports:
  CryptoLexicon         — bullish/bearish word lists with weights
  NaiveBayesSentiment   — log-probability classifier with Laplace smoothing
  LDATopicModel         — variational EM for LDA (K topics)
  SentimentTimeSeries   — rolling index with half-life decay
  NewsImpactStudy       — event study around sentiment regime shifts
  SentimentSignal       — normalize to [-1,1], combine with price signals
  TFIDFVectorizer       — build vocabulary, compute TF-IDF matrix
  ic_sentiment          — IC of sentiment signal vs future returns
"""
module SentimentAnalysis

using Statistics, LinearAlgebra, Random

export CryptoLexicon, NaiveBayesSentiment, LDATopicModel, SentimentTimeSeries,
       NewsImpactStudy, SentimentSignal, TFIDFVectorizer,
       score_document, train!, predict, transform, rolling_sentiment,
       event_study, normalize_signal, combine_signals, ic_sentiment,
       fit_lda!, infer_topics

# ─────────────────────────────────────────────────────────────────────────────
# 1. CRYPTO LEXICON
# ─────────────────────────────────────────────────────────────────────────────

"""
    CryptoLexicon

Bullish/bearish word lists with float sentiment weights.
`bullish_words`: token → positive weight
`bearish_words`: token → negative weight
"""
struct CryptoLexicon
    bullish_words ::Dict{String, Float64}
    bearish_words ::Dict{String, Float64}
    all_words     ::Dict{String, Float64}   # merged
end

function CryptoLexicon()
    bullish = Dict{String, Float64}(
        "moon"        => 1.5, "bull"         => 1.2, "rally"        => 1.1,
        "breakout"    => 1.3, "surge"        => 1.2, "pump"         => 0.8,
        "adoption"    => 1.0, "accumulate"   => 1.0, "bullish"      => 1.3,
        "hodl"        => 0.9, "uptrend"      => 1.1, "buy"          => 0.7,
        "support"     => 0.6, "bounce"       => 0.8, "recovery"     => 0.9,
        "long"        => 0.8, "growth"       => 1.0, "optimistic"   => 1.0,
        "positive"    => 0.7, "strong"       => 0.8, "institutional" => 1.2,
        "etf"         => 1.1, "upgrade"      => 0.9, "partnership"  => 0.8,
        "launch"      => 0.7, "record"       => 0.9, "soar"         => 1.2,
        "rise"        => 0.6, "gain"         => 0.7, "profit"       => 0.8,
        "halving"     => 1.3, "defi"         => 0.8, "layer2"       => 0.7,
        "accumulation"=> 1.1, "oversold"     => 1.0, "undervalued"  => 1.0,
        "capitulation"=> 0.9, "bottom"       => 1.0, "opportunity"  => 0.8,
    )
    bearish = Dict{String, Float64}(
        "crash"       => -1.5, "bear"        => -1.2, "dump"        => -1.3,
        "sell"        => -0.8, "panic"       => -1.4, "fear"        => -1.0,
        "hack"        => -1.5, "scam"        => -1.6, "fraud"       => -1.6,
        "regulation"  => -1.0, "ban"         => -1.4, "bearish"     => -1.3,
        "correction"  => -0.9, "breakdown"   => -1.2, "plunge"      => -1.3,
        "drop"        => -0.8, "fall"        => -0.7, "decline"     => -0.7,
        "loss"        => -0.9, "short"       => -0.8, "resistance"  => -0.5,
        "inflation"   => -0.8, "liquidation" => -1.3, "forced"      => -0.7,
        "default"     => -1.4, "collapse"    => -1.5, "worthless"   => -1.4,
        "bubble"      => -1.1, "overvalued"  => -0.9, "exit"        => -0.7,
        "rug"         => -1.6, "ponzi"       => -1.7, "exploit"     => -1.5,
        "lawsuit"     => -1.2, "insolvency"  => -1.5, "delisting"   => -1.3,
        "overbought"  => -0.8, "distribution"=> -0.7, "overhead"    => -0.5,
    )
    all_words = merge(bullish, bearish)
    return CryptoLexicon(bullish, bearish, all_words)
end

"""
    score_document(lex, doc_tokens) → Float64

Compute sentiment score for a tokenized document.
Returns sum(weight * count) / total_tokens.
"""
function score_document(lex::CryptoLexicon, tokens::Vector{String})
    isempty(tokens) && return 0.0
    total_w = 0.0
    total_n = length(tokens)
    for tok in tokens
        total_w += get(lex.all_words, tok, 0.0)
    end
    return total_w / total_n
end

function score_document(lex::CryptoLexicon, word_counts::Dict{String,Int})
    total_words  = sum(values(word_counts))
    total_words == 0 && return 0.0
    total_w = 0.0
    for (word, cnt) in word_counts
        total_w += get(lex.all_words, word, 0.0) * cnt
    end
    return total_w / total_words
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. TF-IDF VECTORIZER
# ─────────────────────────────────────────────────────────────────────────────

"""
    TFIDFVectorizer

Builds vocabulary and TF-IDF matrix from a corpus.
"""
mutable struct TFIDFVectorizer
    vocab       ::Dict{String, Int}    # word → column index
    idf         ::Vector{Float64}      # IDF values per word
    vocab_size  ::Int
    fitted      ::Bool
    smooth      ::Bool                 # add 1 to df denominator
end

function TFIDFVectorizer(; smooth::Bool=true)
    TFIDFVectorizer(Dict{String,Int}(), Float64[], 0, false, smooth)
end

"""
    fit!(v, docs)

Fit vocabulary and IDF from a corpus of Dict{String,Int} documents.
"""
function fit!(v::TFIDFVectorizer, docs::Vector{Dict{String,Int}})
    # Build vocabulary
    for doc in docs
        for word in keys(doc)
            if !haskey(v.vocab, word)
                v.vocab[word] = length(v.vocab) + 1
            end
        end
    end
    v.vocab_size = length(v.vocab)

    # Document frequency
    N  = length(docs)
    df = zeros(Int, v.vocab_size)
    for doc in docs
        for word in unique(keys(doc))
            idx = get(v.vocab, word, 0)
            if idx > 0; df[idx] += 1; end
        end
    end

    # IDF with smoothing
    if v.smooth
        v.idf = log.((N + 1) ./ (df .+ 1)) .+ 1.0
    else
        v.idf = log.((N .+ 1e-8) ./ (df .+ 1e-8)) .+ 1.0
    end
    v.fitted = true
    return v
end

"""
    transform(v, docs) → Matrix{Float32}

Transform documents to TF-IDF matrix [N_docs × V].
"""
function transform(v::TFIDFVectorizer, docs::Vector{Dict{String,Int}})
    v.fitted || error("Vectorizer not fitted. Call fit! first.")
    N = length(docs)
    X = zeros(Float32, N, v.vocab_size)
    for (i, doc) in enumerate(docs)
        total = sum(values(doc)) + 1e-8f0
        for (word, cnt) in doc
            j = get(v.vocab, word, 0)
            if j > 0
                tf = cnt / total
                X[i, j] = Float32(tf * v.idf[j])
            end
        end
    end
    return X
end

function fit_transform!(v::TFIDFVectorizer, docs::Vector{Dict{String,Int}})
    fit!(v, docs)
    return transform(v, docs)
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. NAIVE BAYES SENTIMENT CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

"""
    NaiveBayesSentiment

Multinomial Naive Bayes with Laplace smoothing.
Classes: -1 (bearish), 0 (neutral), +1 (bullish).
"""
mutable struct NaiveBayesSentiment
    log_likelihoods ::Dict{String, Dict{Int, Float64}}
    log_priors      ::Dict{Int, Float64}
    vocab           ::Set{String}
    vocab_size      ::Int
    alpha           ::Float64    # Laplace smoothing
    threshold       ::Float64    # |score| > threshold → non-neutral label
    fitted          ::Bool
end

function NaiveBayesSentiment(; alpha::Float64=1.0, threshold::Float64=0.1)
    NaiveBayesSentiment(
        Dict{String, Dict{Int,Float64}}(),
        Dict{Int,Float64}(),
        Set{String}(),
        0, alpha, threshold, false
    )
end

function _label(score::Float64, threshold::Float64)
    score > threshold  && return 1
    score < -threshold && return -1
    return 0
end

"""
    train!(nb, docs, scores)

Train Naive Bayes on documents (Dict{String,Int}) with continuous
sentiment scores (labeled via `threshold`).
"""
function train!(nb::NaiveBayesSentiment,
                docs::Vector{Dict{String,Int}},
                scores::Vector{Float64})
    length(docs) == length(scores) || error("docs and scores must be same length")

    labels = [_label(s, nb.threshold) for s in scores]

    # Build vocabulary
    for doc in docs
        for word in keys(doc)
            push!(nb.vocab, word)
        end
    end
    nb.vocab_size = length(nb.vocab)
    vocab_list = collect(nb.vocab)

    # Word counts per class
    class_wc = Dict(c => Dict{String,Float64}() for c in [-1, 0, 1])
    class_total = Dict(c => 0.0 for c in [-1, 0, 1])
    class_n = Dict(c => 0 for c in [-1, 0, 1])

    for (doc, label) in zip(docs, labels)
        class_n[label] += 1
        for (word, cnt) in doc
            class_wc[label][word] = get(class_wc[label], word, 0.0) + cnt
            class_total[label] += cnt
        end
    end

    # Log-likelihoods with Laplace smoothing
    V = nb.vocab_size
    for word in nb.vocab
        nb.log_likelihoods[word] = Dict{Int,Float64}()
        for c in [-1, 0, 1]
            cnt_wc = get(class_wc[c], word, 0.0)
            nb.log_likelihoods[word][c] = log((cnt_wc + nb.alpha) /
                                              (class_total[c] + nb.alpha * V + 1e-8))
        end
    end

    # Log priors
    n_total = length(docs)
    for c in [-1, 0, 1]
        nb.log_priors[c] = log((class_n[c] + 1) / (n_total + 3))
    end

    nb.fitted = true
    return nb
end

"""
    predict(nb, doc) → Int

Predict class (-1, 0, +1) for a single document.
"""
function predict(nb::NaiveBayesSentiment, doc::Dict{String,Int})
    nb.fitted || error("Classifier not trained. Call train! first.")
    log_scores = Dict{Int,Float64}()
    for c in [-1, 0, 1]
        log_scores[c] = nb.log_priors[c]
        for (word, cnt) in doc
            if haskey(nb.log_likelihoods, word)
                log_scores[c] += cnt * nb.log_likelihoods[word][c]
            end
        end
    end
    return argmax(log_scores)
end

"""
    predict_proba(nb, doc) → Dict{Int, Float64}

Return posterior probabilities for each class.
"""
function predict_proba(nb::NaiveBayesSentiment, doc::Dict{String,Int})
    nb.fitted || error("Classifier not trained.")
    log_scores = Dict{Int,Float64}()
    for c in [-1, 0, 1]
        log_scores[c] = nb.log_priors[c]
        for (word, cnt) in doc
            if haskey(nb.log_likelihoods, word)
                log_scores[c] += cnt * nb.log_likelihoods[word][c]
            end
        end
    end
    # Softmax
    max_ls = maximum(values(log_scores))
    exp_scores = Dict(c => exp(log_scores[c] - max_ls) for c in [-1,0,1])
    total = sum(values(exp_scores))
    return Dict(c => exp_scores[c] / total for c in [-1, 0, 1])
end

function predict(nb::NaiveBayesSentiment, docs::Vector{Dict{String,Int}})
    return [predict(nb, d) for d in docs]
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. LDA TOPIC MODEL
# ─────────────────────────────────────────────────────────────────────────────

"""
    LDATopicModel

Latent Dirichlet Allocation via variational EM.
K topics, Dirichlet priors α (doc-topic), β (topic-word).
"""
mutable struct LDATopicModel
    K           ::Int             # number of topics
    alpha       ::Float64         # doc-topic Dirichlet prior
    eta         ::Float64         # topic-word Dirichlet prior
    lambda      ::Matrix{Float64} # K × V topic-word variational params
    vocab       ::Dict{String, Int}
    vocab_size  ::Int
    n_iter      ::Int
    converged   ::Bool
end

function LDATopicModel(K::Int; alpha::Float64=0.1, eta::Float64=0.01, n_iter::Int=100)
    LDATopicModel(K, alpha, eta, zeros(0,0), Dict{String,Int}(), 0, n_iter, false)
end

function _build_vocab_lda(docs::Vector{Dict{String,Int}})
    vocab = Dict{String,Int}()
    for doc in docs
        for word in keys(doc)
            haskey(vocab, word) || (vocab[word] = length(vocab)+1)
        end
    end
    return vocab
end

"""
    fit_lda!(lda, docs)

Fit LDA model using collapsed variational Bayes (simplified).
`docs` is a Vector{Dict{String,Int}} of word count dictionaries.
"""
function fit_lda!(lda::LDATopicModel, docs::Vector{Dict{String,Int}};
                   rng::AbstractRNG=MersenneTwister(42))
    lda.vocab = _build_vocab_lda(docs)
    lda.vocab_size = length(lda.vocab)
    V = lda.vocab_size
    K = lda.K
    D = length(docs)

    # Initialize lambda (topic-word variational parameters) randomly
    lda.lambda = rand(rng, K, V) .+ lda.eta

    # Variational inference: coordinate ascent
    # gamma[d,k]: doc-topic variational param
    gamma  = ones(D, K) .* (lda.alpha + 1.0/K)
    phi    = [ones(K) ./ K for _ in 1:D]   # per-document topic proportions (simplified)

    for iter in 1:lda.n_iter
        lambda_new = fill(lda.eta, K, V)

        for (d, doc) in enumerate(docs)
            # E-step: update phi (topic assignment probabilities per word)
            # Simplified: single-pass update
            log_beta = log.(lda.lambda .+ 1e-10) .- log.(sum(lda.lambda, dims=2) .+ 1e-8)  # K × V
            log_theta_d = log.(gamma[d,:] .+ 1e-10) .- log(sum(gamma[d,:]) + 1e-8)  # K

            for (word, cnt) in doc
                j = get(lda.vocab, word, 0)
                j == 0 && continue
                # log phi_k ∝ log_theta_d[k] + log_beta[k, j]
                log_phi = log_theta_d .+ log_beta[:, j]
                log_phi .-= maximum(log_phi)
                phi_w    = exp.(log_phi)
                phi_w   ./= sum(phi_w) + 1e-10

                # Accumulate to lambda
                for k in 1:K
                    lambda_new[k, j] += cnt * phi_w[k]
                end
                phi[d] .= phi_w
            end

            # M-step (in same loop): update gamma
            for k in 1:K
                doc_phi_sum = sum(cnt * phi[d][k]
                                   for (word, cnt) in doc)
                gamma[d, k] = lda.alpha + doc_phi_sum
            end
        end

        # Update lambda
        delta = maximum(abs.(lambda_new .- lda.lambda))
        lda.lambda .= lambda_new

        if delta < 1e-4
            lda.converged = true
            break
        end
    end

    return lda
end

"""
    infer_topics(lda, doc) → Vector{Float64}

Infer topic distribution for a new document.
Returns K-dimensional vector of topic probabilities.
"""
function infer_topics(lda::LDATopicModel, doc::Dict{String,Int};
                       n_iter::Int=50)
    K = lda.K
    V = lda.vocab_size
    gamma = ones(K) .* (lda.alpha + 1.0/K)
    log_beta = log.(lda.lambda .+ 1e-10) .- log.(sum(lda.lambda, dims=2) .+ 1e-8)

    for _ in 1:n_iter
        gamma_new = fill(lda.alpha, K)
        for (word, cnt) in doc
            j = get(lda.vocab, word, 0)
            j == 0 && continue
            log_theta = log.(gamma .+ 1e-10) .- log(sum(gamma) + 1e-8)
            log_phi   = log_theta .+ log_beta[:, j]
            log_phi  .-= maximum(log_phi)
            phi_w     = exp.(log_phi)
            phi_w    ./= sum(phi_w) + 1e-10
            gamma_new .+= cnt .* phi_w
        end
        gamma .= gamma_new
    end

    theta = gamma ./ (sum(gamma) + 1e-10)
    return theta
end

"""
    top_words(lda, k, n) → Vector{String}

Return top n words for topic k.
"""
function top_words(lda::LDATopicModel, k::Int, n::Int=10)
    lambda_k = lda.lambda[k, :]
    top_idx  = sortperm(lambda_k, rev=true)[1:min(n, length(lambda_k))]
    vocab_inv = Dict(v => k for (k, v) in lda.vocab)
    return [get(vocab_inv, i, "?") for i in top_idx]
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. SENTIMENT TIME SERIES
# ─────────────────────────────────────────────────────────────────────────────

"""
    SentimentTimeSeries

Rolling sentiment index with exponential half-life decay.
"""
mutable struct SentimentTimeSeries
    half_life    ::Float64         # in time units (e.g. days)
    decay_factor ::Float64         # = exp(-log(2) / half_life)
    values       ::Vector{Float64} # daily sentiment scores
    ema          ::Vector{Float64} # exponential MA
    timestamps   ::Vector{Int}
end

function SentimentTimeSeries(half_life::Float64)
    decay = exp(-log(2) / half_life)
    SentimentTimeSeries(half_life, decay, Float64[], Float64[], Int[])
end

"""
    update!(sts, score, t)

Add a new observation to the sentiment time series.
"""
function update!(sts::SentimentTimeSeries, score::Float64, t::Int)
    push!(sts.values, score)
    push!(sts.timestamps, t)

    if isempty(sts.ema)
        push!(sts.ema, score)
    else
        d = sts.decay_factor
        push!(sts.ema, d * last(sts.ema) + (1-d) * score)
    end
end

"""
    rolling_sentiment(raw_scores, window, half_life) → Vector{Float64}

Compute rolling sentiment with exponential weighting over `window` periods.
"""
function rolling_sentiment(raw_scores::Vector{Float64}, window::Int, half_life::Float64)
    n      = length(raw_scores)
    output = fill(NaN, n)
    decay  = exp(-log(2) / half_life)

    for t in window:n
        weights = [decay^(t-s) for s in max(1,t-window+1):t]
        scores  = raw_scores[max(1,t-window+1):t]
        valid   = .!isnan.(scores)
        if sum(valid) < 2; continue; end
        w = weights[valid]
        s = scores[valid]
        output[t] = sum(w .* s) / sum(w)
    end
    return output
end

"""
    fear_greed_index(raw_scores) → Vector{Float64}

Normalize rolling sentiment to [0, 100] Fear & Greed scale.
Values < 25 → extreme fear; > 75 → extreme greed.
"""
function fear_greed_index(raw_scores::Vector{Float64}; window::Int=30,
                           half_life::Float64=7.0)
    smoothed = rolling_sentiment(raw_scores, window, half_life)
    valid    = filter(!isnan, smoothed)
    isempty(valid) && return smoothed

    s_min = quantile(valid, 0.02)
    s_max = quantile(valid, 0.98)
    fg    = [(s - s_min) / (s_max - s_min + 1e-8) * 100 for s in smoothed]
    return clamp.(fg, 0.0, 100.0)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. NEWS IMPACT STUDY
# ─────────────────────────────────────────────────────────────────────────────

"""
    NewsImpactStudy

Event study around sentiment regime shifts.
"""
struct NewsImpactStudy
    window      ::Int              # ±window days around event
    min_shift   ::Float64          # minimum FGI shift to define event
    events      ::Vector{NamedTuple}
    event_returns::Vector{Matrix{Float64}}  # returns around each event
end

function NewsImpactStudy(window::Int=5, min_shift::Float64=20.0)
    NewsImpactStudy(window, min_shift, NamedTuple[], Matrix{Float64}[])
end

"""
    find_events!(study, fgi, returns)

Identify sentiment shift events and store surrounding return windows.
"""
function find_events!(study::NewsImpactStudy, fgi::Vector{Float64},
                       returns::Vector{Float64})
    n = min(length(fgi), length(returns))
    w = study.window

    events   = NamedTuple[]
    ev_rets  = Matrix{Float64}[]

    for t in w+1:n-w
        isnan(fgi[t]) || isnan(fgi[t-5]) && continue
        delta = fgi[t] - fgi[t-5]
        if abs(delta) >= study.min_shift
            push!(events, (t=t, delta=delta, fgi=fgi[t],
                           type=delta > 0 ? "bullish" : "bearish"))
            window_rets = returns[t-w:t+w]
            push!(ev_rets, reshape(window_rets, 1, 2w+1))
        end
    end

    return events, ev_rets
end

"""
    average_event_return(event_returns, direction) → Vector{Float64}

Compute cumulative average return around events of given direction.
"""
function average_event_return(events::Vector{NamedTuple},
                               ev_rets::Vector{Matrix{Float64}},
                               direction::String)
    matching = [(e, r) for (e, r) in zip(events, ev_rets) if e.type == direction]
    isempty(matching) && return Float64[]

    n_cols = size(matching[1][2], 2)
    avg    = zeros(n_cols)
    for (_, r) in matching
        avg .+= vec(r)
    end
    return avg ./ length(matching)
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. SENTIMENT SIGNAL
# ─────────────────────────────────────────────────────────────────────────────

"""
    SentimentSignal

Normalized sentiment signal for trading, combining text sentiment
with optional price momentum.
"""
mutable struct SentimentSignal
    lookback     ::Int
    ema_halflife ::Float64
    raw          ::Vector{Float64}
    normalized   ::Vector{Float64}  # in [-1, 1]
    combined     ::Vector{Float64}  # with price signal if provided
end

function SentimentSignal(lookback::Int=20, ema_halflife::Float64=7.0)
    SentimentSignal(lookback, ema_halflife, Float64[], Float64[], Float64[])
end

"""
    normalize_signal(raw_scores, window) → Vector{Float64}

Map raw sentiment scores to [-1, 1] via rolling rank normalization.
"""
function normalize_signal(raw_scores::Vector{Float64}, window::Int=252)
    n   = length(raw_scores)
    out = fill(NaN, n)
    for t in window:n
        win   = raw_scores[t-window+1:t]
        valid = filter(!isnan, win)
        length(valid) < 5 && continue
        r = mean(valid .< raw_scores[t])
        out[t] = 2.0 * r - 1.0
    end
    return out
end

"""
    combine_signals(sent_signal, price_signal, w_sent, w_price) → Vector{Float64}

Weighted combination of normalized sentiment and price-based signal.
Both signals should be in [-1, 1].
"""
function combine_signals(sent_signal::Vector{Float64}, price_signal::Vector{Float64},
                          w_sent::Float64=0.5, w_price::Float64=0.5)
    n   = min(length(sent_signal), length(price_signal))
    out = Vector{Float64}(undef, n)
    for t in 1:n
        if isnan(sent_signal[t]) && isnan(price_signal[t])
            out[t] = NaN
        elseif isnan(sent_signal[t])
            out[t] = price_signal[t]
        elseif isnan(price_signal[t])
            out[t] = sent_signal[t]
        else
            total = w_sent + w_price
            out[t] = (w_sent * sent_signal[t] + w_price * price_signal[t]) / total
        end
    end
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. IC OF SENTIMENT SIGNAL
# ─────────────────────────────────────────────────────────────────────────────

"""
    ic_sentiment(signal, returns, horizon) → (mean_ic, icir, ics)

Compute Information Coefficient of sentiment signal vs forward returns.
Returns mean IC, IC information ratio, and full IC time series.
"""
function ic_sentiment(signal::Vector{Float64}, returns::Vector{Float64},
                       horizon::Int; min_obs::Int=20)
    n   = min(length(signal), length(returns))
    ics = Float64[]

    for t in 1:n-horizon
        isnan(signal[t]) && continue
        fwd_r = sum(returns[min(t+1,n):min(t+horizon,n)])
        isnan(fwd_r) && continue
        push!(ics, signal[t] * sign(fwd_r) > 0 ? 1.0 : -1.0)
    end

    # Also compute Pearson IC over rolling window
    rolling_ic = Float64[]
    win = 60
    for t in win:n-horizon
        s_win = signal[t-win+1:t]
        r_win = [t2 + horizon <= n ? sum(returns[t2+1:t2+horizon]) : NaN
                  for t2 in t-win+1:t]
        valid = .!isnan.(s_win) .& .!isnan.(r_win)
        sum(valid) < 5 && continue
        push!(rolling_ic, cor(s_win[valid], r_win[valid]))
    end

    mean_ic = isempty(rolling_ic) ? NaN : mean(rolling_ic)
    icir    = isempty(rolling_ic) ? NaN : mean(rolling_ic) / (std(rolling_ic) + 1e-8)
    return (mean_ic=mean_ic, icir=icir, ic_series=rolling_ic)
end

"""
    ic_by_horizon(signal, returns, horizons) → Vector{NamedTuple}

Compute IC at multiple horizons for decay analysis.
"""
function ic_by_horizon(signal::Vector{Float64}, returns::Vector{Float64},
                        horizons::Vector{Int})
    results = NamedTuple[]
    for h in horizons
        r = ic_sentiment(signal, returns, h)
        push!(results, (horizon=h, mean_ic=r.mean_ic, icir=r.icir))
    end
    return results
end

# ─────────────────────────────────────────────────────────────────────────────
# 9. UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

"""
    tokenize(text) → Vector{String}

Simple whitespace+punctuation tokenizer, lowercase.
"""
function tokenize(text::String)
    tokens = String[]
    buf    = Char[]
    for c in lowercase(text)
        if isletter(c) || c == '\''
            push!(buf, c)
        else
            if !isempty(buf)
                push!(tokens, String(buf))
                empty!(buf)
            end
        end
    end
    !isempty(buf) && push!(tokens, String(buf))
    return tokens
end

"""
    bag_of_words(tokens) → Dict{String,Int}

Convert token list to word count dictionary.
"""
function bag_of_words(tokens::Vector{String})
    counts = Dict{String,Int}()
    for tok in tokens
        counts[tok] = get(counts, tok, 0) + 1
    end
    return counts
end

"""
    sentiment_zscore(signal, window) → Vector{Float64}

Compute rolling z-score of raw sentiment signal.
"""
function sentiment_zscore(signal::Vector{Float64}, window::Int=30)
    n   = length(signal)
    out = fill(NaN, n)
    for t in window:n
        win   = signal[t-window+1:t]
        valid = filter(!isnan, win)
        length(valid) < 5 && continue
        mu = mean(valid)
        sd = std(valid) + 1e-8
        out[t] = (signal[t] - mu) / sd
    end
    return out
end

"""
    long_short_sentiment_backtest(signal, returns; q_top, q_bot) → NamedTuple

Simple long-short backtest: long when signal > q_top quantile, short when < q_bot.
"""
function long_short_sentiment_backtest(signal::Vector{Float64},
                                        returns::Vector{Float64};
                                        q_top::Float64=0.75,
                                        q_bot::Float64=0.25,
                                        horizon::Int=1)
    n  = min(length(signal), length(returns))
    valid_s = filter(!isnan, signal)
    isempty(valid_s) && return (sharpe=NaN, total_return=NaN, n_trades=0)

    top_threshold = quantile(valid_s, q_top)
    bot_threshold = quantile(valid_s, q_bot)

    portfolio = Float64[1.0]
    n_trades  = 0

    for t in 1:n-horizon
        isnan(signal[t]) && (push!(portfolio, last(portfolio)); continue)
        fwd = sum(returns[min(t+1,n):min(t+horizon,n)])

        if signal[t] > top_threshold
            pos = 1.0
        elseif signal[t] < bot_threshold
            pos = -1.0
        else
            pos = 0.0
        end

        if pos != 0; n_trades += 1; end
        push!(portfolio, last(portfolio) * (1.0 + pos * fwd))
    end

    port_rets = diff(log.(max.(portfolio, 1e-10)))
    sharpe    = isempty(port_rets) ? NaN :
                mean(port_rets) / (std(port_rets) + 1e-8) * sqrt(252)

    return (sharpe=sharpe, total_return=last(portfolio)-1.0,
            n_trades=n_trades, portfolio=portfolio)
end

# ─────────────────────────────────────────────────────────────────────────────
# 10. EVENT STUDY FRAMEWORK
# ─────────────────────────────────────────────────────────────────────────────

"""
    event_study(event_times, returns, window) → Matrix{Float64}

Compute event-study return matrix: rows = events, cols = [-window:+window].
Returns N_events × (2*window+1) matrix.
"""
function event_study(event_times::Vector{Int}, returns::Vector{Float64},
                      window::Int=5)
    n_events = length(event_times)
    n_ret    = length(returns)
    mat      = fill(NaN, n_events, 2*window+1)

    for (i, t) in enumerate(event_times)
        t_start = t - window
        t_end   = t + window
        (t_start < 1 || t_end > n_ret) && continue
        mat[i, :] = returns[t_start:t_end]
    end

    valid_rows = vec(.!all(isnan, mat, dims=2))
    return mat[valid_rows, :]
end

"""
    car(event_returns, window) → (pre_car, post_car, cumulative)

Compute Cumulative Abnormal Returns from event study matrix.
"""
function car(event_returns::Matrix{Float64}, window::Int=5)
    isempty(event_returns) && return (pre=NaN, post=NaN, cumulative=Float64[])
    avg_rets   = vec(mean(event_returns, dims=1))
    n          = length(avg_rets)
    event_idx  = window + 1
    pre_car    = sum(avg_rets[1:event_idx-1])
    post_car   = sum(avg_rets[event_idx+1:end])
    cumulative = cumsum(avg_rets)
    return (pre=pre_car, post=post_car, cumulative=cumulative)
end

end  # module SentimentAnalysis
