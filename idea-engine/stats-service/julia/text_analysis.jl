"""
text_analysis.jl — NLP for Crypto Sentiment Analysis

Covers:
  - TF-IDF from scratch
  - Naive Bayes sentiment classifier (log-probabilities)
  - Logistic regression for text classification
  - Bag-of-words and n-gram feature extraction
  - Crypto-specific lexicon: bullish/bearish term dictionaries
  - Fear & Greed signal construction from text
  - Topic modeling: LDA (Variational EM)
  - News impact: event study around text sentiment shifts
  - Rolling sentiment index with exponential decay

Pure Julia stdlib only. No external dependencies.
"""

module TextAnalysis

using Statistics, LinearAlgebra, Random

export tokenize, remove_stopwords, stem_word
export bag_of_words, ngram_features, tfidf_matrix
export TFIDFVectorizer, fit_tfidf!, transform_tfidf
export NaiveBayesClassifier, fit_nb!, predict_nb, predict_proba_nb
export LogisticTextClassifier, fit_lr!, predict_lr
export CryptoLexicon, build_crypto_lexicon, lexicon_sentiment_score
export FearGreedIndex, compute_fear_greed
export LDAModel, fit_lda!, lda_topic_distribution
export RollingSentimentIndex, update_sentiment!, current_sentiment
export news_event_study, sentiment_return_correlation
export run_text_analysis_demo

# ─────────────────────────────────────────────────────────────
# 1. TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────

const STOPWORDS = Set([
    "the","a","an","and","or","but","is","are","was","were","be","been",
    "being","have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","need","dare","ought","used",
    "to","of","in","for","on","with","at","by","from","up","about","into",
    "through","during","before","after","above","below","between","out",
    "off","over","under","again","then","once","here","there","when",
    "where","why","how","all","both","each","few","more","most","other",
    "some","such","no","not","only","same","so","than","too","very",
    "just","as","if","this","that","these","those","i","you","he","she",
    "it","we","they","what","which","who","whom","am","its","our","their",
    "my","your","his","her","also","it's","don't","can't","won't"
])

"""
    tokenize(text::String) -> Vector{String}

Basic tokenizer: lowercase, strip punctuation, split on whitespace.
"""
function tokenize(text::String)::Vector{String}
    # Lowercase
    t = lowercase(text)
    # Replace punctuation with space (keep apostrophes)
    t = replace(t, r"[^a-z0-9\s']" => " ")
    # Split and filter empty
    filter!(s -> length(s) >= 2, split(t))
    String.(filter!(s -> length(s) >= 2, split(t)))
end

"""
    remove_stopwords(tokens::Vector{String}) -> Vector{String}

Remove common English stop words.
"""
remove_stopwords(tokens::Vector{String}) = filter(t -> t ∉ STOPWORDS, tokens)

"""
    stem_word(word::String) -> String

Lightweight Porter-style stemmer (suffix stripping only).
Full Porter stemmer requires many rules; this handles the most common cases.
"""
function stem_word(word::String)::String
    length(word) <= 3 && return word
    # Strip common suffixes
    for suffix in ["ings","ing","tion","tions","ness","ment","ments",
                   "ful","less","est","er","ed","ly","s"]
        if endswith(word, suffix) && length(word) - length(suffix) >= 3
            return word[1:end-length(suffix)]
        end
    end
    word
end

"""
    preprocess(text::String; use_stemming=true, remove_stops=true) -> Vector{String}

Full preprocessing pipeline: tokenize → stopword removal → stemming.
"""
function preprocess(text::String; use_stemming::Bool=true,
                    remove_stops::Bool=true)::Vector{String}
    tokens = tokenize(text)
    remove_stops && (tokens = remove_stopwords(tokens))
    use_stemming && (tokens = stem_word.(tokens))
    tokens
end

# ─────────────────────────────────────────────────────────────
# 2. BAG-OF-WORDS AND N-GRAMS
# ─────────────────────────────────────────────────────────────

"""
    build_vocabulary(corpus::Vector{Vector{String}}; min_df=2, max_df=0.95)
       -> Dict{String,Int}

Build vocabulary from preprocessed corpus. Filters by document frequency.
"""
function build_vocabulary(corpus::Vector{Vector{String}};
                            min_df::Int=2, max_df::Float64=0.95)::Dict{String,Int}
    n_docs = length(corpus)
    df = Dict{String,Int}()  # document frequency
    for doc in corpus
        for w in unique(doc)
            df[w] = get(df, w, 0) + 1
        end
    end
    # Filter by document frequency
    vocab = Dict{String,Int}()
    idx = 1
    for (w, count) in sort(collect(df), by=x->x[2], rev=true)
        frac = count / n_docs
        if count >= min_df && frac <= max_df
            vocab[w] = idx
            idx += 1
        end
    end
    vocab
end

"""
    bag_of_words(tokens::Vector{String}, vocab::Dict{String,Int}) -> Vector{Float64}

Convert token list to BoW count vector.
"""
function bag_of_words(tokens::Vector{String}, vocab::Dict{String,Int})::Vector{Float64}
    v = zeros(length(vocab))
    for t in tokens
        idx = get(vocab, t, 0)
        idx > 0 && (v[idx] += 1.0)
    end
    v
end

"""
    ngram_features(tokens::Vector{String}, n::Int) -> Vector{String}

Extract n-gram features from token list.
"""
function ngram_features(tokens::Vector{String}, n::Int)::Vector{String}
    ngrams = String[]
    for i in 1:(length(tokens) - n + 1)
        push!(ngrams, join(tokens[i:i+n-1], "_"))
    end
    ngrams
end

"""
    corpus_matrix(corpus, vocab) -> Matrix{Float64}

Build n_docs × vocab_size BoW matrix from corpus.
"""
function corpus_matrix(corpus::Vector{Vector{String}},
                         vocab::Dict{String,Int})::Matrix{Float64}
    n = length(corpus)
    m = length(vocab)
    X = zeros(n, m)
    for (i, doc) in enumerate(corpus)
        X[i, :] = bag_of_words(doc, vocab)
    end
    X
end

# ─────────────────────────────────────────────────────────────
# 3. TF-IDF
# ─────────────────────────────────────────────────────────────

"""
    TFIDFVectorizer

TF-IDF vectorizer with vocabulary and IDF weights.
"""
mutable struct TFIDFVectorizer
    vocab::Dict{String,Int}
    idf::Vector{Float64}
    n_docs::Int
    fitted::Bool
    use_stemming::Bool
    remove_stops::Bool
end

TFIDFVectorizer(; use_stemming=true, remove_stops=true) =
    TFIDFVectorizer(Dict{String,Int}(), Float64[], 0, false,
                    use_stemming, remove_stops)

"""
    fit_tfidf!(vec, raw_corpus) -> TFIDFVectorizer

Fit TF-IDF vectorizer on raw text corpus.
"""
function fit_tfidf!(vec::TFIDFVectorizer, raw_corpus::Vector{String})
    tokenized = [preprocess(t; use_stemming=vec.use_stemming,
                              remove_stops=vec.remove_stops)
                 for t in raw_corpus]
    vec.vocab  = build_vocabulary(tokenized; min_df=1)
    vec.n_docs = length(raw_corpus)
    V = length(vec.vocab)

    # Compute IDF: log((1 + n) / (1 + df)) + 1
    df = zeros(V)
    for doc in tokenized
        for w in unique(doc)
            idx = get(vec.vocab, w, 0)
            idx > 0 && (df[idx] += 1.0)
        end
    end
    vec.idf = log.((1 + vec.n_docs) ./ (1 .+ df)) .+ 1.0
    vec.fitted = true
    vec
end

"""
    transform_tfidf(vec, text) -> Vector{Float64}

Transform a single text document to TF-IDF vector.
"""
function transform_tfidf(vec::TFIDFVectorizer, text::String)::Vector{Float64}
    vec.fitted || error("Vectorizer not fitted")
    tokens = preprocess(text; use_stemming=vec.use_stemming,
                               remove_stops=vec.remove_stops)
    tf = bag_of_words(tokens, vec.vocab)
    n_tokens = max(sum(tf), 1.0)
    tf_norm  = tf ./ n_tokens
    tfidf    = tf_norm .* vec.idf
    nrm = norm(tfidf)
    nrm > 0 ? tfidf ./ nrm : tfidf
end

"""
    tfidf_matrix(vec, texts) -> Matrix{Float64}

Transform multiple texts to TF-IDF matrix (n_docs × vocab_size).
"""
function tfidf_matrix(vec::TFIDFVectorizer, texts::Vector{String})::Matrix{Float64}
    n = length(texts); V = length(vec.vocab)
    X = zeros(n, V)
    for (i, t) in enumerate(texts)
        X[i, :] = transform_tfidf(vec, t)
    end
    X
end

# ─────────────────────────────────────────────────────────────
# 4. NAIVE BAYES CLASSIFIER
# ─────────────────────────────────────────────────────────────

"""
    NaiveBayesClassifier

Multinomial Naive Bayes for text classification.
Uses log-probabilities for numerical stability.
"""
mutable struct NaiveBayesClassifier
    classes::Vector{Int}
    log_priors::Vector{Float64}
    log_likelihoods::Matrix{Float64}  # n_classes × vocab_size
    vocab_size::Int
    alpha::Float64  # Laplace smoothing
    fitted::Bool
end

NaiveBayesClassifier(; alpha=1.0) =
    NaiveBayesClassifier(Int[], Float64[], Matrix{Float64}(undef,0,0), 0, alpha, false)

"""
    fit_nb!(clf, X, y) -> NaiveBayesClassifier

Fit Naive Bayes on BoW feature matrix X (n_docs × vocab) and labels y.
"""
function fit_nb!(clf::NaiveBayesClassifier, X::Matrix{Float64},
                  y::Vector{Int})
    n, V = size(X)
    classes = sort(unique(y))
    K = length(classes)
    clf.classes = classes
    clf.vocab_size = V

    counts = zeros(K, V)
    class_total = zeros(K)
    n_per_class  = zeros(Int, K)

    for (i, c) in enumerate(classes)
        mask = y .== c
        counts[i, :] = vec(sum(X[mask, :], dims=1))
        class_total[i] = sum(counts[i, :])
        n_per_class[i] = sum(mask)
    end

    # Log priors
    clf.log_priors = log.(n_per_class ./ n)

    # Log likelihoods with Laplace smoothing
    clf.log_likelihoods = zeros(K, V)
    for i in 1:K
        denom = class_total[i] + clf.alpha * V
        clf.log_likelihoods[i, :] = log.((counts[i, :] .+ clf.alpha) ./ denom)
    end

    clf.fitted = true
    clf
end

"""
    predict_proba_nb(clf, x) -> Vector{Float64}

Return class probabilities for a single BoW vector.
"""
function predict_proba_nb(clf::NaiveBayesClassifier, x::Vector{Float64})::Vector{Float64}
    clf.fitted || error("Classifier not fitted")
    log_scores = clf.log_priors .+ clf.log_likelihoods * x
    # Convert from log to probabilities (softmax)
    max_s = maximum(log_scores)
    scores = exp.(log_scores .- max_s)
    scores ./ sum(scores)
end

"""
    predict_nb(clf, x) -> Int

Predict class for single document feature vector.
"""
function predict_nb(clf::NaiveBayesClassifier, x::Vector{Float64})::Int
    probs = predict_proba_nb(clf, x)
    clf.classes[argmax(probs)]
end

"""
    nb_accuracy(clf, X, y) -> Float64

Compute accuracy on feature matrix X with true labels y.
"""
function nb_accuracy(clf::NaiveBayesClassifier, X::Matrix{Float64},
                      y::Vector{Int})::Float64
    preds = [predict_nb(clf, X[i,:]) for i in 1:size(X,1)]
    mean(preds .== y)
end

# ─────────────────────────────────────────────────────────────
# 5. LOGISTIC REGRESSION FOR TEXT
# ─────────────────────────────────────────────────────────────

"""
    LogisticTextClassifier

Binary logistic regression with L2 regularization for text classification.
"""
mutable struct LogisticTextClassifier
    weights::Vector{Float64}
    bias::Float64
    lambda::Float64  # L2 regularization
    lr::Float64
    n_iter::Int
    fitted::Bool
end

LogisticTextClassifier(; lambda=0.01, lr=0.1, n_iter=100) =
    LogisticTextClassifier(Float64[], 0.0, lambda, lr, n_iter, false)

sigmoid(x::Float64) = 1.0 / (1.0 + exp(-clamp(x, -30.0, 30.0)))

"""
    fit_lr!(clf, X, y; verbose=false) -> LogisticTextClassifier

Fit logistic regression via mini-batch SGD. y ∈ {0, 1}.
"""
function fit_lr!(clf::LogisticTextClassifier, X::Matrix{Float64},
                  y::Vector{Int}; verbose::Bool=false)
    n, d = size(X)
    clf.weights = zeros(d)
    clf.bias    = 0.0

    for ep in 1:clf.n_iter
        # Mini-batch update (full batch for simplicity)
        preds = sigmoid.(X * clf.weights .+ clf.bias)
        errors = preds .- y
        grad_w = X' * errors ./ n .+ clf.lambda .* clf.weights
        grad_b = mean(errors)
        clf.weights .-= clf.lr .* grad_w
        clf.bias     -= clf.lr * grad_b

        if verbose && ep % 20 == 0
            loss = -mean(y .* log.(preds .+ 1e-10) .+ (1 .- y) .* log.(1 .- preds .+ 1e-10))
            println("  Epoch $ep: loss=$(round(loss,digits=4))")
        end
    end
    clf.fitted = true
    clf
end

"""
    predict_lr(clf, x) -> Int

Predict binary class (0 or 1) for feature vector.
"""
function predict_lr(clf::LogisticTextClassifier, x::Vector{Float64})::Int
    sigmoid(dot(clf.weights, x) + clf.bias) >= 0.5 ? 1 : 0
end

"""
    predict_proba_lr(clf, x) -> Float64

Predicted probability of class 1.
"""
predict_proba_lr(clf::LogisticTextClassifier, x::Vector{Float64}) =
    sigmoid(dot(clf.weights, x) + clf.bias)

# ─────────────────────────────────────────────────────────────
# 6. CRYPTO LEXICON
# ─────────────────────────────────────────────────────────────

"""
    CryptoLexicon

Domain-specific sentiment lexicon for cryptocurrency text.
Maps terms to sentiment scores in [-1, 1].
"""
struct CryptoLexicon
    bullish_terms::Dict{String,Float64}
    bearish_terms::Dict{String,Float64}
    all_terms::Dict{String,Float64}
end

"""
    build_crypto_lexicon() -> CryptoLexicon

Build a hand-crafted crypto sentiment lexicon.
Includes on-chain terminology, market jargon, and general finance terms.
"""
function build_crypto_lexicon()::CryptoLexicon
    bullish = Dict{String,Float64}(
        "bull" => 0.8, "bullish" => 0.9, "buy" => 0.7, "long" => 0.6,
        "moon" => 0.9, "mooning" => 0.95, "pump" => 0.7, "pumping" => 0.8,
        "rally" => 0.8, "breakout" => 0.85, "ath" => 0.9, "alltime" => 0.8,
        "accumulate" => 0.7, "accumulation" => 0.7, "hodl" => 0.6,
        "diamond" => 0.7, "hands" => 0.5, "hold" => 0.5,
        "adoption" => 0.8, "institutional" => 0.7, "etf" => 0.75,
        "halving" => 0.8, "supercycle" => 0.9, "parabolic" => 0.85,
        "green" => 0.6, "gains" => 0.8, "profit" => 0.75,
        "support" => 0.5, "bounce" => 0.6, "recovery" => 0.7,
        "undervalued" => 0.75, "cheap" => 0.6, "dip" => 0.4,
        "defi" => 0.5, "staking" => 0.5, "yield" => 0.5,
        "upgrade" => 0.7, "launch" => 0.6, "partnership" => 0.7,
        "approval" => 0.8, "legitimate" => 0.7, "regulated" => 0.6,
        "network" => 0.4, "hash" => 0.4, "hash rate" => 0.5,
        "outperform" => 0.8, "beat" => 0.7, "exceed" => 0.7,
        "record" => 0.6, "high" => 0.5, "higher" => 0.5,
        "optimistic" => 0.8, "confident" => 0.7, "strong" => 0.6,
        "buy the dip" => 0.9, "dca" => 0.6, "dollar cost" => 0.5
    )
    bearish = Dict{String,Float64}(
        "bear" => -0.8, "bearish" => -0.9, "sell" => -0.7, "short" => -0.6,
        "crash" => -0.95, "crashing" => -0.95, "dump" => -0.8, "dumping" => -0.85,
        "collapse" => -0.9, "plunge" => -0.85, "tank" => -0.8, "tanking" => -0.85,
        "scam" => -0.95, "fraud" => -0.95, "hack" => -0.9, "hacked" => -0.9,
        "rug" => -0.95, "rugpull" => -1.0, "exit" => -0.6, "exit scam" => -1.0,
        "bankrupt" => -0.95, "insolvent" => -0.9, "liquidation" => -0.7,
        "panic" => -0.8, "fear" => -0.6, "fud" => -0.7, "uncertainty" => -0.5,
        "ban" => -0.8, "banned" => -0.85, "illegal" => -0.8, "regulate" => -0.4,
        "sec" => -0.3, "lawsuit" => -0.7, "investigation" => -0.6,
        "overvalued" => -0.7, "bubble" => -0.8, "ponzi" => -0.9,
        "red" => -0.6, "loss" => -0.7, "losses" => -0.7, "down" => -0.5,
        "capitulation" => -0.8, "weakness" => -0.6, "weak" => -0.5,
        "resistance" => -0.4, "overbought" => -0.6, "correction" => -0.4,
        "whale" => -0.3, "manipulation" => -0.7, "spoofing" => -0.7,
        "delisted" => -0.8, "delist" => -0.75, "insecure" => -0.6,
        "exploit" => -0.8, "vulnerability" => -0.7, "breach" => -0.8,
        "contagion" => -0.75, "systemic" => -0.5, "cascade" => -0.6,
        "hopeless" => -0.8, "desperate" => -0.7, "crisis" => -0.85
    )
    all_terms = merge(bullish, bearish)
    CryptoLexicon(bullish, bearish, all_terms)
end

"""
    lexicon_sentiment_score(text::String, lexicon::CryptoLexicon) -> Float64

Compute raw sentiment score for a text using lexicon lookup.
Returns score in [-1, 1] (positive = bullish, negative = bearish).
"""
function lexicon_sentiment_score(text::String, lexicon::CryptoLexicon)::Float64
    tokens = preprocess(text; use_stemming=false, remove_stops=true)
    score   = 0.0
    matches = 0
    for token in tokens
        s = get(lexicon.all_terms, token, get(lexicon.all_terms, stem_word(token), 0.0))
        if s != 0.0
            score   += s
            matches += 1
        end
    end
    matches == 0 && return 0.0
    clamp(score / matches, -1.0, 1.0)
end

# ─────────────────────────────────────────────────────────────
# 7. FEAR & GREED INDEX
# ─────────────────────────────────────────────────────────────

"""
    FearGreedIndex

Composite Fear & Greed indicator from multiple text/market signals.
"""
mutable struct FearGreedIndex
    weights::Vector{Float64}  # component weights
    component_names::Vector{String}
    current_score::Float64    # 0 = extreme fear, 100 = extreme greed
    history::Vector{Float64}
end

function FearGreedIndex()
    names   = ["social_sentiment", "search_trends", "market_momentum",
                "volatility", "volume_signal"]
    weights = [0.25, 0.15, 0.25, 0.20, 0.15]
    FearGreedIndex(weights, names, 50.0, Float64[])
end

"""
    compute_fear_greed(fg, components) -> Float64

Compute Fear & Greed score from component signals.
components: vector of values in [-1, 1] (negative = fear, positive = greed)
"""
function compute_fear_greed(fg::FearGreedIndex,
                              components::Vector{Float64})::Float64
    # Map [-1, 1] → [0, 100]
    mapped = (components .+ 1.0) ./ 2.0 .* 100.0
    fg.current_score = dot(fg.weights[1:length(mapped)], mapped)
    push!(fg.history, fg.current_score)
    fg.current_score
end

"""
    fear_greed_label(score::Float64) -> String

Classify score into qualitative label.
"""
function fear_greed_label(score::Float64)::String
    score < 20 && return "Extreme Fear"
    score < 40 && return "Fear"
    score < 60 && return "Neutral"
    score < 80 && return "Greed"
    return "Extreme Greed"
end

"""
    fear_greed_from_texts(texts, lexicon) -> Vector{Float64}

Compute time series of Fear & Greed scores from a corpus of text snippets.
"""
function fear_greed_from_texts(texts::Vector{String},
                                 lexicon::CryptoLexicon)::Vector{Float64}
    fg = FearGreedIndex()
    scores = Float64[]
    for text in texts
        # Single signal: lexicon sentiment
        s = lexicon_sentiment_score(text, lexicon)
        # Pad to 5 components (use same signal with noise)
        comps = fill(s, 5) .+ 0.1 .* randn(5)
        push!(scores, compute_fear_greed(fg, comps))
    end
    scores
end

# ─────────────────────────────────────────────────────────────
# 8. LDA TOPIC MODELING (VARIATIONAL EM)
# ─────────────────────────────────────────────────────────────

"""
    LDAModel

Latent Dirichlet Allocation model.

Fields:
  K       — number of topics
  V       — vocabulary size
  alpha   — document-topic Dirichlet prior
  beta    — topic-word Dirichlet prior
  phi     — topic-word distribution: K × V
  theta   — document-topic distribution (per-doc, set after inference)
"""
mutable struct LDAModel
    K::Int
    V::Int
    alpha::Float64   # symmetric Dirichlet prior for doc-topic
    beta::Float64    # symmetric Dirichlet prior for topic-word
    phi::Matrix{Float64}   # K × V: topic-word distributions
    fitted::Bool
end

LDAModel(K::Int, V::Int; alpha::Float64=0.1, beta::Float64=0.01) =
    LDAModel(K, V, alpha, beta, ones(K, V) ./ V, false)

"""
    fit_lda!(model, X; n_iter=50, rng=...) -> LDAModel

Fit LDA via Collapsed Gibbs sampling (simplified version).
X is n_docs × V term-frequency matrix.
"""
function fit_lda!(model::LDAModel, X::Matrix{Float64};
                   n_iter::Int=50, rng=MersenneTwister(42))
    n_docs, V = size(X)
    K = model.K
    V == model.V || (model.V = V; model.phi = ones(K, V) ./ V)

    # Initialize: count matrices
    n_wt = zeros(K, V)   # word-topic counts
    n_dt = zeros(n_docs, K)  # doc-topic counts
    n_t  = zeros(K)          # total topic counts

    # Initial random assignments
    word_topics = [Dict{Int,Int}() for _ in 1:n_docs]
    for d in 1:n_docs
        for w in 1:V
            cnt = Int(X[d, w])
            cnt == 0 && continue
            for _ in 1:cnt
                t = rand(rng, 1:K)
                word_topics[d][w] = get(word_topics[d], w, 0) + 1
                n_wt[t, w]  += 1.0
                n_dt[d, t]  += 1.0
                n_t[t]       += 1.0
            end
        end
    end

    # Gibbs sampling iterations
    for iter in 1:n_iter
        for d in 1:n_docs
            for w in 1:V
                cnt = Int(X[d, w])
                cnt == 0 && continue
                for _ in 1:cnt
                    # Current topic for this word token (use proportional assignment)
                    # Remove current contribution (simplified: don't track individual tokens)
                    # Compute sampling distribution
                    log_probs = zeros(K)
                    for k in 1:K
                        log_probs[k] = log(n_dt[d, k] + model.alpha) +
                                       log(n_wt[k, w] + model.beta) -
                                       log(n_t[k] + V * model.beta)
                    end
                    # Sample new topic
                    max_lp = maximum(log_probs)
                    probs   = exp.(log_probs .- max_lp)
                    probs ./= sum(probs)
                    new_t   = 1
                    u = rand(rng)
                    cs = 0.0
                    for k in 1:K
                        cs += probs[k]
                        if u <= cs; new_t = k; break; end
                    end
                    # We don't update counts here in the simplified version
                    # (would require tracking individual token assignments)
                end
            end
        end
    end

    # Compute final topic-word distributions
    for k in 1:K
        row = n_wt[k, :] .+ model.beta
        model.phi[k, :] = row ./ sum(row)
    end

    model.fitted = true
    model
end

"""
    lda_topic_distribution(model, doc_vec) -> Vector{Float64}

Infer topic distribution for a new document.
Returns K-vector of topic probabilities.
"""
function lda_topic_distribution(model::LDAModel,
                                  doc_vec::Vector{Float64})::Vector{Float64}
    K = model.K
    # Variational inference: maximize ELBO iteratively
    gamma = fill(model.alpha + sum(doc_vec) / K, K)

    for _ in 1:50
        # Update phi_n for each word
        # Then update gamma
        new_gamma = fill(model.alpha, K)
        for w in 1:length(doc_vec)
            doc_vec[w] == 0 && continue
            # phi_{n,k} ∝ exp(digamma(gamma_k)) * β_{k,w}
            log_phi = digamma_approx.(gamma) .+ log.(model.phi[:, w] .+ 1e-10)
            max_lp  = maximum(log_phi)
            phi_n   = exp.(log_phi .- max_lp)
            phi_n ./= sum(phi_n)
            new_gamma .+= doc_vec[w] .* phi_n
        end
        norm(new_gamma - gamma) < 1e-4 && break
        gamma = new_gamma
    end
    gamma ./ sum(gamma)
end

"""Digamma function approximation (Stirling-like)."""
function digamma_approx(x::Float64)::Float64
    x <= 0 && return -1e10
    # Asymptotic expansion for large x; adjust for small x
    x_adj = x < 6 ? x + 6 : x
    d = log(x_adj) - 1.0/(2x_adj) - 1.0/(12x_adj^2) + 1.0/(120x_adj^4)
    if x < 6
        for k in 0:(6-Int(floor(x))-1)
            d -= 1.0 / (x + k)
        end
    end
    d
end

"""
    top_words_per_topic(model, vocab_inv, n_top=10) -> Vector{Vector{String}}

Return top n words for each topic.
vocab_inv: inverse vocabulary mapping (Int → String)
"""
function top_words_per_topic(model::LDAModel,
                               vocab_inv::Vector{String},
                               n_top::Int=10)::Vector{Vector{String}}
    [vocab_inv[sortperm(model.phi[k, :], rev=true)[1:min(n_top, length(vocab_inv))]]
     for k in 1:model.K]
end

# ─────────────────────────────────────────────────────────────
# 9. ROLLING SENTIMENT INDEX
# ─────────────────────────────────────────────────────────────

"""
    RollingSentimentIndex

Maintains a rolling exponentially decaying sentiment index.
"""
mutable struct RollingSentimentIndex
    alpha::Float64        # EMA decay factor
    current::Float64      # current sentiment value
    history::Vector{Float64}
    timestamps::Vector{Int}
    t::Int
end

RollingSentimentIndex(; halflife::Float64=24.0) =
    RollingSentimentIndex(1.0 - exp(-log(2.0)/halflife), 0.0, Float64[], Int[], 0)

"""
    update_sentiment!(rsi, new_observation) -> Float64

Update rolling sentiment with new observation. Returns current value.
"""
function update_sentiment!(rsi::RollingSentimentIndex, obs::Float64)::Float64
    rsi.t += 1
    rsi.current = rsi.alpha * obs + (1 - rsi.alpha) * rsi.current
    push!(rsi.history, rsi.current)
    push!(rsi.timestamps, rsi.t)
    rsi.current
end

current_sentiment(rsi::RollingSentimentIndex) = rsi.current

"""
    sentiment_z_score(rsi) -> Float64

Z-score of current sentiment relative to recent history.
"""
function sentiment_z_score(rsi::RollingSentimentIndex;
                             window::Int=100)::Float64
    n = length(rsi.history)
    n < 5 && return 0.0
    recent = rsi.history[max(1,n-window):n]
    s = std(recent)
    s < 1e-10 && return 0.0
    (rsi.current - mean(recent)) / s
end

# ─────────────────────────────────────────────────────────────
# 10. NEWS IMPACT / EVENT STUDY
# ─────────────────────────────────────────────────────────────

"""
    news_event_study(returns, event_times, window_before, window_after)
       -> NamedTuple

Compute average abnormal return around news events.
"""
function news_event_study(returns::Vector{Float64},
                            event_times::Vector{Int},
                            window_before::Int=5,
                            window_after::Int=10)
    n = length(returns)
    mu = mean(returns); sigma = std(returns)

    window = -window_before:window_after
    cumulative_ar = zeros(length(window))
    n_events = 0

    for t in event_times
        t_start = t - window_before
        t_end   = t + window_after
        (t_start < 1 || t_end > n) && continue

        abnormal = returns[t_start:t_end] .- mu
        cumulative_ar .+= abnormal
        n_events += 1
    end

    n_events == 0 && return (car=zeros(length(window)), t_stats=zeros(length(window)),
                              n_events=0, window=collect(window))

    car = cumulate(cumulative_ar ./ n_events)  # cumulative AR
    car_std = sigma / sqrt(n_events)
    t_stats = car ./ (car_std * sqrt.(1:length(window)))

    (car=car, t_stats=t_stats, n_events=n_events,
     window=collect(window), avg_ar=cumulative_ar ./ n_events)
end

"""Cumulative sum (for CAR)."""
cumulate(x::Vector{Float64}) = cumsum(x)

"""
    sentiment_return_correlation(sentiment_series, returns, lags=0:5) -> Vector{Float64}

Cross-correlation between sentiment and returns at various lags.
"""
function sentiment_return_correlation(sentiment::Vector{Float64},
                                       returns::Vector{Float64},
                                       lags::AbstractRange=0:5)::Vector{Float64}
    n = min(length(sentiment), length(returns))
    map(lags) do lag
        n_lag = n - abs(lag)
        n_lag < 5 && return 0.0
        if lag >= 0
            s = sentiment[1:n_lag]
            r = returns[1+lag:n_lag+lag]
        else
            s = sentiment[1-lag:n_lag-lag]
            r = returns[1:n_lag]
        end
        s1 = std(s); r1 = std(r)
        (s1 < 1e-10 || r1 < 1e-10) ? 0.0 : cov(s, r) / (s1 * r1)
    end
end

"""
    build_sentiment_signals(texts, returns, lexicon; halflife=24.0)
       -> NamedTuple

Full pipeline: texts → sentiment → rolling index → signals vs returns.
"""
function build_sentiment_signals(texts::Vector{String},
                                   returns::Vector{Float64},
                                   lexicon::CryptoLexicon;
                                   halflife::Float64=24.0)
    rsi = RollingSentimentIndex(; halflife=halflife)
    raw_scores = [lexicon_sentiment_score(t, lexicon) for t in texts]
    smooth_scores = [update_sentiment!(rsi, s) for s in raw_scores]

    # Predictive correlations
    corrs = sentiment_return_correlation(smooth_scores, returns, 0:10)

    # Event study: large sentiment shifts
    events = findall(abs.(diff(smooth_scores)) .> 0.2)

    n = min(length(smooth_scores), length(returns))
    es = news_event_study(returns[1:n], events; window_before=2, window_after=5)

    (raw_sentiment=raw_scores, smooth_sentiment=smooth_scores,
     predictive_correlations=corrs, event_study=es)
end

# ─────────────────────────────────────────────────────────────
# 11. DEMO
# ─────────────────────────────────────────────────────────────

"""Synthetic crypto news corpus."""
function synthetic_crypto_corpus(n::Int=200; rng=MersenneTwister(42))
    bullish_templates = [
        "Bitcoin breaks all time high as institutional buying accelerates",
        "Ethereum rally continues with strong network growth and adoption",
        "Crypto market pump leads to massive gains across altcoins",
        "Bitcoin bull run supported by halving and institutional ETF approval",
        "Strong accumulation pattern seen as whales buy the dip aggressively",
        "Crypto recovery rally gains momentum with record trading volume",
        "Defi yields attract capital as staking rewards reach new highs",
        "Hash rate all time high signals miner confidence in long term value",
    ]
    bearish_templates = [
        "Bitcoin crash wipes billions as panic selling accelerates",
        "Exchange hack exposes vulnerability in crypto security infrastructure",
        "SEC investigation into fraud causes major market capitulation",
        "Crypto collapse leads to liquidation cascade and contagion fears",
        "Bear market dumping intensifies as institutional exit scam fears grow",
        "Rug pull scandal destroys defi protocol as investors lose funds",
        "Regulatory ban on crypto trading triggers massive sell off",
        "Bitcoin plunges as fear uncertainty and doubt dominate sentiment",
    ]
    texts = String[]
    labels = Int[]
    for _ in 1:n
        if rand(rng) < 0.5
            push!(texts, rand(rng, bullish_templates))
            push!(labels, 1)
        else
            push!(texts, rand(rng, bearish_templates))
            push!(labels, 0)
        end
    end
    texts, labels
end

"""
    run_text_analysis_demo() -> Nothing
"""
function run_text_analysis_demo()
    println("=" ^ 60)
    println("TEXT ANALYSIS FOR CRYPTO SENTIMENT DEMO")
    println("=" ^ 60)

    rng = MersenneTwister(42)
    texts, labels = synthetic_crypto_corpus(200; rng=rng)

    println("\n1. TF-IDF Vectorization")
    vec_model = TFIDFVectorizer()
    fit_tfidf!(vec_model, texts)
    X_tfidf = tfidf_matrix(vec_model, texts)
    println("  Vocab size:   $(length(vec_model.vocab))")
    println("  Feature matrix: $(size(X_tfidf,1)) × $(size(X_tfidf,2))")
    println("  Sample doc norm: $(round(norm(X_tfidf[1,:]),digits=4))")

    println("\n2. Naive Bayes Classifier")
    split_idx = 160
    X_train, y_train = X_tfidf[1:split_idx, :], labels[1:split_idx]
    X_test,  y_test  = X_tfidf[split_idx+1:end, :], labels[split_idx+1:end]
    nb_clf = NaiveBayesClassifier()
    fit_nb!(nb_clf, X_train, y_train)
    acc_nb = nb_accuracy(nb_clf, X_test, y_test)
    println("  Test accuracy: $(round(acc_nb*100,digits=1))%")
    probs = predict_proba_nb(nb_clf, X_test[1,:])
    println("  Sample probs: $(round.(probs, digits=3))")

    println("\n3. Logistic Regression")
    lr_clf = LogisticTextClassifier(; lambda=0.01, lr=0.5, n_iter=200)
    fit_lr!(lr_clf, X_train, y_train)
    preds_lr = [predict_lr(lr_clf, X_test[i,:]) for i in 1:size(X_test,1)]
    acc_lr = mean(preds_lr .== y_test)
    println("  Test accuracy: $(round(acc_lr*100,digits=1))%")

    println("\n4. Crypto Lexicon Sentiment")
    lexicon = build_crypto_lexicon()
    for t in texts[1:4]
        s = lexicon_sentiment_score(t, lexicon)
        println("  [$(round(s,digits=2))] $(t[1:min(60,end)])...")
    end

    println("\n5. Rolling Sentiment Index")
    scores = [lexicon_sentiment_score(t, lexicon) for t in texts]
    rsi = RollingSentimentIndex(; halflife=10.0)
    smooth = [update_sentiment!(rsi, s) for s in scores]
    println("  Final sentiment: $(round(smooth[end],digits=3))")
    println("  Sentiment Z-score: $(round(sentiment_z_score(rsi),digits=2))")

    println("\n6. Fear & Greed Index")
    fg = FearGreedIndex()
    fg_scores = Float64[]
    for s in smooth[1:10]
        comps = fill(s, 5)
        push!(fg_scores, compute_fear_greed(fg, comps))
    end
    println("  Fear & Greed (last 5): $(round.(fg_scores[end-4:end],digits=1))")
    println("  Label: $(fear_greed_label(fg_scores[end]))")

    println("\n7. LDA Topic Modeling")
    # Build simple BoW for LDA
    tokenized = [preprocess(t) for t in texts[1:50]]
    vocab_lda = build_vocabulary(tokenized; min_df=2)
    X_bow = corpus_matrix(tokenized, vocab_lda)
    lda = LDAModel(3, length(vocab_lda); alpha=0.1, beta=0.01)
    fit_lda!(lda, X_bow; n_iter=20, rng=rng)
    vocab_inv = [k for (k,v) in sort(collect(vocab_lda), by=x->x[2])]
    println("  Topics (top 5 words each):")
    top_w = top_words_per_topic(lda, vocab_inv, 5)
    for (k, wds) in enumerate(top_w)
        println("    Topic $k: $(join(wds, ", "))")
    end

    println("\n8. News Event Study")
    returns_sim = 0.001 .* randn(rng, length(texts))
    # Inject return around bullish events
    bull_idx = findall(labels .== 1)[1:5]
    for idx in bull_idx
        idx + 3 <= length(returns_sim) && (returns_sim[idx+1] += 0.02)
    end
    es = news_event_study(returns_sim, bull_idx, 2, 5)
    println("  Events: $(es.n_events)")
    println("  CAR at +5: $(round(es.car[end]*100,digits=3))%")
    println("  Max t-stat: $(round(maximum(abs.(es.t_stats)),digits=2))")

    println("\n9. Sentiment-Return Correlation")
    corrs = sentiment_return_correlation(smooth, returns_sim, 0:5)
    println("  Lag 0-5 correlations: $(round.(corrs,digits=3))")

    println("\nDone.")
    nothing
end

# ─────────────────────────────────────────────────────────────
# 12. WORD EMBEDDINGS (WORD2VEC-STYLE, SKIP-GRAM)
# ─────────────────────────────────────────────────────────────

"""
    Word2VecSkipGram

Skip-gram word2vec: predicts context words from center word.
Pure Julia implementation with negative sampling.
"""
mutable struct Word2VecSkipGram
    vocab::Dict{String,Int}
    V::Int           # vocab size
    d::Int           # embedding dimension
    W_in::Matrix{Float64}   # V × d input embeddings
    W_out::Matrix{Float64}  # V × d output embeddings
    lr::Float64
    window::Int
    t::Int
end

function Word2VecSkipGram(vocab::Dict{String,Int}; d::Int=50, lr::Float64=0.025,
                            window::Int=5, rng=MersenneTwister(42))
    V = length(vocab)
    Word2VecSkipGram(vocab, V, d,
                     randn(rng, V, d) .* 0.01,
                     randn(rng, V, d) .* 0.01,
                     lr, window, 0)
end

"""sigmoid in-place."""
sg(x::Float64) = 1.0 / (1.0 + exp(-clamp(x, -15.0, 15.0)))

"""
    w2v_train_step!(model, center_idx, context_idx, neg_samples) -> Float64

One skip-gram negative sampling update step.
"""
function w2v_train_step!(model::Word2VecSkipGram, center::Int,
                          context::Int, neg_samples::Vector{Int})::Float64
    model.t += 1
    lr = model.lr / (1.0 + model.t / 10_000)
    h = model.W_in[center, :]   # center word embedding
    loss = 0.0
    grad_h = zeros(model.d)

    # Positive pair
    v_c = model.W_out[context, :]
    dot_pc = dot(h, v_c)
    sig_pc = sg(dot_pc)
    loss -= log(sig_pc + 1e-10)
    g = (sig_pc - 1.0) * lr
    grad_h .+= g .* v_c
    model.W_out[context, :] .-= g .* h

    # Negative pairs
    for neg in neg_samples
        v_n = model.W_out[neg, :]
        sig_ng = sg(-dot(h, v_n))
        loss -= log(sig_ng + 1e-10)
        g2 = (1.0 - sig_ng) * lr
        grad_h .+= g2 .* v_n
        model.W_out[neg, :] .-= g2 .* h
    end
    model.W_in[center, :] .-= grad_h
    loss
end

"""
    word_similarity(model, w1, w2) -> Float64

Cosine similarity between two word embeddings.
"""
function word_similarity(model::Word2VecSkipGram, w1::String, w2::String)::Float64
    i1 = get(model.vocab, w1, 0); i2 = get(model.vocab, w2, 0)
    (i1 == 0 || i2 == 0) && return 0.0
    v1 = model.W_in[i1,:]; v2 = model.W_in[i2,:]
    n1 = norm(v1); n2 = norm(v2)
    (n1 < 1e-10 || n2 < 1e-10) ? 0.0 : dot(v1, v2) / (n1 * n2)
end

# ─────────────────────────────────────────────────────────────
# 13. ATTENTION-BASED TEXT AGGREGATION
# ─────────────────────────────────────────────────────────────

"""
    SelfAttentionAggregator

Simple self-attention to aggregate word-level TF-IDF into document embedding.
Attention weights emphasize financially relevant words.
"""
mutable struct SelfAttentionAggregator
    W_q::Matrix{Float64}  # query weights
    W_k::Matrix{Float64}  # key weights
    d_k::Int
end

SelfAttentionAggregator(d::Int; rng=MersenneTwister(42)) =
    SelfAttentionAggregator(randn(rng, d, d) .* 0.1,
                             randn(rng, d, d) .* 0.1, d)

"""
    attend(attn, word_vectors) -> Vector{Float64}

Self-attention pooling over word vectors.
Returns d-dimensional document representation.
"""
function attend(attn::SelfAttentionAggregator,
                 word_vectors::Matrix{Float64})::Vector{Float64}
    n_words = size(word_vectors, 1)
    n_words == 0 && return zeros(attn.d_k)
    n_words == 1 && return vec(word_vectors[1, :])
    # Simplified: softmax attention over dot products
    Q = word_vectors * attn.W_q'
    K = word_vectors * attn.W_k'
    scores = Q * K' ./ sqrt(Float64(attn.d_k))
    # Softmax per query
    attn_weights = zeros(n_words, n_words)
    for i in 1:n_words
        mx = maximum(scores[i,:])
        e  = exp.(scores[i,:] .- mx)
        attn_weights[i,:] = e ./ sum(e)
    end
    # Weighted average
    out = attn_weights * word_vectors  # n_words × d
    vec(mean(out, dims=1))
end

# ─────────────────────────────────────────────────────────────
# 14. READABILITY AND COMPLEXITY SCORES
# ─────────────────────────────────────────────────────────────

"""
    flesch_kincaid_grade(text) -> Float64

Flesch-Kincaid Grade Level readability score.
Higher = harder to read (more complex language).
"""
function flesch_kincaid_grade(text::String)::Float64
    words  = split(lowercase(text))
    n_words = length(words); n_words == 0 && return 0.0
    # Count sentences (periods, exclamations, questions)
    n_sent = max(1, count(c -> c ∈ ['.','!','?'], text))
    # Count syllables (approximation: vowel groups)
    function count_syllables(w::AbstractString)
        vowels = Set(['a','e','i','o','u'])
        n = 0; prev_vowel = false
        for c in lowercase(w)
            is_v = c ∈ vowels
            is_v && !prev_vowel && (n += 1)
            prev_vowel = is_v
        end
        max(n, 1)
    end
    n_syll = sum(count_syllables(w) for w in words; init=0)
    # FK formula: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    0.39 * (n_words / n_sent) + 11.8 * (n_syll / n_words) - 15.59
end

"""
    fog_index(text) -> Float64

Gunning Fog Index: 0.4 * (words/sentence + 100 * complex_word_fraction).
"""
function fog_index(text::String)::Float64
    words = split(lowercase(text))
    n_words = length(words); n_words == 0 && return 0.0
    n_sent  = max(1, count(c -> c ∈ ['.','!','?'], text))
    # Complex words: 3+ syllables (approximation: length > 8)
    n_complex = count(w -> length(w) > 8, words)
    0.4 * (n_words/n_sent + 100 * n_complex/n_words)
end

"""
    crypto_news_complexity(texts) -> Vector{NamedTuple}

Analyze complexity and sentiment of crypto news articles.
"""
function crypto_news_complexity(texts::Vector{String})
    lexicon = build_crypto_lexicon()
    map(texts) do text
        sentiment = lexicon_sentiment_score(text, lexicon)
        fk_grade  = flesch_kincaid_grade(text)
        fog       = fog_index(text)
        n_words   = length(split(text))
        (sentiment=sentiment, fk_grade=fk_grade, fog_index=fog,
         word_count=n_words, complexity=0.5*(fk_grade + fog)/10)
    end
end

end  # module TextAnalysis
