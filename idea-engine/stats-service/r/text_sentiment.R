# =============================================================================
# text_sentiment.R
# Text / Sentiment Analytics for Crypto Markets
# Pure base R -- no external packages
# =============================================================================
# Financial intuition: In crypto, narrative drives price -- "Bitcoin is dead"
# headlines reliably precede bounces; "institutional adoption" headlines
# precede blow-off tops. Quantifying this text signal in a rigorous pipeline
# (TF-IDF + Naive Bayes + HMM regime) transforms qualitative chatter into
# a tradeable factor that is orthogonal to price-based signals.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. TEXT PREPROCESSING
# ---------------------------------------------------------------------------

#' Lowercase, strip punctuation, tokenise
tokenise <- function(text) {
  text <- tolower(text)
  text <- gsub("[^a-z0-9 ]", " ", text)
  tokens <- strsplit(trimws(text), "\\s+")[[1]]
  tokens[nchar(tokens) > 0]
}

#' Remove stop words (minimal crypto-relevant list)
STOP_WORDS <- c("a","an","the","is","are","was","were","be","been","being",
                 "have","has","had","do","does","did","will","would","could",
                 "should","may","might","shall","can","to","of","in","for",
                 "on","with","at","by","from","as","it","its","this","that",
                 "and","or","but","not","no","we","i","you","he","she","they",
                 "our","your","their","my","his","her","its","also","just",
                 "more","so","up","out","if","about","after","before","very")

remove_stopwords <- function(tokens) {
  tokens[!tokens %in% STOP_WORDS]
}

#' Build vocabulary from list of token vectors
build_vocab <- function(token_lists, min_freq = 2L) {
  all_tokens <- unlist(token_lists)
  freq       <- table(all_tokens)
  vocab      <- names(freq[freq >= min_freq])
  sort(vocab)
}

#' Document term matrix (counts)
build_dtm <- function(token_lists, vocab) {
  n     <- length(token_lists)
  v     <- length(vocab)
  vocab_idx <- setNames(seq_len(v), vocab)
  dtm   <- matrix(0L, n, v)
  for (i in seq_len(n)) {
    for (tok in token_lists[[i]]) {
      j <- vocab_idx[[tok]]
      if (!is.null(j)) dtm[i, j] <- dtm[i, j] + 1L
    }
  }
  colnames(dtm) <- vocab
  dtm
}

# ---------------------------------------------------------------------------
# 2. TF-IDF FROM SCRATCH
# ---------------------------------------------------------------------------
# TF-IDF (term frequency – inverse document frequency): down-weights terms
# that appear in many documents, boosting rare informative terms.

#' TF: raw count / doc length
tf <- function(dtm) {
  row_sums <- rowSums(dtm)
  row_sums[row_sums == 0] <- 1
  sweep(dtm, 1, row_sums, "/")
}

#' IDF: log(N / df + 1)  (smooth variant)
idf <- function(dtm) {
  N  <- nrow(dtm)
  df <- colSums(dtm > 0)
  log((N + 1) / (df + 1)) + 1
}

#' Full TF-IDF matrix
tfidf <- function(dtm) {
  TF <- tf(dtm)
  IF <- idf(dtm)
  sweep(TF, 2, IF, "*")
}

#' Top-k terms by TF-IDF score for document i
top_terms <- function(tfidf_mat, i, k = 10L) {
  sc  <- tfidf_mat[i, ]
  idx <- order(sc, decreasing = TRUE)[1:min(k, length(sc))]
  data.frame(term = colnames(tfidf_mat)[idx], score = sc[idx])
}

# ---------------------------------------------------------------------------
# 3. CRYPTO BULLISH / BEARISH LEXICON
# ---------------------------------------------------------------------------
# Weighted lexicon: positive weights = bullish, negative = bearish.
# Tuned for crypto market discourse.

CRYPTO_LEXICON <- list(
  # Strong bullish (+2 to +3)
  "moon"          =  3.0,
  "moonshot"      =  3.0,
  "bullrun"       =  3.0,
  "breakout"      =  2.5,
  "accumulate"    =  2.0,
  "adoption"      =  2.0,
  "institutional" =  2.0,
  "halving"       =  2.0,
  "etf"           =  2.0,
  "approval"      =  2.0,
  "listing"       =  1.5,
  "partnership"   =  1.5,
  "launch"        =  1.5,
  "upgrade"       =  1.5,
  "bullish"       =  2.5,
  "pump"          =  1.5,
  "rally"         =  2.0,
  "surge"         =  2.0,
  "ath"           =  2.5,
  "hodl"          =  1.5,
  "buy"           =  1.0,
  "long"          =  1.0,
  "support"       =  1.0,
  "recover"       =  1.5,
  "bounce"        =  1.5,
  "green"         =  1.0,
  "gains"         =  1.5,
  "profit"        =  1.5,
  "optimistic"    =  1.5,
  "confidence"    =  1.0,
  "growth"        =  1.5,
  "mainstream"    =  1.5,
  "soar"          =  2.0,
  "skyrocket"     =  2.5,
  "record"        =  1.5,
  "strong"        =  1.0,
  # Mild bearish (-1)
  "sell"          = -1.0,
  "short"         = -1.0,
  "resistance"    = -1.0,
  "correction"    = -1.5,
  "pullback"      = -1.5,
  "red"           = -1.0,
  "loss"          = -1.5,
  "weak"          = -1.0,
  "drop"          = -1.5,
  "fall"          = -1.0,
  "decline"       = -1.5,
  # Strong bearish (-2 to -3)
  "crash"         = -3.0,
  "dump"          = -2.5,
  "collapse"      = -3.0,
  "hack"          = -2.5,
  "exploit"       = -2.5,
  "scam"          = -2.5,
  "fraud"         = -3.0,
  "ban"           = -2.5,
  "regulation"    = -1.5,
  "crackdown"     = -2.5,
  "insolvent"     = -3.0,
  "bankrupt"      = -3.0,
  "dead"          = -2.5,
  "bearish"       = -2.5,
  "panic"         = -2.5,
  "fear"          = -2.0,
  "liquidation"   = -2.5,
  "capitulation"  = -2.5,
  "fud"           = -2.0,
  "rug"           = -3.0,
  "ponzi"         = -3.0,
  "bubble"        = -2.0,
  "overvalued"    = -2.0,
  "warning"       = -1.5,
  "risk"          = -1.0,
  "uncertainty"   = -1.5,
  "lawsuit"       = -2.0,
  "sec"           = -1.5,
  "depegged"      = -3.0,
  "contagion"     = -2.5
)

#' Compute raw lexicon score for a token vector
lexicon_score <- function(tokens) {
  scores <- unlist(CRYPTO_LEXICON)
  words  <- names(scores)
  s <- 0.0
  for (tok in tokens) {
    idx <- match(tok, words)
    if (!is.na(idx)) s <- s + scores[idx]
  }
  s
}

# ---------------------------------------------------------------------------
# 4. NAIVE BAYES SENTIMENT CLASSIFIER
# ---------------------------------------------------------------------------
# Multinomial Naive Bayes with Laplace smoothing.
# Classes: {bullish=1, bearish=-1, neutral=0}

nb_train <- function(dtm, labels, alpha = 1.0) {
  classes <- sort(unique(labels))
  V       <- ncol(dtm)
  priors  <- log(table(labels) / length(labels))
  log_likes <- lapply(classes, function(c) {
    mask    <- labels == c
    counts  <- colSums(dtm[mask, , drop=FALSE]) + alpha
    log(counts / sum(counts))
  })
  names(log_likes) <- classes
  list(classes   = classes,
       priors    = priors,
       log_likes = log_likes,
       vocab     = colnames(dtm))
}

nb_predict <- function(model, dtm_new) {
  n  <- nrow(dtm_new)
  K  <- length(model$classes)
  log_post <- matrix(NA, n, K)
  for (k in seq_len(K)) {
    c      <- model$classes[k]
    ll     <- model$log_likes[[as.character(c)]]
    log_post[, k] <- model$priors[as.character(c)] + dtm_new %*% ll
  }
  pred_idx <- apply(log_post, 1, which.max)
  model$classes[pred_idx]
}

#' NB accuracy
nb_accuracy <- function(true, pred) mean(true == pred, na.rm = TRUE)

# ---------------------------------------------------------------------------
# 5. SYNTHETIC NEWS GENERATOR (for demo)
# ---------------------------------------------------------------------------

BULLISH_TEMPLATES <- c(
  "bitcoin breaks out to new all time high record bulls celebrate",
  "ethereum upgrade launch bullish momentum institutions accumulating",
  "btc rally strong recovery from support hodl strategy paying off",
  "crypto adoption surge mainstream institutional etf approval optimistic",
  "bitcoin soaring gains record growth confidence market sentiment",
  "altcoins moon pump bullish run accumulate opportunity green",
  "defi protocol partnership launch bullish momentum recovery bounce"
)

BEARISH_TEMPLATES <- c(
  "bitcoin crash dump massive liquidation panic capitulation bear market",
  "crypto collapse fraud hack exploit scandal scam rug pull",
  "bitcoin dead bearish fud regulation crackdown sec lawsuit",
  "exchange insolvent bankrupt contagion fear uncertainty depegged",
  "crash dump sell short resistance overvalued bubble warning panic",
  "crypto ban regulation uncertainty risk bearish outlook weakness",
  "massive liquidation cascade decline drop collapse fear uncertainty"
)

NEUTRAL_TEMPLATES <- c(
  "bitcoin trading sideways market analysis technical review update",
  "crypto price action volatile mixed signals watch resistance support",
  "market consolidation bitcoin holding level range bound activity",
  "analysis review report market data volume open interest metrics",
  "bitcoin update development progress network metrics performance",
  "crypto ecosystem activity transactions addresses growth steady",
  "blockchain technology development upgrade progress neutral impact"
)

generate_news <- function(n = 300L, seed = 42L) {
  set.seed(seed)
  labels <- sample(c("bullish","bearish","neutral"),
                   n, replace=TRUE, prob=c(0.35,0.35,0.30))
  texts <- character(n)
  for (i in seq_len(n)) {
    pool <- switch(labels[i],
                   bullish = BULLISH_TEMPLATES,
                   bearish = BEARISH_TEMPLATES,
                   neutral = NEUTRAL_TEMPLATES)
    base <- sample(pool, 1)
    # Add random noise words
    noise <- paste(sample(c("market","price","btc","crypto","update",
                             "today","news","latest","trend","data"),
                           sample(2:5,1)), collapse=" ")
    texts[i] <- paste(base, noise)
  }
  data.frame(text = texts, label = labels, stringsAsFactors = FALSE)
}

# ---------------------------------------------------------------------------
# 6. ROLLING SENTIMENT INDEX WITH EXPONENTIAL DECAY
# ---------------------------------------------------------------------------

#' Compute per-document lexicon score from text column
compute_sentiment_scores <- function(texts) {
  vapply(texts, function(t) {
    toks <- remove_stopwords(tokenise(t))
    lexicon_score(toks)
  }, numeric(1))
}

#' Exponentially weighted rolling sentiment (lambda = decay factor)
ewm_sentiment <- function(scores, lambda = 0.9) {
  n   <- length(scores)
  ewm <- numeric(n)
  ewm[1] <- scores[1]
  for (i in 2:n) {
    ewm[i] <- lambda * ewm[i-1] + (1 - lambda) * scores[i]
  }
  ewm
}

#' Normalise sentiment to [-1, 1] using rolling z-score
normalise_sentiment <- function(scores, window = 20L) {
  n   <- length(scores)
  out <- numeric(n)
  for (i in seq_len(n)) {
    lo  <- max(1L, i - window + 1L)
    xw  <- scores[lo:i]
    mu  <- mean(xw, na.rm = TRUE)
    sg  <- sd(xw, na.rm = TRUE)
    if (is.na(sg) || sg < 1e-8) out[i] <- 0 else out[i] <- (scores[i] - mu) / sg
  }
  tanh(out)   # squash to (-1,1)
}

# ---------------------------------------------------------------------------
# 7. NEWS IMPACT EVENT STUDY
# ---------------------------------------------------------------------------
# Classic event study: measure abnormal returns around high-sentiment events.

event_study <- function(sentiment_scores, asset_returns,
                         threshold = 1.5,
                         window_before = 3L,
                         window_after  = 5L) {
  n <- length(sentiment_scores)
  events_bull <- which(sentiment_scores >  threshold)
  events_bear <- which(sentiment_scores < -threshold)

  extract_window <- function(events, rets) {
    total_win <- window_before + window_after + 1L
    mat <- matrix(NA, length(events), total_win)
    for (i in seq_along(events)) {
      t0  <- events[i]
      idx <- (t0 - window_before):(t0 + window_after)
      if (any(idx < 1) || any(idx > n)) next
      mat[i, ] <- rets[idx]
    }
    mat
  }

  bull_windows <- extract_window(events_bull, asset_returns)
  bear_windows <- extract_window(events_bear, asset_returns)

  # CAR (Cumulative Average Return) over window
  car_bull <- apply(bull_windows, 2, mean, na.rm=TRUE)
  car_bear <- apply(bear_windows, 2, mean, na.rm=TRUE)

  list(
    car_bullish = car_bull,
    car_bearish = car_bear,
    n_bullish   = length(events_bull),
    n_bearish   = length(events_bear),
    cum_bull    = cumsum(car_bull),
    cum_bear    = cumsum(car_bear),
    time_axis   = (-window_before):window_after
  )
}

# ---------------------------------------------------------------------------
# 8. SENTIMENT REGIME DETECTION (3-STATE HMM)
# ---------------------------------------------------------------------------
# Hidden states: 1=bearish regime, 2=neutral, 3=bullish
# Observation: sentiment score (discretised or continuous)
# We implement Baum-Welch EM for Gaussian emission HMM.

hmm_init <- function(K = 3L, obs) {
  # Initialise from k-means-like quantiles
  qs <- quantile(obs, seq(0, 1, length.out = K + 1), na.rm=TRUE)
  mu <- sapply(seq_len(K), function(k) mean(c(qs[k], qs[k+1])))
  sg <- rep(sd(obs, na.rm=TRUE) / K, K)
  pi <- rep(1/K, K)
  A  <- matrix(0.8, K, K)
  diag(A) <- 0.8; A <- A / rowSums(A)
  list(pi = pi, A = A, mu = mu, sigma = sg, K = K)
}

hmm_gaussian_emit <- function(x, mu, sigma) {
  dnorm(x, mean = mu, sd = pmax(sigma, 1e-6))
}

#' HMM Forward pass -- returns log-likelihood and alpha matrix
hmm_forward <- function(obs, model) {
  T_  <- length(obs); K <- model$K
  B   <- matrix(0, T_, K)
  for (k in seq_len(K))
    B[, k] <- hmm_gaussian_emit(obs, model$mu[k], model$sigma[k])
  B[B < 1e-300] <- 1e-300

  alpha <- matrix(0, T_, K)
  alpha[1, ] <- model$pi * B[1, ]
  scale      <- numeric(T_); scale[1] <- sum(alpha[1,])
  alpha[1, ] <- alpha[1, ] / scale[1]

  for (t in 2:T_) {
    for (k in seq_len(K))
      alpha[t, k] <- sum(alpha[t-1, ] * model$A[, k]) * B[t, k]
    scale[t]    <- sum(alpha[t,])
    if (scale[t] < 1e-300) scale[t] <- 1e-300
    alpha[t, ]  <- alpha[t, ] / scale[t]
  }
  list(alpha = alpha, scale = scale,
       loglik = sum(log(scale)))
}

#' HMM Backward pass
hmm_backward <- function(obs, model, scale) {
  T_   <- length(obs); K <- model$K
  B    <- matrix(0, T_, K)
  for (k in seq_len(K))
    B[, k] <- hmm_gaussian_emit(obs, model$mu[k], model$sigma[k])
  B[B < 1e-300] <- 1e-300

  beta <- matrix(0, T_, K)
  beta[T_, ] <- 1
  for (t in (T_-1):1) {
    for (i in seq_len(K))
      beta[t, i] <- sum(model$A[i, ] * B[t+1, ] * beta[t+1, ])
    beta[t, ] <- beta[t, ] / scale[t+1]
  }
  beta
}

#' Baum-Welch EM for Gaussian HMM
hmm_fit <- function(obs, K = 3L, max_iter = 50L, tol = 1e-4) {
  model  <- hmm_init(K, obs)
  T_     <- length(obs)
  logliks <- numeric(max_iter)

  for (iter in seq_len(max_iter)) {
    # E-step
    fwd  <- hmm_forward(obs, model)
    bwd  <- hmm_backward(obs, model, fwd$scale)
    alpha <- fwd$alpha; beta <- bwd

    gamma <- alpha * beta
    gamma <- gamma / rowSums(gamma)

    # Xi: T-1 x K x K
    B  <- matrix(0, T_, K)
    for (k in seq_len(K))
      B[, k] <- hmm_gaussian_emit(obs, model$mu[k], model$sigma[k])
    B[B < 1e-300] <- 1e-300

    xi_sum <- matrix(0, K, K)
    for (t in 1:(T_-1)) {
      for (i in seq_len(K)) {
        for (j in seq_len(K)) {
          xi_sum[i,j] <- xi_sum[i,j] +
            alpha[t,i] * model$A[i,j] * B[t+1,j] * beta[t+1,j]
        }
      }
    }
    xi_sum <- xi_sum / sum(xi_sum)

    # M-step
    model$pi <- gamma[1, ]
    model$A  <- xi_sum / rowSums(xi_sum)
    model$A[is.nan(model$A)] <- 1/K

    for (k in seq_len(K)) {
      gk <- gamma[, k] + 1e-8
      model$mu[k]    <- sum(gk * obs) / sum(gk)
      model$sigma[k] <- sqrt(sum(gk * (obs - model$mu[k])^2) / sum(gk))
      model$sigma[k] <- max(model$sigma[k], 1e-4)
    }
    logliks[iter] <- fwd$loglik
    if (iter > 1 && abs(logliks[iter] - logliks[iter-1]) < tol) break
  }

  # Viterbi decoding
  states <- hmm_viterbi(obs, model)
  list(model = model, states = states, loglik = logliks[1:iter])
}

#' Viterbi decoding
hmm_viterbi <- function(obs, model) {
  T_ <- length(obs); K <- model$K
  B  <- matrix(0, T_, K)
  for (k in seq_len(K))
    B[, k] <- log(pmax(hmm_gaussian_emit(obs, model$mu[k], model$sigma[k]), 1e-300))

  delta <- matrix(-Inf, T_, K)
  psi   <- matrix(0L, T_, K)
  delta[1, ] <- log(pmax(model$pi, 1e-300)) + B[1, ]

  for (t in 2:T_) {
    for (j in seq_len(K)) {
      vals       <- delta[t-1, ] + log(pmax(model$A[, j], 1e-300))
      psi[t, j]  <- which.max(vals)
      delta[t,j] <- max(vals) + B[t, j]
    }
  }

  path <- integer(T_)
  path[T_] <- which.max(delta[T_, ])
  for (t in (T_-1):1) path[t] <- psi[t+1, path[t+1]]
  path
}

# ---------------------------------------------------------------------------
# 9. COMBINED PRICE + SENTIMENT SIGNAL
# ---------------------------------------------------------------------------

#' Blend momentum signal with sentiment signal
combined_signal <- function(price_returns, sentiment_norm,
                              w_price = 0.5, w_sent = 0.5) {
  # Momentum: sign of N-bar MA of returns
  n  <- length(price_returns)
  mo <- roll_mean_vec(price_returns, 10L)
  mo_norm <- tanh(mo / (sd(mo, na.rm=TRUE) + 1e-12))
  # Blend
  blend <- w_price * mo_norm + w_sent * sentiment_norm
  blend
}

roll_mean_vec <- function(x, w) {
  n <- length(x); out <- rep(NA, n)
  for (i in w:n) out[i] <- mean(x[(i-w+1):i], na.rm=TRUE)
  out
}

# ---------------------------------------------------------------------------
# 10. FEAR & GREED INDEX RECONSTRUCTION
# ---------------------------------------------------------------------------
# Components (proxies from price data):
#   1. Volatility: low vol -> greed, high vol -> fear
#   2. Momentum:   rising prices -> greed, falling -> fear
#   3. Volume:     high relative volume -> greed
#   4. Sentiment:  lexicon score
#   5. Dominance:  BTC dominance proxy (none here -- use 50)
# Weights: 0.25, 0.25, 0.10, 0.40

fear_greed_index <- function(returns, sentiment_scores,
                               vol_window = 30L, mom_window = 30L) {
  n <- length(returns)
  # 1. Vol score: invert rolling vol
  vol_roll <- rep(NA, n)
  for (i in vol_window:n)
    vol_roll[i] <- sd(returns[(i-vol_window+1):i], na.rm=TRUE)
  vol_score  <- 100 * (1 - (vol_roll - min(vol_roll, na.rm=TRUE)) /
                         (diff(range(vol_roll, na.rm=TRUE)) + 1e-12))

  # 2. Momentum score
  cum_ret <- cumprod(1 + returns)
  mom_score <- rep(50, n)
  for (i in (mom_window+1):n) {
    r_mom <- cum_ret[i] / cum_ret[i - mom_window] - 1
    mom_score[i] <- pmin(100, pmax(0, 50 + 200 * r_mom))
  }

  # 3. Volume (simulated: use abs returns as activity proxy)
  vol_act <- abs(returns)
  vol_act_roll <- rep(NA, n)
  for (i in vol_window:n)
    vol_act_roll[i] <- mean(vol_act[(i-vol_window+1):i], na.rm=TRUE)
  act_score <- 100 * (vol_act_roll - min(vol_act_roll, na.rm=TRUE)) /
    (diff(range(vol_act_roll, na.rm=TRUE)) + 1e-12)

  # 4. Sentiment score -> [0,100]
  sent_norm <- (sentiment_scores - min(sentiment_scores, na.rm=TRUE)) /
    (diff(range(sentiment_scores, na.rm=TRUE)) + 1e-12) * 100

  fg <- 0.25 * vol_score + 0.25 * mom_score + 0.10 * act_score + 0.40 * sent_norm
  fg[is.na(fg)] <- 50

  # Label
  label <- cut(fg, breaks = c(0,25,45,55,75,100),
               labels = c("Extreme Fear","Fear","Neutral","Greed","Extreme Greed"),
               include.lowest = TRUE)
  data.frame(fg_index = fg, label = as.character(label))
}

# ---------------------------------------------------------------------------
# 11. SENTIMENT REGIME TRADING SIGNAL
# ---------------------------------------------------------------------------

sentiment_regime_signal <- function(hmm_states, fear_greed, threshold = 60) {
  n <- length(hmm_states)
  # +1 if HMM in bullish state AND FG > threshold
  # -1 if HMM in bearish state AND FG < (100-threshold)
  #  0 otherwise
  signal <- integer(n)
  # Identify which HMM state has highest mean emission (bullish)
  # (assumes states are ordered 1=bear,2=neutral,3=bull by training)
  for (t in seq_len(n)) {
    if (hmm_states[t] == 3L && fear_greed[t] > threshold) {
      signal[t] <- 1L
    } else if (hmm_states[t] == 1L && fear_greed[t] < (100 - threshold)) {
      signal[t] <- -1L
    }
  }
  signal
}

# ---------------------------------------------------------------------------
# 12. SIGNAL EVALUATION
# ---------------------------------------------------------------------------

evaluate_signal <- function(signal, forward_returns, holding = 1L) {
  n    <- length(signal)
  rets <- numeric(n)
  for (t in seq_len(n - holding)) {
    if (signal[t] != 0) {
      rets[t] <- signal[t] * sum(forward_returns[(t+1):(t+holding)])
    }
  }
  eq <- cumprod(1 + rets)
  mu <- mean(rets[rets != 0], na.rm = TRUE)
  sg <- sd(rets[rets != 0], na.rm = TRUE)
  data.frame(
    sharpe    = if (!is.na(sg) && sg > 0) mu / sg * sqrt(252) else NA,
    hit_rate  = mean(rets[signal != 0] > 0, na.rm = TRUE),
    avg_ret   = mu,
    n_trades  = sum(signal != 0),
    total_ret = tail(eq, 1) - 1,
    max_dd    = min((eq - cummax(eq)) / cummax(eq), na.rm = TRUE)
  )
}

# ---------------------------------------------------------------------------
# 13. MAIN DEMO
# ---------------------------------------------------------------------------

run_sentiment_demo <- function() {
  cat("=== Text Sentiment Analytics Demo ===\n\n")
  set.seed(42)

  # Generate synthetic news corpus
  cat("Generating synthetic crypto news corpus (300 docs)...\n")
  news <- generate_news(n = 300L)
  cat(sprintf("  Docs: %d  |  Bullish: %d  Bearish: %d  Neutral: %d\n",
              nrow(news),
              sum(news$label == "bullish"),
              sum(news$label == "bearish"),
              sum(news$label == "neutral")))

  # Tokenise
  cat("\n--- 1. TF-IDF ---\n")
  token_lists <- lapply(news$text, function(t) remove_stopwords(tokenise(t)))
  vocab       <- build_vocab(token_lists, min_freq = 2L)
  cat(sprintf("  Vocabulary size: %d terms\n", length(vocab)))
  dtm  <- build_dtm(token_lists, vocab)
  tfidf_mat <- tfidf(dtm)
  top5 <- top_terms(tfidf_mat, 1L, k = 5L)
  cat("  Top-5 TF-IDF terms for doc 1:\n"); print(top5)

  # Naive Bayes
  cat("\n--- 2. Naive Bayes Classifier ---\n")
  label_int <- ifelse(news$label == "bullish", 1L,
                       ifelse(news$label == "bearish", -1L, 0L))
  train_idx <- 1:200; test_idx <- 201:300
  nb_mod    <- nb_train(dtm[train_idx,], label_int[train_idx])
  preds     <- nb_predict(nb_mod, dtm[test_idx,])
  acc       <- nb_accuracy(label_int[test_idx], preds)
  cat(sprintf("  Test accuracy: %.1f%%\n", acc * 100))

  # Lexicon scoring
  cat("\n--- 3. Lexicon Sentiment Scores ---\n")
  sent_raw  <- compute_sentiment_scores(news$text)
  cat(sprintf("  Score range: %.1f to %.1f  |  Mean: %.2f\n",
              min(sent_raw), max(sent_raw), mean(sent_raw)))
  cat(sprintf("  Bullish docs mean: %.2f  |  Bearish docs mean: %.2f\n",
              mean(sent_raw[news$label=="bullish"]),
              mean(sent_raw[news$label=="bearish"])))

  # EWM sentiment
  cat("\n--- 4. Exponentially Weighted Sentiment ---\n")
  ewm_sent  <- ewm_sentiment(sent_raw, lambda = 0.9)
  sent_norm <- normalise_sentiment(sent_raw, window = 20L)
  cat(sprintf("  EWM final value: %.3f  |  Norm range: %.2f to %.2f\n",
              tail(ewm_sent, 1), min(sent_norm, na.rm=TRUE), max(sent_norm, na.rm=TRUE)))

  # HMM regime detection
  cat("\n--- 5. HMM Sentiment Regime Detection (K=3) ---\n")
  hmm_res  <- hmm_fit(ewm_sent, K = 3L, max_iter = 30L)
  state_counts <- table(hmm_res$states)
  cat("  State frequencies:", paste(names(state_counts), state_counts, sep="=", collapse="  "), "\n")
  cat("  HMM means (ordered):", round(sort(hmm_res$model$mu), 3), "\n")

  # Simulate asset returns and event study
  cat("\n--- 6. News Impact Event Study ---\n")
  sim_rets <- rnorm(300, 0, 0.02) + 0.005 * tanh(sent_raw / 5)
  es <- event_study(sent_norm, sim_rets, threshold = 0.8)
  cat(sprintf("  Bullish events: %d  |  Bearish events: %d\n",
              es$n_bullish, es$n_bearish))
  cat("  CAR around bullish events (t=-3..+5):",
      round(es$cum_bull, 4), "\n")

  # Fear & Greed
  cat("\n--- 7. Fear & Greed Index ---\n")
  fg <- fear_greed_index(sim_rets, sent_raw)
  cat("  Label distribution:\n")
  print(table(fg$label))
  cat(sprintf("  Mean F&G: %.1f  (range %.0f-%.0f)\n",
              mean(fg$fg_index), min(fg$fg_index), max(fg$fg_index)))

  # Combined signal
  cat("\n--- 8. Combined Price + Sentiment Signal ---\n")
  sig_combined <- combined_signal(sim_rets, sent_norm, w_price=0.4, w_sent=0.6)
  sig_binary   <- ifelse(sig_combined > 0.2, 1L, ifelse(sig_combined < -0.2, -1L, 0L))
  ev <- evaluate_signal(sig_binary, sim_rets)
  cat(sprintf("  Sharpe: %.3f  |  Hit rate: %.1f%%  |  Trades: %d\n",
              ev$sharpe, ev$hit_rate * 100, ev$n_trades))

  # Regime trading signal
  cat("\n--- 9. Sentiment Regime Trading Signal ---\n")
  reg_sig <- sentiment_regime_signal(hmm_res$states, fg$fg_index)
  ev2 <- evaluate_signal(reg_sig, sim_rets)
  cat(sprintf("  Regime signal -- Sharpe: %.3f  |  Trades: %d  |  MaxDD: %.1f%%\n",
              ev2$sharpe, ev2$n_trades, ev2$max_dd * 100))

  cat("\nDone.\n")
  invisible(list(news = news, hmm = hmm_res, fg = fg,
                 event_study = es, nb = nb_mod))
}

if (interactive()) {
  sentiment_results <- run_sentiment_demo()
}

# ---------------------------------------------------------------------------
# 14. LATENT SEMANTIC ANALYSIS (LSA)
# ---------------------------------------------------------------------------
# Reduce TF-IDF matrix to low-dimensional semantic space via SVD.
# Financial intuition: LSA groups synonymous crypto terms (e.g., "moon" and
# "ath" and "rally" map to the same latent bullish concept).

lsa <- function(tfidf_matrix, n_components = 10L) {
  # Truncated SVD: keep top n_components singular values
  sv   <- svd(tfidf_matrix, nu = n_components, nv = n_components)
  k    <- min(n_components, length(sv$d))
  U    <- sv$u[, 1:k, drop=FALSE]
  S    <- diag(sv$d[1:k], k)
  Vt   <- t(sv$v[, 1:k, drop=FALSE])
  # Document embeddings in semantic space
  doc_embed  <- U %*% S
  term_embed <- t(Vt)
  list(doc_embed=doc_embed, term_embed=term_embed,
       singular_values=sv$d[1:k], Vt=Vt)
}

#' Cosine similarity between two vectors
cosine_sim <- function(a, b) {
  sum(a*b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)) + 1e-12)
}

#' Find most similar documents to query in LSA space
lsa_query <- function(query_embed, all_embed, top_k=5L) {
  sims  <- apply(all_embed, 1, function(d) cosine_sim(query_embed, d))
  idx   <- order(sims, decreasing=TRUE)[1:min(top_k, nrow(all_embed))]
  data.frame(rank=seq_along(idx), doc_idx=idx, similarity=sims[idx])
}

# ---------------------------------------------------------------------------
# 15. WORD2VEC-STYLE SKIP-GRAM (SIMPLIFIED NEGATIVE SAMPLING)
# ---------------------------------------------------------------------------
# Train word vectors; crypto-specific context window captures co-occurrence.
# We implement a highly simplified version: PMI-based word vectors.

pmi_word_vectors <- function(token_lists, vocab, window=3L, d=20L, seed=1L) {
  set.seed(seed)
  V  <- length(vocab); vi <- setNames(seq_len(V), vocab)
  # Co-occurrence matrix
  cooc <- matrix(0, V, V)
  for (doc in token_lists) {
    n_tok <- length(doc)
    for (i in seq_len(n_tok)) {
      ci <- vi[[doc[i]]]; if (is.null(ci)) next
      for (j in max(1,i-window):min(n_tok,i+window)) {
        if (j==i) next
        cj <- vi[[doc[j]]]; if (is.null(cj)) next
        cooc[ci,cj] <- cooc[ci,cj] + 1
      }
    }
  }
  # PMI = log(P(w,c)/P(w)P(c))
  total <- sum(cooc) + 1e-8
  pw  <- rowSums(cooc) / total
  pc  <- colSums(cooc) / total
  pmi <- log(pmax(cooc/total, 1e-12) / pmax(outer(pw, pc), 1e-12))
  pmi <- pmax(pmi, 0)   # Positive PMI
  # Truncated SVD for embeddings
  sv  <- svd(pmi, nu=d, nv=0)
  k   <- min(d, length(sv$d))
  embeddings <- sv$u[,1:k,drop=FALSE] %*% diag(sqrt(sv$d[1:k]),k)
  rownames(embeddings) <- vocab
  embeddings
}

# ---------------------------------------------------------------------------
# 16. TEMPORAL SENTIMENT MOMENTUM
# ---------------------------------------------------------------------------
# Is yesterday's sentiment predictive of today's sentiment?
# If so, sentiment exhibits momentum -- compound the signal.

sentiment_momentum <- function(sentiment_scores, lookback=5L) {
  n    <- length(sentiment_scores)
  mom  <- rep(NA, n)
  for (t in (lookback+1):n) {
    mom[t] <- mean(sentiment_scores[(t-lookback):(t-1)], na.rm=TRUE)
  }
  # Blend current and momentum
  blend <- 0.5 * sentiment_scores + 0.5 * mom
  blend[is.na(blend)] <- sentiment_scores[is.na(blend)]
  list(momentum=mom, blended=blend)
}

# ---------------------------------------------------------------------------
# 17. ENTITY-LEVEL SENTIMENT (PER COIN)
# ---------------------------------------------------------------------------

COIN_NAMES <- c("bitcoin","btc","ethereum","eth","solana","sol",
                 "bnb","xrp","cardano","ada","doge","shib")

extract_coin_sentiment <- function(texts, scores) {
  n    <- length(texts)
  coins_mentioned <- lapply(texts, function(t) {
    toks <- tokenise(t)
    intersect(toks, COIN_NAMES)
  })
  coin_scores <- list()
  for (coin in unique(COIN_NAMES)) {
    idx <- sapply(coins_mentioned, function(cm) coin %in% cm)
    if (sum(idx) > 0) {
      coin_scores[[coin]] <- data.frame(
        coin = coin,
        n_mentions = sum(idx),
        mean_score = mean(scores[idx], na.rm=TRUE),
        positive_frac = mean(scores[idx] > 0, na.rm=TRUE)
      )
    }
  }
  if (length(coin_scores) == 0) return(data.frame())
  do.call(rbind, coin_scores)
}

# ---------------------------------------------------------------------------
# 18. SENTIMENT SIGNAL BACKTEST
# ---------------------------------------------------------------------------

backtest_sentiment_signal <- function(signal, forward_returns, tc=0.001) {
  n   <- length(signal)
  pos <- sign(signal)
  rets <- numeric(n)
  for (t in 2:n) {
    cost    <- abs(pos[t] - pos[t-1]) * tc
    rets[t] <- pos[t-1] * forward_returns[t] - cost
  }
  eq <- cumprod(1 + rets)
  list(returns=rets, equity=eq,
       sharpe=sharpe_ratio(rets[rets!=0]),
       max_dd=min((eq-cummax(eq))/cummax(eq), na.rm=TRUE))
}

# ---------------------------------------------------------------------------
# 19. SENTIMENT EXTENDED DEMO
# ---------------------------------------------------------------------------

run_sentiment_extended_demo <- function() {
  cat("=== Sentiment Extended Demo ===\n\n")
  set.seed(42)

  news   <- generate_news(n=200L)
  tokens <- lapply(news$text, function(t) remove_stopwords(tokenise(t)))
  vocab  <- build_vocab(tokens, min_freq=2L)
  dtm    <- build_dtm(tokens, vocab)

  cat("--- 1. LSA Semantic Space ---\n")
  tfidf_m <- tfidf(dtm)
  lsa_res  <- lsa(tfidf_m, n_components=5L)
  cat(sprintf("  Doc embedding shape: %dx%d\n",
              nrow(lsa_res$doc_embed), ncol(lsa_res$doc_embed)))
  # Find similar docs to doc 1
  similar <- lsa_query(lsa_res$doc_embed[1,], lsa_res$doc_embed, top_k=3L)
  cat("  Top-3 docs similar to doc 1:\n"); print(similar)

  cat("\n--- 2. PMI Word Vectors ---\n")
  if (length(vocab) >= 5L) {
    wv <- pmi_word_vectors(tokens, vocab[1:min(50L,length(vocab))], d=5L)
    cat(sprintf("  Word vectors shape: %dx%d\n", nrow(wv), ncol(wv)))
  }

  cat("\n--- 3. Sentiment Momentum ---\n")
  scores <- compute_sentiment_scores(news$text)
  sm     <- sentiment_momentum(scores, lookback=5L)
  cat(sprintf("  Correlation: raw vs momentum signal: %.4f\n",
              cor(scores[6:length(scores)], sm$momentum[6:length(sm$momentum)],
                  use="complete.obs")))

  cat("\n--- 4. Coin-Level Sentiment ---\n")
  coin_sent <- extract_coin_sentiment(news$text, scores)
  if (nrow(coin_sent) > 0) {
    print(head(coin_sent[order(-coin_sent$n_mentions),], 5))
  }

  cat("\n--- 5. Sentiment Signal Backtest ---\n")
  sim_rets <- rnorm(200, 0, 0.02) + 0.005 * tanh(scores / 5)
  bt <- backtest_sentiment_signal(tanh(scores/3), sim_rets, tc=0.001)
  cat(sprintf("  Sharpe=%.3f  MaxDD=%.1f%%\n", bt$sharpe, bt$max_dd*100))

  invisible(list(lsa=lsa_res, sm=sm, coin_sent=coin_sent, bt=bt))
}

if (interactive()) {
  sent_ext <- run_sentiment_extended_demo()
}
