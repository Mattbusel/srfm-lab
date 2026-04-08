##############################################################################
# network_models.R -- Financial Network Analysis
# Adjacency construction, centrality, community detection, contagion,
# bipartite, Granger/transfer-entropy networks, visualization, dynamics
##############################################################################

# ---------------------------------------------------------------------------
# Adjacency Matrix from Correlation (threshold, MST, PMFG)
# ---------------------------------------------------------------------------
correlation_adjacency <- function(returns, method = c("threshold", "mst", "pmfg"),
                                   threshold = 0.5) {
  method <- match.arg(method)
  C <- cor(returns, use = "pairwise.complete.obs")
  n <- ncol(C)
  D <- sqrt(2 * (1 - C))
  switch(method,
    threshold = {
      A <- (abs(C) >= threshold) * 1
      diag(A) <- 0
      list(adjacency = A, correlation = C, distance = D, method = "threshold")
    },
    mst = {
      mst <- minimum_spanning_tree(D)
      list(adjacency = mst$adjacency, correlation = C, distance = D,
           mst_edges = mst$edges, total_weight = mst$total_weight, method = "MST")
    },
    pmfg = {
      pmfg <- planar_maximally_filtered_graph(C, D)
      list(adjacency = pmfg$adjacency, correlation = C, distance = D,
           edges = pmfg$edges, method = "PMFG")
    }
  )
}

# ---------------------------------------------------------------------------
# Minimum Spanning Tree (Prim's algorithm)
# ---------------------------------------------------------------------------
minimum_spanning_tree <- function(D) {
  n <- nrow(D)
  diag(D) <- Inf
  in_tree <- rep(FALSE, n)
  in_tree[1] <- TRUE
  A <- matrix(0, n, n)
  edges <- list()
  total_weight <- 0
  for (step in 1:(n - 1)) {
    best_i <- 0; best_j <- 0; best_w <- Inf
    tree_nodes <- which(in_tree)
    non_tree <- which(!in_tree)
    for (i in tree_nodes) {
      for (j in non_tree) {
        if (D[i, j] < best_w) {
          best_w <- D[i, j]; best_i <- i; best_j <- j
        }
      }
    }
    A[best_i, best_j] <- 1; A[best_j, best_i] <- 1
    in_tree[best_j] <- TRUE
    edges[[step]] <- c(best_i, best_j, best_w)
    total_weight <- total_weight + best_w
  }
  if (!is.null(colnames(D))) { rownames(A) <- colnames(D); colnames(A) <- colnames(D) }
  list(adjacency = A, edges = do.call(rbind, edges), total_weight = total_weight)
}

# ---------------------------------------------------------------------------
# Planar Maximally Filtered Graph (greedy approximation)
# ---------------------------------------------------------------------------
planar_maximally_filtered_graph <- function(C, D) {
  n <- nrow(C)
  max_edges <- 3 * (n - 2)
  pairs <- expand.grid(i = 1:n, j = 1:n)
  pairs <- pairs[pairs$i < pairs$j, ]
  pairs$corr <- mapply(function(i, j) C[i, j], pairs$i, pairs$j)
  pairs <- pairs[order(-abs(pairs$corr)), ]
  A <- matrix(0, n, n)
  edges <- list()
  n_edges <- 0
  degree <- rep(0, n)
  for (row in 1:nrow(pairs)) {
    if (n_edges >= max_edges) break
    i <- pairs$i[row]; j <- pairs$j[row]
    A[i, j] <- 1; A[j, i] <- 1
    degree[i] <- degree[i] + 1; degree[j] <- degree[j] + 1
    if (!is_planar_check(A, n_edges + 1, n)) {
      A[i, j] <- 0; A[j, i] <- 0
      degree[i] <- degree[i] - 1; degree[j] <- degree[j] - 1
      next
    }
    n_edges <- n_edges + 1
    edges[[n_edges]] <- c(i, j, pairs$corr[row])
  }
  if (!is.null(colnames(C))) { rownames(A) <- colnames(C); colnames(A) <- colnames(C) }
  list(adjacency = A, edges = do.call(rbind, edges), n_edges = n_edges)
}

is_planar_check <- function(A, n_edges, n_vertices) {
  n_edges <= 3 * n_vertices - 6
}

# ---------------------------------------------------------------------------
# Degree Centrality
# ---------------------------------------------------------------------------
degree_centrality <- function(A) {
  n <- nrow(A)
  deg <- rowSums(A != 0)
  norm_deg <- deg / (n - 1)
  list(degree = deg, normalized = norm_deg)
}

# ---------------------------------------------------------------------------
# Betweenness Centrality
# ---------------------------------------------------------------------------
betweenness_centrality <- function(A) {
  n <- nrow(A)
  bc <- rep(0, n)
  for (s in 1:n) {
    sp <- bfs_shortest_paths(A, s)
    sigma <- sp$sigma
    dist <- sp$dist
    delta <- rep(0, n)
    vertices_by_dist <- order(-dist)
    for (w in vertices_by_dist) {
      if (dist[w] == Inf || dist[w] == 0) next
      preds <- sp$predecessors[[w]]
      for (v in preds) {
        frac <- sigma[v] / sigma[w]
        delta[v] <- delta[v] + frac * (1 + delta[w])
      }
      if (w != s) bc[w] <- bc[w] + delta[w]
    }
  }
  bc <- bc / 2
  norm_bc <- bc / ((n - 1) * (n - 2) / 2)
  if (!is.null(rownames(A))) names(bc) <- rownames(A)
  list(betweenness = bc, normalized = norm_bc)
}

bfs_shortest_paths <- function(A, source) {
  n <- nrow(A)
  dist <- rep(Inf, n)
  sigma <- rep(0, n)
  predecessors <- vector("list", n)
  dist[source] <- 0; sigma[source] <- 1
  queue <- source
  while (length(queue) > 0) {
    v <- queue[1]; queue <- queue[-1]
    neighbors <- which(A[v, ] != 0)
    for (w in neighbors) {
      if (dist[w] == Inf) {
        dist[w] <- dist[v] + 1
        queue <- c(queue, w)
      }
      if (dist[w] == dist[v] + 1) {
        sigma[w] <- sigma[w] + sigma[v]
        predecessors[[w]] <- c(predecessors[[w]], v)
      }
    }
  }
  list(dist = dist, sigma = sigma, predecessors = predecessors)
}

# ---------------------------------------------------------------------------
# Closeness Centrality
# ---------------------------------------------------------------------------
closeness_centrality <- function(A) {
  n <- nrow(A)
  cc <- numeric(n)
  for (i in 1:n) {
    sp <- bfs_shortest_paths(A, i)
    reachable <- sp$dist[sp$dist < Inf & sp$dist > 0]
    if (length(reachable) > 0) {
      cc[i] <- length(reachable) / sum(reachable)
    }
  }
  if (!is.null(rownames(A))) names(cc) <- rownames(A)
  list(closeness = cc)
}

# ---------------------------------------------------------------------------
# Eigenvector Centrality
# ---------------------------------------------------------------------------
eigenvector_centrality <- function(A, tol = 1e-10, max_iter = 1000) {
  n <- nrow(A)
  x <- rep(1, n)
  for (iter in 1:max_iter) {
    x_new <- A %*% x
    norm_x <- sqrt(sum(x_new^2))
    if (norm_x < 1e-15) break
    x_new <- x_new / norm_x
    if (max(abs(x_new - x)) < tol) break
    x <- as.vector(x_new)
  }
  x <- as.vector(x_new)
  x <- x / max(abs(x))
  if (!is.null(rownames(A))) names(x) <- rownames(A)
  list(eigenvector = x, eigenvalue = norm_x, iterations = iter)
}

# ---------------------------------------------------------------------------
# PageRank
# ---------------------------------------------------------------------------
pagerank <- function(A, damping = 0.85, tol = 1e-10, max_iter = 1000) {
  n <- nrow(A)
  out_degree <- rowSums(A != 0)
  out_degree[out_degree == 0] <- 1
  M <- A / out_degree
  pr <- rep(1 / n, n)
  for (iter in 1:max_iter) {
    pr_new <- (1 - damping) / n + damping * as.vector(t(M) %*% pr)
    pr_new <- pr_new / sum(pr_new)
    if (max(abs(pr_new - pr)) < tol) break
    pr <- pr_new
  }
  if (!is.null(rownames(A))) names(pr_new) <- rownames(A)
  list(pagerank = pr_new, iterations = iter)
}

# ---------------------------------------------------------------------------
# All Centralities
# ---------------------------------------------------------------------------
all_centralities <- function(A) {
  deg <- degree_centrality(A)
  btw <- betweenness_centrality(A)
  cls <- closeness_centrality(A)
  eig <- eigenvector_centrality(A)
  pr <- pagerank(A)
  n <- nrow(A)
  result <- data.frame(
    degree = deg$degree, degree_norm = deg$normalized,
    betweenness = btw$betweenness, betweenness_norm = btw$normalized,
    closeness = cls$closeness,
    eigenvector = eig$eigenvector,
    pagerank = pr$pagerank
  )
  if (!is.null(rownames(A))) rownames(result) <- rownames(A)
  result
}

# ---------------------------------------------------------------------------
# Community Detection: Louvain
# ---------------------------------------------------------------------------
louvain_communities <- function(A) {
  n <- nrow(A)
  m <- sum(A) / 2
  if (m == 0) return(list(membership = 1:n, modularity = 0))
  k <- rowSums(A)
  community <- 1:n
  improved <- TRUE
  while (improved) {
    improved <- FALSE
    for (i in sample(1:n)) {
      old_c <- community[i]
      neighbor_comms <- unique(community[which(A[i, ] != 0)])
      if (length(neighbor_comms) == 0) next
      best_c <- old_c; best_dq <- 0
      ki <- k[i]
      for (c_new in neighbor_comms) {
        if (c_new == old_c) next
        members_new <- which(community == c_new)
        members_old <- which(community == old_c & (1:n) != i)
        sum_in_new <- sum(A[i, members_new])
        sum_tot_new <- sum(k[members_new])
        sum_in_old <- sum(A[i, members_old])
        sum_tot_old <- sum(k[members_old])
        dq <- (sum_in_new - sum_tot_new * ki / (2 * m)) / (2 * m) -
          (sum_in_old - sum_tot_old * ki / (2 * m)) / (2 * m)
        if (dq > best_dq) {
          best_dq <- dq; best_c <- c_new
        }
      }
      if (best_c != old_c) {
        community[i] <- best_c
        improved <- TRUE
      }
    }
  }
  uc <- unique(community)
  membership <- match(community, uc)
  mod <- compute_modularity(A, membership)
  list(membership = membership, n_communities = length(uc),
       modularity = mod, method = "Louvain")
}

compute_modularity <- function(A, membership) {
  m <- sum(A) / 2
  if (m == 0) return(0)
  k <- rowSums(A)
  n <- nrow(A)
  Q <- 0
  for (i in 1:n) {
    for (j in 1:n) {
      if (membership[i] == membership[j]) {
        Q <- Q + A[i, j] - k[i] * k[j] / (2 * m)
      }
    }
  }
  Q / (2 * m)
}

# ---------------------------------------------------------------------------
# Community Detection: Spectral Clustering
# ---------------------------------------------------------------------------
spectral_clustering <- function(A, k = 2) {
  n <- nrow(A)
  D <- diag(rowSums(A))
  D_inv_sqrt <- diag(1 / sqrt(pmax(diag(D), 1e-10)))
  L_norm <- diag(n) - D_inv_sqrt %*% A %*% D_inv_sqrt
  eig <- eigen(L_norm, symmetric = TRUE)
  idx <- order(eig$values)
  U <- eig$vectors[, idx[1:k], drop = FALSE]
  U_norm <- U / sqrt(rowSums(U^2) + 1e-10)
  km <- kmeans_impl(U_norm, k, max_iter = 100)
  mod <- compute_modularity(A, km$cluster)
  list(membership = km$cluster, n_communities = k, modularity = mod,
       eigenvalues = eig$values[idx[1:(k + 1)]], method = "Spectral")
}

kmeans_impl <- function(X, k, max_iter = 100) {
  n <- nrow(X); p <- ncol(X)
  centers <- X[sample(n, k), , drop = FALSE]
  cluster <- integer(n)
  for (iter in 1:max_iter) {
    dists <- matrix(0, n, k)
    for (j in 1:k) {
      dists[, j] <- rowSums((X - matrix(centers[j, ], n, p, byrow = TRUE))^2)
    }
    new_cluster <- apply(dists, 1, which.min)
    if (identical(new_cluster, cluster)) break
    cluster <- new_cluster
    for (j in 1:k) {
      members <- which(cluster == j)
      if (length(members) > 0) centers[j, ] <- colMeans(X[members, , drop = FALSE])
    }
  }
  list(cluster = cluster, centers = centers, iterations = iter)
}

# ---------------------------------------------------------------------------
# Community Detection: Label Propagation
# ---------------------------------------------------------------------------
label_propagation <- function(A, max_iter = 100) {
  n <- nrow(A)
  labels <- 1:n
  for (iter in 1:max_iter) {
    order_i <- sample(1:n)
    changed <- FALSE
    for (i in order_i) {
      neighbors <- which(A[i, ] != 0)
      if (length(neighbors) == 0) next
      neighbor_labels <- labels[neighbors]
      weights <- A[i, neighbors]
      tab <- tapply(weights, neighbor_labels, sum)
      best_label <- as.integer(names(which.max(tab)))
      if (best_label != labels[i]) {
        labels[i] <- best_label
        changed <- TRUE
      }
    }
    if (!changed) break
  }
  uc <- unique(labels)
  membership <- match(labels, uc)
  mod <- compute_modularity(A, membership)
  list(membership = membership, n_communities = length(uc),
       modularity = mod, iterations = iter, method = "LabelProp")
}

# ---------------------------------------------------------------------------
# Network Topology Metrics
# ---------------------------------------------------------------------------
network_topology <- function(A) {
  n <- nrow(A)
  deg <- rowSums(A != 0)
  density <- sum(A != 0) / (n * (n - 1))
  # Clustering coefficient
  cc <- numeric(n)
  for (i in 1:n) {
    neighbors <- which(A[i, ] != 0)
    ki <- length(neighbors)
    if (ki < 2) { cc[i] <- 0; next }
    links <- 0
    for (a in 1:(ki - 1)) {
      for (b in (a + 1):ki) {
        if (A[neighbors[a], neighbors[b]] != 0) links <- links + 1
      }
    }
    cc[i] <- 2 * links / (ki * (ki - 1))
  }
  avg_cc <- mean(cc)
  # Average path length
  total_dist <- 0; count <- 0
  for (i in 1:n) {
    sp <- bfs_shortest_paths(A, i)
    finite <- sp$dist[sp$dist < Inf & sp$dist > 0]
    total_dist <- total_dist + sum(finite)
    count <- count + length(finite)
  }
  avg_path <- if (count > 0) total_dist / count else Inf
  # Small-world measure
  random_cc <- mean(deg) / n
  random_path <- if (mean(deg) > 1) log(n) / log(mean(deg)) else Inf
  sigma_sw <- (avg_cc / max(random_cc, 1e-10)) / (avg_path / max(random_path, 1e-10))
  # Assortativity
  edges_from <- c(); edges_to <- c()
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      if (A[i, j] != 0) { edges_from <- c(edges_from, i); edges_to <- c(edges_to, j) }
    }
  }
  if (length(edges_from) > 1) {
    deg_from <- deg[edges_from]; deg_to <- deg[edges_to]
    assortativity <- cor(deg_from, deg_to)
  } else {
    assortativity <- NA
  }
  list(n = n, density = density, avg_degree = mean(deg),
       clustering_coefficient = avg_cc, avg_path_length = avg_path,
       small_world_sigma = sigma_sw, assortativity = assortativity,
       degree_distribution = table(deg))
}

# ---------------------------------------------------------------------------
# Dynamic Networks: Rolling Correlation Network
# ---------------------------------------------------------------------------
rolling_network <- function(returns, window = 60, step = 20,
                             threshold = 0.5) {
  n_obs <- nrow(returns)
  n_assets <- ncol(returns)
  starts <- seq(1, n_obs - window + 1, by = step)
  snapshots <- list()
  topology_ts <- list()
  for (s in seq_along(starts)) {
    idx <- starts[s]:(starts[s] + window - 1)
    R_win <- returns[idx, ]
    net <- correlation_adjacency(R_win, method = "threshold", threshold = threshold)
    topo <- network_topology(net$adjacency)
    snapshots[[s]] <- list(start = starts[s], end = starts[s] + window - 1,
                           adjacency = net$adjacency)
    topology_ts[[s]] <- c(start = starts[s], density = topo$density,
                           avg_cc = topo$clustering_coefficient,
                           avg_path = topo$avg_path_length,
                           assortativity = topo$assortativity)
  }
  topo_df <- as.data.frame(do.call(rbind, topology_ts))
  list(snapshots = snapshots, topology = topo_df, n_snapshots = length(starts),
       method = "Rolling-Network")
}

# ---------------------------------------------------------------------------
# Contagion: SIR on Financial Network
# ---------------------------------------------------------------------------
sir_contagion <- function(A, initial_infected, beta = 0.3, gamma = 0.1,
                           max_steps = 100) {
  n <- nrow(A)
  state <- rep(0, n)  # 0=S, 1=I, 2=R
  state[initial_infected] <- 1
  history <- matrix(0, max_steps + 1, 3)
  colnames(history) <- c("S", "I", "R")
  history[1, ] <- c(sum(state == 0), sum(state == 1), sum(state == 2))
  for (t in 1:max_steps) {
    new_state <- state
    infected <- which(state == 1)
    if (length(infected) == 0) {
      history[(t + 1):nrow(history), ] <- matrix(history[t, ],
        nrow = nrow(history) - t, ncol = 3, byrow = TRUE)
      break
    }
    for (i in infected) {
      neighbors <- which(A[i, ] != 0 & state == 0)
      for (j in neighbors) {
        if (runif(1) < beta * abs(A[i, j])) new_state[j] <- 1
      }
      if (runif(1) < gamma) new_state[i] <- 2
    }
    state <- new_state
    history[t + 1, ] <- c(sum(state == 0), sum(state == 1), sum(state == 2))
  }
  list(final_state = state, history = history[1:(t + 1), ],
       total_infected = sum(state >= 1), peak_infected = max(history[, "I"]),
       method = "SIR")
}

# ---------------------------------------------------------------------------
# Contagion: DebtRank
# ---------------------------------------------------------------------------
debtrank <- function(A_liability, equity, initial_shocked, shock_fraction = 1) {
  n <- nrow(A_liability)
  assets <- equity + colSums(A_liability)
  h <- rep(0, n)
  h[initial_shocked] <- shock_fraction
  state <- rep(0, n)  # 0=undistressed, 1=distressed, 2=inactive
  state[initial_shocked] <- 1
  total_assets <- sum(assets)
  dr <- sum(h * assets) / total_assets
  for (iter in 1:n) {
    h_new <- h
    new_distressed <- which(state == 1)
    if (length(new_distressed) == 0) break
    for (i in new_distressed) {
      creditors <- which(A_liability[, i] > 0)
      for (j in creditors) {
        if (state[j] == 2) next
        loss <- A_liability[j, i] * h[i]
        delta_h <- min(loss / equity[j], 1 - h_new[j])
        h_new[j] <- h_new[j] + delta_h
        if (h_new[j] > 0 && state[j] == 0) state[j] <- 1
      }
    }
    state[new_distressed] <- 2
    h <- h_new
    if (all(state != 1)) break
  }
  dr_final <- sum(h * assets) / total_assets
  list(debtrank = dr_final, h = h, state = state,
       systemic_loss = sum(h * assets), method = "DebtRank")
}

# ---------------------------------------------------------------------------
# Contagion: Fire-Sale Cascade
# ---------------------------------------------------------------------------
fire_sale_cascade <- function(holdings, equity, leverage_limit = 10,
                               price_impact = 0.01, max_rounds = 50) {
  # holdings: n_banks x n_assets matrix
  n_banks <- nrow(holdings); n_assets <- ncol(holdings)
  prices <- rep(1, n_assets)
  portfolio_value <- holdings %*% prices
  total_assets <- portfolio_value
  debt <- total_assets - equity
  bank_active <- rep(TRUE, n_banks)
  price_history <- matrix(1, max_rounds + 1, n_assets)
  defaults <- integer(0)
  for (round in 1:max_rounds) {
    portfolio_value <- holdings %*% prices
    equity_current <- portfolio_value - debt
    defaulted_now <- which(equity_current <= 0 & bank_active)
    if (length(defaulted_now) > 0) {
      defaults <- c(defaults, defaulted_now)
      bank_active[defaulted_now] <- FALSE
    }
    leverage <- portfolio_value / pmax(equity_current, 1e-10)
    forced_sellers <- which(leverage > leverage_limit & bank_active)
    if (length(forced_sellers) == 0 && length(defaulted_now) == 0) break
    sell_pressure <- rep(0, n_assets)
    for (b in c(forced_sellers, defaulted_now)) {
      if (b %in% defaulted_now) {
        sell_pressure <- sell_pressure + holdings[b, ]
        holdings[b, ] <- 0
      } else {
        target_sell <- portfolio_value[b] - equity_current[b] * leverage_limit
        sell_frac <- min(target_sell / portfolio_value[b], 1)
        sell_amount <- holdings[b, ] * sell_frac
        sell_pressure <- sell_pressure + sell_amount
        holdings[b, ] <- holdings[b, ] - sell_amount
      }
    }
    for (a in 1:n_assets) {
      prices[a] <- prices[a] * exp(-price_impact * sell_pressure[a])
    }
    price_history[round + 1, ] <- prices
  }
  list(final_prices = prices, defaults = defaults, n_defaults = length(defaults),
       price_history = price_history[1:(round + 1), ],
       rounds = round, method = "Fire-Sale-Cascade")
}

# ---------------------------------------------------------------------------
# Bipartite Network: Fund-Stock Holdings
# ---------------------------------------------------------------------------
bipartite_network <- function(holdings_matrix) {
  n_funds <- nrow(holdings_matrix)
  n_stocks <- ncol(holdings_matrix)
  B <- (holdings_matrix > 0) * 1
  fund_proj <- B %*% t(B)
  diag(fund_proj) <- 0
  stock_proj <- t(B) %*% B
  diag(stock_proj) <- 0
  fund_overlap <- fund_proj
  for (i in 1:n_funds) {
    for (j in 1:n_funds) {
      denom <- sqrt(sum(B[i, ]) * sum(B[j, ]))
      if (denom > 0) fund_overlap[i, j] <- fund_proj[i, j] / denom
    }
  }
  list(bipartite = B, fund_projection = fund_proj,
       stock_projection = stock_proj, fund_overlap = fund_overlap,
       n_funds = n_funds, n_stocks = n_stocks,
       fund_diversification = rowSums(B),
       stock_crowding = colSums(B), method = "Bipartite")
}

# ---------------------------------------------------------------------------
# Systemic Importance Score
# ---------------------------------------------------------------------------
systemic_importance <- function(A, equity = NULL, method = c("composite", "debtrank")) {
  method <- match.arg(method)
  n <- nrow(A)
  cent <- all_centralities(A)
  if (method == "composite") {
    scores <- 0.3 * cent$degree_norm + 0.3 * cent$betweenness_norm +
      0.2 * cent$eigenvector + 0.2 * cent$pagerank
    if (!is.null(equity)) {
      size_score <- equity / sum(equity)
      scores <- 0.6 * scores + 0.4 * size_score
    }
  } else {
    if (is.null(equity)) equity <- rep(1, n)
    scores <- numeric(n)
    for (i in 1:n) {
      dr <- debtrank(A, equity, i, 1)
      scores[i] <- dr$debtrank
    }
  }
  rank_order <- order(-scores)
  list(scores = scores, ranking = rank_order,
       centralities = cent, method = "Systemic-Importance")
}

# ---------------------------------------------------------------------------
# Granger Causality Network
# ---------------------------------------------------------------------------
granger_network <- function(returns, lags = 2, significance = 0.05) {
  n <- ncol(returns)
  TT <- nrow(returns)
  A <- matrix(0, n, n)
  pval_mat <- matrix(1, n, n)
  for (i in 1:n) {
    for (j in 1:n) {
      if (i == j) next
      yi <- returns[(lags + 1):TT, i]
      regs_r <- matrix(0, length(yi), lags)
      regs_u <- matrix(0, length(yi), 2 * lags)
      for (l in 1:lags) {
        regs_r[, l] <- returns[(lags + 1 - l):(TT - l), i]
        regs_u[, l] <- returns[(lags + 1 - l):(TT - l), i]
        regs_u[, lags + l] <- returns[(lags + 1 - l):(TT - l), j]
      }
      fit_r <- lm.fit(cbind(1, regs_r), yi)
      fit_u <- lm.fit(cbind(1, regs_u), yi)
      ssr_r <- sum(fit_r$residuals^2)
      ssr_u <- sum(fit_u$residuals^2)
      n_eff <- length(yi)
      F_stat <- ((ssr_r - ssr_u) / lags) / (ssr_u / (n_eff - 2 * lags - 1))
      pval <- 1 - pf(F_stat, lags, n_eff - 2 * lags - 1)
      pval_mat[i, j] <- pval
      if (pval < significance) A[i, j] <- F_stat
    }
  }
  if (!is.null(colnames(returns))) {
    rownames(A) <- colnames(A) <- colnames(returns)
    rownames(pval_mat) <- colnames(pval_mat) <- colnames(returns)
  }
  list(adjacency = A, pvalues = pval_mat, n_edges = sum(A != 0),
       density = sum(A != 0) / (n * (n - 1)), method = "Granger-Network")
}

# ---------------------------------------------------------------------------
# Transfer Entropy Network
# ---------------------------------------------------------------------------
transfer_entropy_network <- function(returns, lags = 1, bins = 5,
                                      significance = 0.05, n_boot = 200) {
  n <- ncol(returns)
  TT <- nrow(returns)
  discretize <- function(x, nb) {
    breaks <- quantile(x, probs = seq(0, 1, length.out = nb + 1), na.rm = TRUE)
    breaks[1] <- breaks[1] - 1
    as.integer(cut(x, breaks))
  }
  ret_disc <- apply(returns, 2, discretize, nb = bins)
  entropy <- function(x) {
    p <- table(x) / length(x)
    -sum(p * log(p + 1e-15))
  }
  cond_entropy <- function(x, y) {
    joint <- paste(x, y, sep = "_")
    p_joint <- table(joint) / length(joint)
    p_y <- table(y) / length(y)
    H_joint <- -sum(p_joint * log(p_joint + 1e-15))
    H_y <- -sum(p_y * log(p_y + 1e-15))
    H_joint - H_y
  }
  te_mat <- matrix(0, n, n)
  pval_mat <- matrix(1, n, n)
  for (i in 1:n) {
    for (j in 1:n) {
      if (i == j) next
      x_fut <- ret_disc[(lags + 1):TT, i]
      x_past <- ret_disc[1:(TT - lags), i]
      y_past <- ret_disc[1:(TT - lags), j]
      xy_past <- paste(x_past, y_past, sep = "_")
      te <- cond_entropy(x_fut, x_past) - cond_entropy(x_fut, xy_past)
      te_mat[i, j] <- max(te, 0)
      boot_te <- numeric(n_boot)
      for (b in 1:n_boot) {
        y_shuf <- sample(y_past)
        xy_shuf <- paste(x_past, y_shuf, sep = "_")
        boot_te[b] <- cond_entropy(x_fut, x_past) - cond_entropy(x_fut, xy_shuf)
      }
      pval_mat[i, j] <- mean(boot_te >= te)
    }
  }
  A <- te_mat * (pval_mat < significance)
  if (!is.null(colnames(returns))) {
    rownames(A) <- colnames(A) <- colnames(returns)
  }
  list(adjacency = A, te_matrix = te_mat, pvalues = pval_mat,
       n_edges = sum(A != 0), method = "TE-Network")
}

# ---------------------------------------------------------------------------
# Network Visualization: Force-Directed Layout (Fruchterman-Reingold)
# ---------------------------------------------------------------------------
force_directed_layout <- function(A, iterations = 500, temp_init = 1,
                                   cool_rate = 0.95) {
  n <- nrow(A)
  pos <- matrix(runif(n * 2, -1, 1), n, 2)
  area <- 4; k_const <- sqrt(area / n)
  temp <- temp_init
  for (iter in 1:iterations) {
    disp <- matrix(0, n, 2)
    # Repulsive forces
    for (i in 1:n) {
      for (j in 1:n) {
        if (i == j) next
        delta <- pos[i, ] - pos[j, ]
        dist_ij <- max(sqrt(sum(delta^2)), 0.01)
        force_r <- k_const^2 / dist_ij
        disp[i, ] <- disp[i, ] + (delta / dist_ij) * force_r
      }
    }
    # Attractive forces
    for (i in 1:(n - 1)) {
      for (j in (i + 1):n) {
        if (A[i, j] == 0) next
        delta <- pos[i, ] - pos[j, ]
        dist_ij <- max(sqrt(sum(delta^2)), 0.01)
        force_a <- dist_ij^2 / k_const * abs(A[i, j])
        disp[i, ] <- disp[i, ] - (delta / dist_ij) * force_a
        disp[j, ] <- disp[j, ] + (delta / dist_ij) * force_a
      }
    }
    for (i in 1:n) {
      disp_len <- max(sqrt(sum(disp[i, ]^2)), 0.01)
      pos[i, ] <- pos[i, ] + (disp[i, ] / disp_len) * min(disp_len, temp)
    }
    temp <- temp * cool_rate
  }
  pos[, 1] <- (pos[, 1] - min(pos[, 1])) / max(diff(range(pos[, 1])), 1e-10)
  pos[, 2] <- (pos[, 2] - min(pos[, 2])) / max(diff(range(pos[, 2])), 1e-10)
  colnames(pos) <- c("x", "y")
  if (!is.null(rownames(A))) rownames(pos) <- rownames(A)
  list(layout = pos, method = "Force-Directed")
}

# ---------------------------------------------------------------------------
# Network Visualization: Circular Layout
# ---------------------------------------------------------------------------
circular_layout <- function(A) {
  n <- nrow(A)
  angles <- seq(0, 2 * pi, length.out = n + 1)[1:n]
  pos <- cbind(cos(angles), sin(angles))
  colnames(pos) <- c("x", "y")
  if (!is.null(rownames(A))) rownames(pos) <- rownames(A)
  list(layout = pos, method = "Circular")
}

# ---------------------------------------------------------------------------
# Network Visualization: Hierarchical Layout
# ---------------------------------------------------------------------------
hierarchical_layout <- function(A, root = NULL) {
  n <- nrow(A)
  if (is.null(root)) root <- which.max(rowSums(A != 0))
  visited <- rep(FALSE, n)
  level <- rep(0, n)
  queue <- root
  visited[root] <- TRUE
  while (length(queue) > 0) {
    v <- queue[1]; queue <- queue[-1]
    neighbors <- which(A[v, ] != 0 & !visited)
    for (w in neighbors) {
      level[w] <- level[v] + 1
      visited[w] <- TRUE
      queue <- c(queue, w)
    }
  }
  if (any(!visited)) level[!visited] <- max(level) + 1
  max_level <- max(level)
  pos <- matrix(0, n, 2)
  pos[, 2] <- 1 - level / max(max_level, 1)
  for (l in 0:max_level) {
    nodes_at_l <- which(level == l)
    nl <- length(nodes_at_l)
    if (nl > 0) pos[nodes_at_l, 1] <- seq(0, 1, length.out = nl + 2)[2:(nl + 1)]
  }
  colnames(pos) <- c("x", "y")
  list(layout = pos, levels = level, method = "Hierarchical")
}

# ---------------------------------------------------------------------------
# Network Statistics Over Time
# ---------------------------------------------------------------------------
network_evolution <- function(returns, window = 60, step = 10,
                               threshold = 0.5) {
  rolling <- rolling_network(returns, window, step, threshold)
  topo <- rolling$topology
  centrality_ts <- list()
  for (s in seq_along(rolling$snapshots)) {
    A <- rolling$snapshots[[s]]$adjacency
    cent <- all_centralities(A)
    top_deg <- order(-cent$degree)[1:min(5, nrow(cent))]
    centrality_ts[[s]] <- list(start = rolling$snapshots[[s]]$start,
                                top_degree = top_deg,
                                max_betweenness = which.max(cent$betweenness))
  }
  comm_ts <- list()
  for (s in seq_along(rolling$snapshots)) {
    A <- rolling$snapshots[[s]]$adjacency
    if (sum(A) > 0) {
      comm <- louvain_communities(A)
      comm_ts[[s]] <- list(n_communities = comm$n_communities,
                           modularity = comm$modularity)
    } else {
      comm_ts[[s]] <- list(n_communities = ncol(A), modularity = 0)
    }
  }
  comm_df <- data.frame(
    start = sapply(rolling$snapshots, function(s) s$start),
    n_communities = sapply(comm_ts, function(c) c$n_communities),
    modularity = sapply(comm_ts, function(c) c$modularity)
  )
  list(topology = topo, community_evolution = comm_df,
       centrality_evolution = centrality_ts, method = "Network-Evolution")
}

# ---------------------------------------------------------------------------
# Application: Identify Systemically Important Institutions
# ---------------------------------------------------------------------------
identify_sifi <- function(returns, balance_sheets = NULL, threshold = 0.5) {
  net <- correlation_adjacency(returns, method = "threshold", threshold = threshold)
  A <- net$adjacency
  cent <- all_centralities(A)
  topo <- network_topology(A)
  equity <- if (!is.null(balance_sheets)) balance_sheets$equity else NULL
  si <- systemic_importance(A, equity, method = "composite")
  comm <- louvain_communities(A)
  n <- ncol(returns)
  sifi_score <- si$scores
  sifi_rank <- rank(-sifi_score)
  is_sifi <- sifi_rank <= max(3, ceiling(n * 0.1))
  result <- data.frame(
    institution = colnames(returns),
    score = sifi_score,
    rank = sifi_rank,
    is_sifi = is_sifi,
    community = comm$membership,
    degree = cent$degree,
    betweenness = cent$betweenness,
    eigenvector = cent$eigenvector
  )
  result <- result[order(result$rank), ]
  list(sifi_table = result, network = net, topology = topo,
       communities = comm, method = "SIFI-Identification")
}

# ---------------------------------------------------------------------------
# Application: Detect Herding from Network Topology
# ---------------------------------------------------------------------------
detect_herding <- function(returns, window = 60, step = 20, threshold = 0.5) {
  evolution <- network_evolution(returns, window, step, threshold)
  topo <- evolution$topology
  density_z <- (topo$density - mean(topo$density)) / sd(topo$density)
  cc_z <- (topo$avg_cc - mean(topo$avg_cc)) / sd(topo$avg_cc)
  herding_score <- 0.5 * density_z + 0.5 * cc_z
  herding_episodes <- which(herding_score > 1.5)
  avg_corr <- numeric(nrow(topo))
  for (s in seq_along(evolution$centrality_evolution)) {
    A <- if (s <= length(evolution$centrality_evolution)) {
      snap_idx <- s
      snap <- evolution$topology$start[snap_idx]
      # re-derive from topology
      avg_corr[s] <- topo$density[s]
    }
  }
  list(herding_score = herding_score, herding_episodes = herding_episodes,
       density_z = density_z, cc_z = cc_z, topology = topo,
       method = "Herding-Detection")
}
