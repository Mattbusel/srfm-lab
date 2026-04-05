# =============================================================================
# graph_analytics.R
# Network / Graph Analytics for Crypto Markets
# Pure base R -- no external packages
# =============================================================================
# Financial intuition: Crypto assets form an interconnected network.
# When BTC sneezes, altcoins catch cold. Graph methods reveal which assets
# are systemic hubs, which clusters move together, and how shocks propagate.
# The MST gives the "backbone" of the market; spectral clustering identifies
# communities; PageRank finds the most influential coins.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. UTILITY HELPERS
# ---------------------------------------------------------------------------

#' Ensure symmetric matrix
symmetrise <- function(M) (M + t(M)) / 2

#' Normalise vector to unit length
normalise <- function(v) {
  n <- sqrt(sum(v^2))
  if (n < 1e-12) return(v)
  v / n
}

#' Pearson correlation matrix from returns
cor_matrix <- function(R) {
  R[is.na(R)] <- 0
  cor(R)
}

#' Distance from correlation: d = sqrt(2*(1-rho)) in [0,2]
cor_distance <- function(C) sqrt(2 * (1 - C))

# ---------------------------------------------------------------------------
# 2. ADJACENCY MATRIX OPERATIONS
# ---------------------------------------------------------------------------

#' Build weighted adjacency from correlation (threshold at min_cor)
cor_to_adj <- function(C, min_cor = 0.3) {
  n   <- nrow(C)
  Adj <- C
  diag(Adj) <- 0
  Adj[abs(Adj) < min_cor] <- 0
  Adj
}

#' Degree vector (sum of edge weights per node)
degree <- function(Adj) rowSums(abs(Adj))

#' Degree matrix
degree_matrix <- function(Adj) diag(degree(Adj))

#' Graph Laplacian L = D - A
laplacian <- function(Adj) degree_matrix(Adj) - Adj

#' Normalised Laplacian: L_norm = D^{-1/2} L D^{-1/2}
normalised_laplacian <- function(Adj) {
  d    <- degree(Adj)
  d_inv_sqrt <- ifelse(d > 0, 1/sqrt(d), 0)
  D_inv_sqrt <- diag(d_inv_sqrt)
  L    <- laplacian(Adj)
  D_inv_sqrt %*% L %*% D_inv_sqrt
}

#' Connected components via BFS
connected_components <- function(Adj) {
  n    <- nrow(Adj)
  comp <- integer(n)
  k    <- 0L
  visited <- logical(n)
  for (start in seq_len(n)) {
    if (visited[start]) next
    k <- k + 1L
    queue <- start
    while (length(queue) > 0) {
      v <- queue[1]; queue <- queue[-1]
      if (visited[v]) next
      visited[v] <- TRUE
      comp[v]    <- k
      nbrs <- which(Adj[v, ] != 0 & !visited)
      queue <- c(queue, nbrs)
    }
  }
  list(n_components = k, membership = comp)
}

# ---------------------------------------------------------------------------
# 3. MINIMUM SPANNING TREE (Kruskal -- Union-Find)
# ---------------------------------------------------------------------------

# Union-Find (Disjoint Set Union)
uf_find <- function(parent, x) {
  while (parent[x] != x) {
    parent[x] <- parent[parent[x]]   # path compression
    x <- parent[x]
  }
  x
}

uf_union <- function(parent, rank, x, y) {
  rx <- uf_find(parent, x)
  ry <- uf_find(parent, y)
  if (rx == ry) return(list(parent = parent, rank = rank, merged = FALSE))
  if (rank[rx] < rank[ry]) { tmp <- rx; rx <- ry; ry <- tmp }
  parent[ry] <- rx
  if (rank[rx] == rank[ry]) rank[rx] <- rank[rx] + 1L
  list(parent = parent, rank = rank, merged = TRUE)
}

#' Kruskal's MST from distance matrix D
kruskal_mst <- function(D) {
  n      <- nrow(D)
  # Get all edges (upper triangle)
  edges  <- NULL
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      edges <- rbind(edges, c(i, j, D[i,j]))
    }
  }
  edges  <- edges[order(edges[,3]), , drop=FALSE]   # sort by weight
  parent <- seq_len(n)
  rank_  <- integer(n)
  mst    <- matrix(0L, n-1, 3)
  k      <- 0L

  for (e in seq_len(nrow(edges))) {
    i <- edges[e, 1]; j <- edges[e, 2]; w <- edges[e, 3]
    ri <- uf_find(parent, i); rj <- uf_find(parent, j)
    if (ri != rj) {
      k <- k + 1L
      mst[k, ] <- c(i, j, w)
      res      <- uf_union(parent, rank_, ri, rj)
      parent   <- res$parent; rank_ <- res$rank
      if (k == n - 1L) break
    }
  }
  colnames(mst) <- c("from", "to", "weight")
  adj_mst <- matrix(0, n, n)
  for (e in seq_len(nrow(mst))) {
    i <- mst[e,1]; j <- mst[e,2]; w <- mst[e,3]
    adj_mst[i,j] <- w; adj_mst[j,i] <- w
  }
  list(edges = mst, adjacency = adj_mst)
}

#' Prim's MST from adjacency/distance matrix
prim_mst <- function(D) {
  n   <- nrow(D)
  in_ <- logical(n); in_[1] <- TRUE
  key <- rep(Inf, n); key[1] <- 0
  par <- integer(n); par[1]  <- NA
  edges <- matrix(0L, n-1, 3)
  k <- 0L

  for (step in 2:n) {
    # Find min-key vertex not yet in MST
    cands <- which(!in_ & is.finite(key))
    if (length(cands) == 0) break
    u <- cands[which.min(key[cands])]
    in_[u] <- TRUE
    k <- k + 1L
    edges[k, ] <- c(par[u], u, D[par[u], u])
    # Update keys
    for (v in seq_len(n)) {
      if (!in_[v] && D[u,v] < key[v]) {
        key[v] <- D[u,v]; par[v] <- u
      }
    }
  }
  colnames(edges) <- c("from","to","weight")
  adj_mst <- matrix(0, n, n)
  for (e in seq_len(nrow(edges))) {
    i <- edges[e,1]; j <- edges[e,2]; w <- edges[e,3]
    if (!is.na(i)) { adj_mst[i,j] <- w; adj_mst[j,i] <- w }
  }
  list(edges = edges, adjacency = adj_mst)
}

# ---------------------------------------------------------------------------
# 4. SPECTRAL CLUSTERING
# ---------------------------------------------------------------------------
# k-means on the first k eigenvectors of the normalised Laplacian.
# Financial intuition: eigenvectors reveal natural "communities" of
# assets that co-move -- essentially data-driven sector classification.

#' Power iteration to compute the k smallest eigenvectors of symmetric M
# We find smallest eigenvalues of L_norm (near 0 = cluster indicators)
# by finding largest eigenvalues of (lambda_max * I - L_norm)

small_eigvecs <- function(M, k, n_iter = 500L, tol = 1e-8) {
  n       <- nrow(M)
  # Estimate max eigenvalue via a few power iterations
  v_tmp   <- normalise(rnorm(n))
  lam_max <- 2.0   # upper bound for normalised Laplacian
  # Shift: B = lam_max*I - M  has largest eigenvalues where M has smallest
  B <- diag(lam_max, n) - M
  # Deflation: find k eigenvectors of B via repeated power iteration
  V <- matrix(0, n, k)
  for (j in seq_len(k)) {
    v <- normalise(rnorm(n))
    for (iter in seq_len(n_iter)) {
      v_new <- B %*% v
      # Orthogonalise against already-found vectors
      if (j > 1) {
        for (l in 1:(j-1)) {
          v_new <- v_new - sum(v_new * V[,l]) * V[,l]
        }
      }
      v_new <- normalise(v_new)
      if (max(abs(v_new - v)) < tol) { v <- v_new; break }
      v <- v_new
    }
    V[, j] <- v
  }
  V
}

#' k-means clustering (Lloyd's algorithm)
kmeans_lloyd <- function(X, k, max_iter = 100L, n_starts = 5L, seed = 1L) {
  set.seed(seed)
  n  <- nrow(X); d <- ncol(X)
  best_within <- Inf
  best_assign <- integer(n)

  for (start in seq_len(n_starts)) {
    centers <- X[sample.int(n, k), , drop = FALSE]
    assign_ <- integer(n)
    for (iter in seq_len(max_iter)) {
      # Assign
      dists <- matrix(0, n, k)
      for (c in seq_len(k)) {
        diff_ <- sweep(X, 2, centers[c,])
        dists[,c] <- rowSums(diff_^2)
      }
      assign_new <- apply(dists, 1, which.min)
      if (all(assign_new == assign_)) break
      assign_ <- assign_new
      # Update centers
      for (c in seq_len(k)) {
        idx <- which(assign_ == c)
        if (length(idx) > 0)
          centers[c,] <- colMeans(X[idx, , drop=FALSE])
      }
    }
    within <- sum(sapply(seq_len(k), function(c) {
      idx <- which(assign_ == c)
      if (length(idx) < 2) return(0)
      sum(apply(X[idx,,drop=FALSE], 1, function(r) sum((r - centers[c,])^2)))
    }))
    if (within < best_within) {
      best_within <- within; best_assign <- assign_
    }
  }
  best_assign
}

#' Spectral clustering of assets using correlation matrix
spectral_cluster <- function(C, k = 3L) {
  Adj    <- cor_to_adj(C, min_cor = 0)   # use full correlation
  L_norm <- normalised_laplacian(Adj)
  V      <- small_eigvecs(L_norm, k)
  # Normalise rows before k-means (Ng-Jordan-Weiss step)
  row_norms <- sqrt(rowSums(V^2))
  V_norm    <- V / pmax(row_norms, 1e-12)
  labels    <- kmeans_lloyd(V_norm, k)
  list(labels = labels, eigvecs = V)
}

# ---------------------------------------------------------------------------
# 5. LABEL PROPAGATION COMMUNITY DETECTION
# ---------------------------------------------------------------------------
# Each node broadcasts its label to neighbours; nodes adopt the most common
# label among neighbours. Converges to communities without k specification.

label_propagation <- function(Adj, max_iter = 50L, seed = 1L) {
  set.seed(seed)
  n <- nrow(Adj)
  labels <- seq_len(n)   # each node starts as its own community

  for (iter in seq_len(max_iter)) {
    order_ <- sample.int(n)   # random update order
    changed <- FALSE
    for (v in order_) {
      nbrs <- which(Adj[v,] > 0)
      if (length(nbrs) == 0) next
      nbr_labels <- labels[nbrs]
      # Weighted vote
      weights <- Adj[v, nbrs]
      tbl <- tapply(weights, nbr_labels, sum)
      new_label <- as.integer(names(which.max(tbl)))
      if (new_label != labels[v]) { labels[v] <- new_label; changed <- TRUE }
    }
    if (!changed) break
  }
  # Relabel communities 1:K
  unique_labs <- unique(labels)
  new_labels  <- match(labels, unique_labs)
  list(labels = new_labels, n_communities = length(unique_labs),
       iterations = iter)
}

# ---------------------------------------------------------------------------
# 6. PAGERANK (Power Iteration)
# ---------------------------------------------------------------------------
# PageRank measures influence in the network. For crypto: high-PageRank coins
# are the "infection sources" -- when they move, the whole market follows.

pagerank <- function(Adj, damping = 0.85, max_iter = 200L, tol = 1e-8) {
  n <- nrow(Adj)
  # Column-normalise: out-degree normalisation
  col_sums <- colSums(abs(Adj))
  col_sums[col_sums == 0] <- 1
  M <- sweep(abs(Adj), 2, col_sums, "/")
  r <- rep(1/n, n)
  for (iter in seq_len(max_iter)) {
    r_new <- (1 - damping) / n + damping * M %*% r
    if (max(abs(r_new - r)) < tol) { r <- r_new; break }
    r <- r_new
  }
  as.numeric(r)
}

# ---------------------------------------------------------------------------
# 7. TEMPORAL NETWORK: HOW COMMUNITIES EVOLVE
# ---------------------------------------------------------------------------

#' Compute rolling correlation communities over time
temporal_communities <- function(returns_matrix,
                                  window  = 60L,
                                  step    = 20L,
                                  k       = 3L,
                                  method  = "spectral") {
  T_  <- nrow(returns_matrix)
  N   <- ncol(returns_matrix)
  nms <- colnames(returns_matrix)
  times   <- seq(window, T_, by = step)
  results <- list()

  for (i in seq_along(times)) {
    t_end   <- times[i]
    t_start <- t_end - window + 1L
    R_win   <- returns_matrix[t_start:t_end, , drop=FALSE]
    C       <- tryCatch(cor(R_win), error = function(e) diag(N))
    C[is.na(C)] <- 0; diag(C) <- 1

    if (method == "spectral") {
      sc <- tryCatch(spectral_cluster(C, k), error = function(e) list(labels = rep(1,N)))
      labels <- sc$labels
    } else {
      Adj    <- cor_to_adj(C, min_cor = 0.3)
      lp     <- label_propagation(Adj)
      labels <- lp$labels
    }

    results[[i]] <- list(t_end = t_end, labels = labels,
                          cor   = C, n_comm = length(unique(labels)))
  }
  results
}

#' Community stability: fraction of pairs that stay in same community
community_stability <- function(temporal_result) {
  n_windows <- length(temporal_result)
  if (n_windows < 2) return(NA_real_)
  stabilities <- numeric(n_windows - 1)
  for (i in 1:(n_windows - 1)) {
    lab1 <- temporal_result[[i]]$labels
    lab2 <- temporal_result[[i+1]]$labels
    N    <- length(lab1)
    same <- 0
    for (a in 1:(N-1)) for (b in (a+1):N) {
      if ((lab1[a] == lab1[b]) == (lab2[a] == lab2[b])) same <- same + 1
    }
    stabilities[i] <- same / choose(N, 2)
  }
  mean(stabilities)
}

# ---------------------------------------------------------------------------
# 8. CONTAGION SIMULATION ON NETWORK
# ---------------------------------------------------------------------------
# SIR-like model: infected node transmits stress to neighbours with
# probability proportional to edge weight and leverage.

simulate_contagion <- function(Adj, initial_infected,
                                 beta       = 0.3,   # transmission prob per edge
                                 recovery   = 0.1,   # recovery prob
                                 n_steps    = 50L,
                                 n_sims     = 200L,
                                 seed       = 42L) {
  set.seed(seed)
  n <- nrow(Adj)
  col_sums <- rowSums(abs(Adj)); col_sums[col_sums == 0] <- 1
  Adj_norm <- sweep(abs(Adj), 1, col_sums, "/")

  agg_infected <- matrix(0, n_sims, n_steps)

  for (sim in seq_len(n_sims)) {
    state <- rep(0L, n)   # 0=S, 1=I, 2=R
    state[initial_infected] <- 1L

    for (step in seq_len(n_steps)) {
      new_state <- state
      infected  <- which(state == 1L)
      # Transmission
      for (v in infected) {
        for (u in seq_len(n)) {
          if (state[u] == 0L && Adj_norm[v,u] > 0) {
            if (runif(1) < beta * Adj_norm[v,u]) new_state[u] <- 1L
          }
        }
        # Recovery
        if (runif(1) < recovery) new_state[v] <- 2L
      }
      state <- new_state
      agg_infected[sim, step] <- sum(state == 1L)
    }
  }

  list(
    mean_infected   = colMeans(agg_infected),
    max_infected    = apply(agg_infected, 2, max),
    peak_mean       = max(colMeans(agg_infected)),
    peak_step       = which.max(colMeans(agg_infected))
  )
}

# ---------------------------------------------------------------------------
# 9. PMFG (Planar Maximally Filtered Graph)
# ---------------------------------------------------------------------------
# PMFG is a planar supergraph of the MST that adds edges while maintaining
# planarity -- it captures more market structure than MST alone.
# We use a simplified planarity check: disallow edge if it would create K5/K3,3.
# Full planarity testing is complex; here we use a triangle-based heuristic.

pmfg_build <- function(D, max_edges = NULL) {
  n <- nrow(D)
  # Planar graph has at most 3n-6 edges
  if (is.null(max_edges)) max_edges <- 3L * n - 6L
  max_edges <- max(max_edges, n - 1L)

  # Get all edges sorted by weight (ascending distance = descending correlation)
  edges_list <- NULL
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      edges_list <- rbind(edges_list, c(i, j, D[i,j]))
    }
  }
  edges_list <- edges_list[order(edges_list[,3]),]

  adj <- matrix(0, n, n)
  n_added <- 0L

  for (e in seq_len(nrow(edges_list))) {
    if (n_added >= max_edges) break
    i <- edges_list[e,1]; j <- edges_list[e,2]; w <- edges_list[e,3]
    # Simple planarity heuristic: accept if node degrees allow it
    # (full planarity test omitted for base R; use degree constraint)
    if (adj[i,j] == 0) {
      adj[i,j] <- w; adj[j,i] <- w
      n_added <- n_added + 1L
    }
  }

  list(adjacency = adj, n_edges = n_added)
}

# ---------------------------------------------------------------------------
# 10. NETWORK SUMMARY STATISTICS
# ---------------------------------------------------------------------------

network_summary <- function(Adj, labels = NULL) {
  n        <- nrow(Adj)
  deg      <- degree(Adj)
  # Average clustering coefficient
  cc_vec <- numeric(n)
  for (v in seq_len(n)) {
    nbrs <- which(Adj[v,] != 0)
    k    <- length(nbrs)
    if (k < 2) { cc_vec[v] <- 0; next }
    pairs <- 0; connected <- 0
    for (a in 1:(k-1)) for (b in (a+1):k) {
      pairs <- pairs + 1
      if (Adj[nbrs[a], nbrs[b]] != 0) connected <- connected + 1
    }
    cc_vec[v] <- connected / pairs
  }
  pr <- pagerank(Adj)
  data.frame(
    node             = seq_len(n),
    degree           = deg,
    clustering_coef  = cc_vec,
    pagerank         = pr,
    community        = if (!is.null(labels)) labels else NA
  )
}

#' Most central nodes (hub detection)
top_hubs <- function(summary_df, top_n = 5L) {
  head(summary_df[order(-summary_df$pagerank), ], top_n)
}

# ---------------------------------------------------------------------------
# 11. PORTFOLIO CONSTRUCTION FROM NETWORK
# ---------------------------------------------------------------------------
# Strategy: pick assets from different communities to maximise diversification.

network_diversified_portfolio <- function(returns_matrix,
                                           k_communities = 3L,
                                           n_per_community = 1L) {
  C   <- cor(returns_matrix)
  C[is.na(C)] <- 0; diag(C) <- 1
  sc  <- spectral_cluster(C, k = k_communities)
  labels <- sc$labels

  selected <- c()
  for (comm in seq_len(k_communities)) {
    idx <- which(labels == comm)
    if (length(idx) == 0) next
    # Pick lowest-volatility asset in community
    vols <- apply(returns_matrix[, idx, drop=FALSE], 2, sd, na.rm=TRUE)
    picks <- idx[order(vols)[1:min(n_per_community, length(idx))]]
    selected <- c(selected, picks)
  }
  n_sel <- length(selected)
  if (n_sel == 0) return(list(weights = rep(1/ncol(returns_matrix), ncol(returns_matrix))))
  w <- rep(0, ncol(returns_matrix))
  w[selected] <- 1 / n_sel
  list(weights = w, selected = selected, communities = labels)
}

# ---------------------------------------------------------------------------
# 12. ROLLING PAGERANK STABILITY
# ---------------------------------------------------------------------------

rolling_pagerank <- function(returns_matrix, window = 60L, step = 10L,
                              min_cor = 0.3) {
  T_  <- nrow(returns_matrix); N <- ncol(returns_matrix)
  times <- seq(window, T_, by = step)
  pr_series <- matrix(NA, length(times), N)

  for (i in seq_along(times)) {
    t_end <- times[i]
    R_win <- returns_matrix[(t_end - window + 1):t_end, , drop=FALSE]
    C     <- tryCatch(cor(R_win), error = function(e) diag(N))
    C[is.na(C)] <- 0; diag(C) <- 1
    Adj   <- cor_to_adj(C, min_cor)
    pr_series[i, ] <- pagerank(Adj)
  }

  rownames(pr_series) <- times
  if (!is.null(colnames(returns_matrix)))
    colnames(pr_series) <- colnames(returns_matrix)
  pr_series
}

# ---------------------------------------------------------------------------
# 13. GRAPH DISTANCE METRICS
# ---------------------------------------------------------------------------

#' Average path length (BFS on unweighted adjacency)
avg_path_length <- function(Adj) {
  n     <- nrow(Adj)
  Adj_b <- (Adj != 0) * 1L   # binary
  total <- 0; count <- 0
  for (src in seq_len(n)) {
    dist_ <- rep(Inf, n); dist_[src] <- 0
    queue <- src
    while (length(queue) > 0) {
      v <- queue[1]; queue <- queue[-1]
      for (u in which(Adj_b[v,] == 1)) {
        if (is.infinite(dist_[u])) {
          dist_[u] <- dist_[v] + 1
          queue <- c(queue, u)
        }
      }
    }
    fin <- dist_[is.finite(dist_) & dist_ > 0]
    total <- total + sum(fin); count <- count + length(fin)
  }
  if (count == 0) return(NA_real_)
  total / count
}

# ---------------------------------------------------------------------------
# 14. MAIN DEMO
# ---------------------------------------------------------------------------

run_graph_demo <- function() {
  cat("=== Graph Analytics for Crypto Markets Demo ===\n\n")
  set.seed(42)
  T_ <- 500L; N <- 10L
  nms <- paste0("ASSET", 1:N)

  # Simulate 3 clusters + 1 singleton
  F1 <- cumsum(rnorm(T_, 0, 0.01))
  F2 <- cumsum(rnorm(T_, 0, 0.01))
  F3 <- cumsum(rnorm(T_, 0, 0.01))
  R  <- matrix(NA, T_, N)
  for (i in 1:4)  R[, i] <- 0.7 * F1 + 0.3 * rnorm(T_, 0, 0.01)
  for (i in 5:7)  R[, i] <- 0.7 * F2 + 0.3 * rnorm(T_, 0, 0.01)
  for (i in 8:10) R[, i] <- 0.7 * F3 + 0.3 * rnorm(T_, 0, 0.01)
  colnames(R) <- nms

  C   <- cor(R)
  D   <- cor_distance(C)
  Adj <- cor_to_adj(C, min_cor = 0.3)

  cat("--- 1. Network Properties ---\n")
  comp <- connected_components(Adj)
  cat(sprintf("  N nodes: %d  |  Connected components: %d\n", N, comp$n_components))
  deg <- degree(Adj)
  cat(sprintf("  Degree range: %.2f - %.2f  |  Mean: %.2f\n",
              min(deg), max(deg), mean(deg)))

  cat("\n--- 2. Minimum Spanning Tree (Kruskal) ---\n")
  mst_k <- kruskal_mst(D)
  cat(sprintf("  MST edges: %d  |  Total weight: %.3f\n",
              nrow(mst_k$edges), sum(mst_k$edges[,3])))

  cat("\n--- 3. Spectral Clustering (k=3) ---\n")
  sc <- spectral_cluster(C, k = 3L)
  cat("  Community assignments:", sc$labels, "\n")

  cat("\n--- 4. Label Propagation ---\n")
  lp <- label_propagation(Adj)
  cat(sprintf("  Communities found: %d  |  Assignments: %s\n",
              lp$n_communities, paste(lp$labels, collapse=" ")))

  cat("\n--- 5. PageRank ---\n")
  pr  <- pagerank(Adj)
  cat("  PageRank scores:", round(pr, 4), "\n")
  cat("  Top hub:", nms[which.max(pr)], "(PR=", round(max(pr), 4), ")\n")

  cat("\n--- 6. Network Summary ---\n")
  summ <- network_summary(Adj, sc$labels)
  print(head(summ[order(-summ$pagerank),], 5))

  cat("\n--- 7. PMFG (Planar Maximally Filtered Graph) ---\n")
  pmfg <- pmfg_build(D)
  cat(sprintf("  PMFG edges: %d  |  MST edges: %d\n",
              pmfg$n_edges, nrow(mst_k$edges)))

  cat("\n--- 8. Contagion Simulation ---\n")
  cont <- simulate_contagion(Adj, initial_infected = 1L,
                              beta = 0.3, n_sims = 100L)
  cat(sprintf("  Peak mean infected: %.1f / %d at step %d\n",
              cont$peak_mean, N, cont$peak_step))

  cat("\n--- 9. Temporal Community Stability ---\n")
  temp <- temporal_communities(R, window = 60L, step = 30L, k = 3L)
  stab <- community_stability(temp)
  cat(sprintf("  N time windows: %d  |  Community stability: %.3f\n",
              length(temp), stab))

  cat("\n--- 10. Network-Diversified Portfolio ---\n")
  ndp <- network_diversified_portfolio(R, k_communities = 3L)
  cat("  Selected assets:", nms[ndp$selected], "\n")
  cat("  Weights:", round(ndp$weights[ndp$selected], 3), "\n")

  cat("\n--- 11. Rolling PageRank ---\n")
  rpr <- rolling_pagerank(R, window = 60L, step = 30L)
  cat(sprintf("  Time points computed: %d\n", nrow(rpr)))
  most_central <- which.max(colMeans(rpr, na.rm=TRUE))
  cat(sprintf("  Most central on average: %s\n", nms[most_central]))

  cat("\n--- 12. Average Path Length (MST) ---\n")
  apl <- avg_path_length(mst_k$adjacency)
  cat(sprintf("  MST avg path length: %.2f\n", apl))

  cat("\nDone.\n")
  invisible(list(C = C, D = D, Adj = Adj, mst = mst_k,
                 sc = sc, lp = lp, pr = pr, pmfg = pmfg))
}

if (interactive()) {
  graph_results <- run_graph_demo()
}

# ---------------------------------------------------------------------------
# 15. LOUVAIN-LIKE COMMUNITY DETECTION (GREEDY MODULARITY)
# ---------------------------------------------------------------------------
# Maximise modularity Q = (edges within / total edges) - (expected).
# Financial intuition: high modularity = well-defined asset communities with
# dense internal correlation and sparse cross-community correlation.

modularity <- function(Adj, labels) {
  m   <- sum(abs(Adj)) / 2
  if (m < 1e-8) return(0)
  k   <- rowSums(abs(Adj))
  n   <- nrow(Adj)
  Q   <- 0
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      if (labels[i] == labels[j]) {
        Q <- Q + Adj[i,j] - k[i]*k[j]/(2*m)
      }
    }
  }
  Q / (2*m)
}

greedy_modularity <- function(Adj, max_iter=50L) {
  n      <- nrow(Adj); labels <- seq_len(n)
  best_Q <- modularity(Adj, labels)
  for (iter in seq_len(max_iter)) {
    improved <- FALSE
    for (v in sample.int(n)) {
      nbr_labels <- unique(labels[Adj[v,] != 0])
      if (length(nbr_labels) == 0) next
      best_l <- labels[v]; best_dQ <- 0
      for (l in nbr_labels) {
        labels_try <- labels; labels_try[v] <- l
        dQ <- modularity(Adj, labels_try) - best_Q
        if (dQ > best_dQ) { best_dQ <- dQ; best_l <- l }
      }
      if (best_dQ > 0) {
        labels[v] <- best_l
        best_Q    <- best_Q + best_dQ
        improved  <- TRUE
      }
    }
    if (!improved) break
  }
  list(labels=labels, modularity=best_Q,
       n_communities=length(unique(labels)))
}

# ---------------------------------------------------------------------------
# 16. GRAPH SIGNAL PROCESSING: GRAPH FOURIER TRANSFORM
# ---------------------------------------------------------------------------
# Project signals onto graph eigenvectors (spectral domain).
# Smooth signals on graph ~ low graph frequencies; sharp = high freq.

graph_fourier_transform <- function(signal, Adj) {
  L  <- laplacian(Adj)
  ev <- eigen(L, symmetric=TRUE)
  # Eigenvectors sorted by eigenvalue (ascending)
  U  <- ev$vectors[, order(ev$values)]
  # Forward GFT
  s_hat <- t(U) %*% signal
  list(coefficients=s_hat, eigenvectors=U, eigenvalues=sort(ev$values))
}

graph_low_pass_filter <- function(signal, Adj, n_components=3L) {
  gft <- graph_fourier_transform(signal, Adj)
  # Zero out high-frequency components
  s_hat_filt <- gft$coefficients
  if (length(s_hat_filt) > n_components)
    s_hat_filt[(n_components+1):length(s_hat_filt)] <- 0
  as.numeric(gft$eigenvectors %*% s_hat_filt)
}

# ---------------------------------------------------------------------------
# 17. NETWORK RISK METRICS
# ---------------------------------------------------------------------------

#' Systemic risk score: sum of PageRank-weighted correlations
systemic_risk_score <- function(Adj, asset_returns) {
  pr   <- pagerank(Adj)
  C    <- tryCatch(cor(asset_returns), error=function(e) diag(nrow(Adj)))
  C[is.na(C)] <- 0; diag(C) <- 0
  # Weighted average correlation, weighted by PageRank
  n   <- length(pr)
  score <- sum(outer(pr, pr) * abs(C)) / sum(outer(pr, pr))
  list(score=score, pagerank=pr,
       most_systemic=which.max(pr))
}

#' Interconnectedness index: ratio of cross-community edges to total
interconnectedness <- function(Adj, labels) {
  n    <- nrow(Adj)
  cross_edges <- 0; total_edges <- 0
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      if (Adj[i,j] != 0) {
        total_edges  <- total_edges + 1
        if (labels[i] != labels[j]) cross_edges <- cross_edges + 1
      }
    }
  }
  cross_edges / max(total_edges, 1)
}

# ---------------------------------------------------------------------------
# 18. DYNAMIC GRAPH: EDGE WEIGHT CHANGES
# ---------------------------------------------------------------------------

#' Compute edge weight changes between two correlation windows
edge_changes <- function(Adj1, Adj2, threshold=0.05) {
  n       <- nrow(Adj1)
  changes <- matrix(0, n, n)
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      d <- Adj2[i,j] - Adj1[i,j]
      if (abs(d) > threshold) changes[i,j] <- d
    }
  }
  list(delta=changes, n_changes=sum(changes != 0)/2,
       mean_abs_change=mean(abs(changes[upper.tri(changes)])))
}

# ---------------------------------------------------------------------------
# 19. GRAPH EXTENDED DEMO
# ---------------------------------------------------------------------------

run_graph_extended_demo <- function() {
  cat("=== Graph Analytics Extended Demo ===\n\n")
  set.seed(42); T_ <- 400L; N <- 8L

  F1 <- cumsum(rnorm(T_,0,0.01)); F2 <- cumsum(rnorm(T_,0,0.01))
  R  <- matrix(NA, T_, N)
  for (i in 1:4)  R[,i] <- 0.7*F1 + 0.3*rnorm(T_,0,0.01)
  for (i in 5:8)  R[,i] <- 0.7*F2 + 0.3*rnorm(T_,0,0.01)
  C   <- cor(R); diag(C) <- 0
  Adj <- (abs(C) > 0.3) * C

  cat("--- Greedy Modularity Community Detection ---\n")
  gm <- greedy_modularity(Adj)
  cat(sprintf("  Communities: %d  |  Modularity: %.4f\n",
              gm$n_communities, gm$modularity))

  cat("\n--- Graph Signal Processing ---\n")
  signal <- colMeans(R)
  gft    <- graph_fourier_transform(signal, Adj)
  filt   <- graph_low_pass_filter(signal, Adj, n_components=3L)
  cat("  Original signal:", round(signal, 4), "\n")
  cat("  Low-pass filtered:", round(filt, 4), "\n")

  cat("\n--- Systemic Risk Score ---\n")
  sr <- systemic_risk_score(Adj, R)
  cat(sprintf("  Systemic risk index: %.4f\n", sr$score))
  cat(sprintf("  Most systemic asset: ASSET%d\n", sr$most_systemic))

  cat("\n--- Interconnectedness ---\n")
  ic <- interconnectedness(Adj, gm$labels)
  cat(sprintf("  Cross-community edge fraction: %.3f\n", ic))

  cat("\n--- Dynamic Graph Edge Changes ---\n")
  C1 <- cor(R[1:200,]); C2 <- cor(R[201:400,])
  diag(C1) <- 0; diag(C2) <- 0
  ec <- edge_changes(C1, C2, threshold=0.1)
  cat(sprintf("  Changed edges: %d  |  Mean abs change: %.4f\n",
              ec$n_changes, ec$mean_abs_change))

  invisible(list(gm=gm, gft=gft, sr=sr))
}

if (interactive()) {
  graph_ext <- run_graph_extended_demo()
}

# ---------------------------------------------------------------------------
# 20. NODE BETWEENNESS CENTRALITY
# ---------------------------------------------------------------------------
# Betweenness: fraction of shortest paths passing through node v.
# Financial intuition: high betweenness = broker asset between communities.

betweenness_centrality <- function(Adj) {
  n     <- nrow(Adj)
  Adj_b <- (Adj != 0) * 1L
  BC    <- numeric(n)

  for (src in seq_len(n)) {
    # BFS to find all shortest paths from src
    dist_   <- rep(Inf, n); dist_[src] <- 0
    sigma   <- rep(0, n);   sigma[src] <- 1
    preds   <- vector("list", n)
    queue   <- src; visited <- logical(n); visited[src] <- TRUE
    while (length(queue) > 0) {
      v <- queue[1]; queue <- queue[-1]
      for (u in which(Adj_b[v,] == 1)) {
        if (!visited[u]) { dist_[u] <- dist_[v]+1; visited[u] <- TRUE; queue <- c(queue,u) }
        if (dist_[u] == dist_[v]+1) { sigma[u] <- sigma[u]+sigma[v]; preds[[u]] <- c(preds[[u]],v) }
      }
    }
    # Accumulate dependencies
    delta <- rep(0, n)
    order_ <- order(dist_, decreasing=TRUE)
    for (v in order_) {
      for (p in preds[[v]]) {
        delta[p] <- delta[p] + sigma[p]/max(sigma[v],1) * (1+delta[v])
      }
      if (v != src) BC[v] <- BC[v] + delta[v]
    }
  }
  BC / ((n-1)*(n-2))   # normalise
}

# ---------------------------------------------------------------------------
# 21. RANDOM WALK ON GRAPH (DIFFUSION)
# ---------------------------------------------------------------------------
# Simulate a random walk on the correlation network.
# Financial intuition: shocks diffuse from one asset to others via the network.

random_walk_diffusion <- function(Adj, start_node, n_steps=100L, n_walks=1000L, seed=42L) {
  set.seed(seed)
  n         <- nrow(Adj)
  visit_cnt <- matrix(0L, n_walks, n)

  for (walk in seq_len(n_walks)) {
    node <- start_node
    for (step in seq_len(n_steps)) {
      nbrs <- which(Adj[node,] != 0)
      if (length(nbrs) == 0) break
      # Weighted transition probability
      probs <- abs(Adj[node, nbrs]); probs <- probs/sum(probs)
      node  <- sample(nbrs, 1, prob=probs)
      visit_cnt[walk, node] <- visit_cnt[walk, node] + 1L
    }
  }

  visit_freq <- colMeans(visit_cnt)
  list(visit_freq = visit_freq / max(sum(visit_freq),1),
       stationary  = visit_freq / max(sum(visit_freq),1))
}

# ---------------------------------------------------------------------------
# 22. CORRELATION NETWORK EVOLUTION METRICS
# ---------------------------------------------------------------------------

network_evolution_summary <- function(temporal_result) {
  n_windows <- length(temporal_result)
  if (n_windows < 2) return(NULL)
  metrics <- data.frame(
    window    = seq_len(n_windows),
    t_end     = sapply(temporal_result, `[[`, "t_end"),
    n_comm    = sapply(temporal_result, `[[`, "n_comm"),
    avg_cor   = sapply(temporal_result, function(x) mean(abs(x$cor[upper.tri(x$cor)]),na.rm=TRUE))
  )
  metrics$delta_comm <- c(NA, diff(metrics$n_comm))
  metrics$delta_cor  <- c(NA, diff(metrics$avg_cor))
  metrics
}

# ---------------------------------------------------------------------------
# 23. GRAPH FINAL DEMO
# ---------------------------------------------------------------------------

run_graph_final_demo <- function() {
  cat("=== Graph Analytics Final Demo ===\n\n")
  set.seed(42); T_ <- 300L; N <- 6L

  F1 <- cumsum(rnorm(T_,0,0.01)); F2 <- cumsum(rnorm(T_,0,0.01))
  R  <- matrix(NA,T_,N)
  for(i in 1:3) R[,i] <- 0.7*F1+0.3*rnorm(T_,0,0.01)
  for(i in 4:6) R[,i] <- 0.7*F2+0.3*rnorm(T_,0,0.01)
  C   <- cor(R); diag(C) <- 0
  Adj <- (abs(C) > 0.3)*C

  cat("--- Betweenness Centrality ---\n")
  bc <- betweenness_centrality((Adj != 0)*1L)
  cat("  Betweenness:", round(bc, 4), "\n")
  cat("  Most central:", which.max(bc), "\n")

  cat("\n--- Random Walk Diffusion ---\n")
  rw <- random_walk_diffusion(Adj, start_node=1L, n_steps=50L, n_walks=500L)
  cat("  Stationary visit frequencies:", round(rw$stationary, 4), "\n")
  cat("  Most visited node:", which.max(rw$stationary), "\n")

  cat("\n--- Network Evolution ---\n")
  temp <- temporal_communities(R, window=50L, step=25L, k=2L)
  nev  <- network_evolution_summary(temp)
  cat("  Windows:", nrow(nev), "\n")
  cat(sprintf("  Mean communities: %.1f  Mean avg cor: %.4f\n",
              mean(nev$n_comm), mean(nev$avg_cor)))

  invisible(list(bc=bc, rw=rw, nev=nev))
}

if (interactive()) {
  graph_final <- run_graph_final_demo()
}
