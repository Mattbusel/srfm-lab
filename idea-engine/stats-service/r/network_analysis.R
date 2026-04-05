# =============================================================================
# network_analysis.R
# Financial network analysis for crypto markets
# Base R only -- no external packages
# =============================================================================
# Financial intuition: Crypto assets are not isolated -- they form a network
# of interconnected price movements. Network analysis reveals which coins
# are "hubs" (most connected), which clusters form, and how systemic risk
# propagates through the ecosystem. BTC is typically the central hub, but
# DeFi tokens form their own sub-cluster.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. MANTEGNA DISTANCE METRIC
# -----------------------------------------------------------------------------

#' Mantegna (1999) ultrametric distance for correlations
#' d(i,j) = sqrt(2*(1 - rho_ij))
#' Maps Pearson correlations to a proper metric space for MST construction
#' @param cor_mat correlation matrix
mantegna_distance <- function(cor_mat) {
  sqrt(2 * (1 - cor_mat))
}

#' Compute pairwise distance matrix from returns
#' @param returns_mat T x N matrix of returns
returns_to_distance <- function(returns_mat) {
  cor_mat <- cor(returns_mat, use = "pairwise.complete.obs")
  mantegna_distance(cor_mat)
}

# -----------------------------------------------------------------------------
# 2. MINIMUM SPANNING TREE (MST)
# -----------------------------------------------------------------------------

#' Build Minimum Spanning Tree using Kruskal's algorithm
#' MST connects all N nodes using N-1 edges of minimum total weight
#' In finance: MST of correlation-based distances reveals the backbone
#' of market structure -- the most important dependencies
#' @param dist_mat symmetric distance matrix (N x N)
#' @return list: edges (data.frame), adjacency matrix
minimum_spanning_tree <- function(dist_mat) {
  N <- nrow(dist_mat)
  asset_names <- rownames(dist_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", seq_len(N))

  # Create edge list, sorted by distance
  edges <- data.frame(i=integer(), j=integer(), dist=numeric())
  for (i in 1:(N-1)) {
    for (j in (i+1):N) {
      edges <- rbind(edges, data.frame(i=i, j=j, dist=dist_mat[i,j]))
    }
  }
  edges <- edges[order(edges$dist), ]

  # Kruskal: Union-Find data structure
  parent <- seq_len(N)
  find_root <- function(x) {
    while (parent[x] != x) x <- parent[x]
    x
  }
  union_sets <- function(x, y) {
    rx <- find_root(x); ry <- find_root(y)
    if (rx != ry) parent[rx] <<- ry
  }

  mst_edges <- data.frame(i=integer(), j=integer(), dist=numeric(),
                            name_i=character(), name_j=character())
  for (k in seq_len(nrow(edges))) {
    e <- edges[k, ]
    if (find_root(e$i) != find_root(e$j)) {
      union_sets(e$i, e$j)
      mst_edges <- rbind(mst_edges, data.frame(
        i=e$i, j=e$j, dist=e$dist,
        name_i=asset_names[e$i], name_j=asset_names[e$j]
      ))
      if (nrow(mst_edges) == N - 1) break
    }
  }

  # Build adjacency matrix
  adj <- matrix(0, N, N, dimnames=list(asset_names, asset_names))
  for (k in seq_len(nrow(mst_edges))) {
    e <- mst_edges[k,]
    adj[e$i, e$j] <- 1; adj[e$j, e$i] <- 1
  }

  # Node degree in MST
  degree <- rowSums(adj)
  # Hub: highest degree node
  hub <- which.max(degree)

  cat("=== Minimum Spanning Tree ===\n")
  cat(sprintf("Nodes: %d, Edges: %d\n", N, nrow(mst_edges)))
  cat(sprintf("Hub node: %s (degree=%d)\n", asset_names[hub], degree[hub]))
  cat("Top 5 shortest edges (most correlated pairs):\n")
  print(head(mst_edges[order(mst_edges$dist), c("name_i","name_j","dist")], 5))

  invisible(list(edges=mst_edges, adjacency=adj, degree=degree,
                 hub=asset_names[hub], hub_degree=degree[hub]))
}

# -----------------------------------------------------------------------------
# 3. PLANAR MAXIMALLY FILTERED GRAPH (PMFG)
# -----------------------------------------------------------------------------

#' Build Planar Maximally Filtered Graph (PMFG)
#' PMFG retains 3*(N-2) edges (more than MST's N-1) while remaining planar
#' Contains MST as subgraph; provides richer market structure information
#' @param dist_mat distance matrix
pmfg <- function(dist_mat) {
  N <- nrow(dist_mat)
  asset_names <- rownames(dist_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", seq_len(N))
  max_edges <- 3 * (N - 2)

  # Edge list sorted by distance
  edges_all <- data.frame(i=integer(), j=integer(), dist=numeric())
  for (i in 1:(N-1)) {
    for (j in (i+1):N) {
      edges_all <- rbind(edges_all, data.frame(i=i, j=j, dist=dist_mat[i,j]))
    }
  }
  edges_all <- edges_all[order(edges_all$dist), ]

  # Planarity check: simplified Kuratowski-based check via adjacency
  # Full planarity testing is complex; we use a simplified version:
  # For small N (<= 20), check if adding edge creates K5 or K3,3 subgraph
  # For larger N, use approximation: limit edges per node
  adj   <- matrix(0, N, N)
  pmfg_edges <- data.frame(i=integer(), j=integer(), dist=numeric())
  n_edges <- 0

  for (k in seq_len(nrow(edges_all))) {
    if (n_edges >= max_edges) break
    e <- edges_all[k, ]
    i <- e$i; j <- e$j

    # Simple planarity heuristic: check if adding edge would create K5
    # (degree > 4 for any node in a complete subgraph = impossible in planar)
    # Use a simpler check: max degree <= N-1, no two nodes both connected to 4+ common nodes
    can_add <- TRUE
    if (N <= 20) {
      # Check common neighbors
      common_n <- sum(adj[i,] & adj[j,])
      if (common_n >= 3) {
        # Potential K4 subgraph, check more carefully
        common_idx <- which(adj[i,] & adj[j,])
        # If all common neighbors are also connected = K4 already
        if (length(common_idx) >= 3) {
          triplets <- combn(common_idx, 3)
          for (col in seq_len(ncol(triplets))) {
            a <- triplets[1,col]; b <- triplets[2,col]; c_n <- triplets[3,col]
            if (adj[a,b] && adj[b,c_n] && adj[a,c_n] &&
                adj[i,a] && adj[i,b] && adj[i,c_n] &&
                adj[j,a] && adj[j,b] && adj[j,c_n]) {
              can_add <- FALSE; break
            }
          }
        }
      }
    }

    if (can_add) {
      adj[i,j] <- 1; adj[j,i] <- 1
      n_edges <- n_edges + 1
      pmfg_edges <- rbind(pmfg_edges, data.frame(
        i=i, j=j, dist=e$dist,
        name_i=asset_names[i], name_j=asset_names[j]
      ))
    }
  }

  degree <- rowSums(adj)
  rownames(adj) <- colnames(adj) <- asset_names

  cat("=== PMFG ===\n")
  cat(sprintf("Nodes: %d, Edges: %d (target: %d)\n", N, n_edges, max_edges))

  invisible(list(edges=pmfg_edges, adjacency=adj, degree=degree))
}

# -----------------------------------------------------------------------------
# 4. COMMUNITY DETECTION (GREEDY MODULARITY)
# -----------------------------------------------------------------------------

#' Greedy modularity community detection (Newman 2004)
#' Finds clusters of highly correlated assets (communities in the network)
#' Modularity Q measures quality: Q > 0.3 indicates good community structure
#' @param adj_mat adjacency matrix (binary or weighted)
greedy_modularity <- function(adj_mat) {
  N <- nrow(adj_mat)
  asset_names <- rownames(adj_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", seq_len(N))

  m <- sum(adj_mat) / 2  # total edges
  if (m == 0) return(list(communities=seq_len(N), modularity=0))

  # Degree vector
  k <- rowSums(adj_mat)

  # Start: each node is its own community
  community <- seq_len(N)

  # Modularity of a partition
  modularity <- function(comm) {
    Q <- 0
    for (c in unique(comm)) {
      idx <- which(comm == c)
      # Internal edges
      e_c <- sum(adj_mat[idx, idx]) / 2
      # Total degree within community
      a_c <- sum(k[idx]) / 2
      Q <- Q + e_c/m - (a_c/m)^2
    }
    Q
  }

  Q_curr <- modularity(community)

  # Greedy merge: at each step merge pair of communities that increases Q most
  for (iter in seq_len(N - 1)) {
    best_dQ <- -Inf
    best_merge <- c(NA, NA)
    unique_comms <- unique(community)
    if (length(unique_comms) < 2) break

    for (ci in seq_len(length(unique_comms)-1)) {
      for (cj in (ci+1):length(unique_comms)) {
        c1 <- unique_comms[ci]; c2 <- unique_comms[cj]
        # Proposed merged community
        community_new <- community
        community_new[community == c2] <- c1
        dQ <- modularity(community_new) - Q_curr
        if (dQ > best_dQ) {
          best_dQ <- dQ
          best_merge <- c(c1, c2)
        }
      }
    }

    if (best_dQ <= 0 && iter > 2) break
    # Execute best merge
    community[community == best_merge[2]] <- best_merge[1]
    Q_curr <- Q_curr + best_dQ
  }

  # Relabel communities 1..K
  comm_labels <- match(community, unique(community))
  n_communities <- max(comm_labels)

  cat("=== Community Detection (Greedy Modularity) ===\n")
  cat(sprintf("Communities found: %d\n", n_communities))
  cat(sprintf("Modularity Q = %.4f (%s)\n", Q_curr,
              ifelse(Q_curr > 0.3, "strong", ifelse(Q_curr > 0.1, "moderate", "weak"))))
  for (c in seq_len(n_communities)) {
    members <- asset_names[comm_labels == c]
    cat(sprintf("  Community %d (%d assets): %s\n",
                c, length(members), paste(members, collapse=", ")))
  }

  invisible(list(communities=comm_labels, modularity=Q_curr,
                 n_communities=n_communities))
}

# -----------------------------------------------------------------------------
# 5. SYSTEMIC RISK: CoVaR, MES, SRISK
# -----------------------------------------------------------------------------

#' CoVaR: VaR of financial system conditional on institution i being in distress
#' DeltaCoVaR = CoVaR_{system|distress} - CoVaR_{system|normal}
#' @param r_i return series of institution i
#' @param r_sys return series of the system (e.g., market)
#' @param alpha VaR confidence level
covar <- function(r_i, r_sys, alpha = 0.05) {
  # Quantile regression of r_sys on r_i
  # CoVaR at quantile alpha = fitted value when r_i = VaR_alpha(i)
  # We use a simple linear quantile regression
  n <- length(r_i)
  tau <- alpha

  # Quantile regression via linear programming (simplex method)
  quantile_reg <- function(y, x, tau) {
    n <- length(y)
    X <- cbind(1, x)
    # Use iteratively reweighted least squares approximation
    b <- lm(y ~ x)$coefficients
    for (iter in 1:50) {
      resid <- y - X %*% b
      weights <- ifelse(resid >= 0, tau, 1 - tau) / abs(resid + 1e-8)
      W  <- diag(weights)
      b_new <- solve(t(X) %*% W %*% X) %*% t(X) %*% W %*% y
      if (max(abs(b_new - b)) < 1e-6) break
      b <- b_new
    }
    b
  }

  # VaR of institution i at level alpha
  var_i_normal  <- quantile(r_i, tau)      # alpha quantile
  var_i_distress <- quantile(r_i, alpha)    # same (conditioning on distress)

  # Estimate CoVaR
  b_covar <- quantile_reg(r_sys, r_i, tau)
  covar_normal    <- b_covar[1] + b_covar[2] * median(r_i)
  covar_distress  <- b_covar[1] + b_covar[2] * var_i_normal
  delta_covar     <- covar_distress - covar_normal

  cat(sprintf("CoVaR (alpha=%.2f): DeltaCoVaR = %.4f\n", alpha, delta_covar))
  list(covar_distress=covar_distress, covar_normal=covar_normal,
       delta_covar=delta_covar, beta_covar=b_covar[2])
}

#' Marginal Expected Shortfall (MES)
#' MES_i = E[r_i | r_sys < VaR_alpha(r_sys)]
#' Higher MES = institution contributes more to system losses in crises
#' @param r_i return series of asset i
#' @param r_sys system return
#' @param alpha tail probability
mes <- function(r_i, r_sys, alpha = 0.05) {
  threshold <- quantile(r_sys, alpha)
  crisis_idx <- r_sys < threshold
  if (sum(crisis_idx) < 5) return(NA)
  mes_val <- mean(r_i[crisis_idx])
  mes_val
}

#' Compute systemic risk matrix: DeltaCoVaR and MES for all assets vs system
systemic_risk_matrix <- function(returns_mat, system_idx = 1, alpha = 0.05) {
  N <- ncol(returns_mat)
  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", seq_len(N))
  r_sys <- returns_mat[, system_idx]

  results <- lapply(seq_len(N), function(i) {
    if (i == system_idx) return(list(mes=NA, delta_covar=NA))
    r_i <- returns_mat[, i]
    mes_i  <- mes(r_i, r_sys, alpha)
    covar_i <- covar(r_i, r_sys, alpha)
    list(mes=mes_i, delta_covar=covar_i$delta_covar)
  })

  mes_vals   <- sapply(results, `[[`, "mes")
  dcovar_vals <- sapply(results, `[[`, "delta_covar")

  df <- data.frame(
    asset = asset_names,
    MES   = mes_vals,
    DeltaCoVaR = dcovar_vals
  )
  df <- df[order(df$MES), ]

  cat("=== Systemic Risk Measures ===\n")
  cat(sprintf("System: %s (alpha=%.2f)\n", asset_names[system_idx], alpha))
  print(df)

  invisible(df)
}

# -----------------------------------------------------------------------------
# 6. GRANGER CAUSALITY NETWORK (VAR-based)
# -----------------------------------------------------------------------------

#' Build Granger causality network: fit bivariate VARs for all pairs
#' Edge i->j exists if lagged r_i Granger-causes r_j (F-test p < threshold)
#' @param returns_mat T x N matrix
#' @param p VAR lag order
#' @param alpha significance level
granger_network <- function(returns_mat, p = 5, alpha = 0.05) {
  N <- ncol(returns_mat)
  T_obs <- nrow(returns_mat)
  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", seq_len(N))

  # Build design matrix for VAR(p)
  gc_mat <- matrix(NA, N, N)  # gc_mat[i,j] = p-value for i->j causality
  rownames(gc_mat) <- colnames(gc_mat) <- asset_names

  for (j in seq_len(N)) {
    y <- returns_mat[(p+1):T_obs, j]
    T_eff <- length(y)

    # Unrestricted model: y on lags of y AND lags of each other series
    for (i in seq_len(N)) {
      if (i == j) { gc_mat[i,j] <- NA; next }

      # Restricted: y regressed on own lags only
      X_r <- matrix(1, T_eff, p*1 + 1)
      for (k in seq_len(p)) X_r[, k+1] <- returns_mat[(p+1-k):(T_obs-k), j]
      b_r <- solve(t(X_r)%*%X_r) %*% t(X_r) %*% y
      ss_r <- sum((y - X_r%*%b_r)^2)

      # Unrestricted: also include lags of series i
      X_u <- cbind(X_r, matrix(0, T_eff, p))
      for (k in seq_len(p)) X_u[, p+1+k] <- returns_mat[(p+1-k):(T_obs-k), i]
      b_u <- solve(t(X_u)%*%X_u) %*% t(X_u) %*% y
      ss_u <- sum((y - X_u%*%b_u)^2)

      # F-test
      df1 <- p; df2 <- T_eff - 2*p - 1
      if (df2 < 1 || ss_u == 0) { gc_mat[i,j] <- NA; next }
      F_stat <- ((ss_r - ss_u) / df1) / (ss_u / df2)
      gc_mat[i,j] <- pf(F_stat, df1, df2, lower.tail=FALSE)
    }
  }

  # Binary adjacency matrix
  gc_adj <- (gc_mat < alpha) * 1
  gc_adj[is.na(gc_adj)] <- 0

  # In-degree and out-degree
  out_degree <- rowSums(gc_adj, na.rm=TRUE)
  in_degree  <- colSums(gc_adj, na.rm=TRUE)

  cat("=== Granger Causality Network ===\n")
  cat(sprintf("Assets: %d, Significant edges: %d\n", N, sum(gc_adj)/2))
  cat("Out-degree (causes others):\n")
  print(sort(out_degree, decreasing=TRUE))
  cat("In-degree (caused by others):\n")
  print(sort(in_degree, decreasing=TRUE))

  invisible(list(gc_pval=gc_mat, gc_adj=gc_adj,
                 out_degree=out_degree, in_degree=in_degree))
}

# -----------------------------------------------------------------------------
# 7. NETWORK CENTRALITY MEASURES
# -----------------------------------------------------------------------------

#' Degree centrality: normalized degree
degree_centrality <- function(adj_mat) {
  N <- nrow(adj_mat)
  rowSums(adj_mat) / (N - 1)
}

#' Betweenness centrality: fraction of shortest paths passing through node
#' Uses BFS for unweighted graphs
betweenness_centrality <- function(adj_mat) {
  N <- nrow(adj_mat)
  betweenness <- numeric(N)

  bfs_paths <- function(source) {
    dist  <- rep(Inf, N); dist[source] <- 0
    sigma <- numeric(N); sigma[source] <- 1
    queue <- c(source)
    pred  <- vector("list", N)
    visited <- logical(N); visited[source] <- TRUE

    while (length(queue) > 0) {
      v <- queue[1]; queue <- queue[-1]
      for (w in which(adj_mat[v, ] > 0)) {
        if (!visited[w]) {
          visited[w] <- TRUE
          dist[w] <- dist[v] + 1
          queue <- c(queue, w)
        }
        if (dist[w] == dist[v] + 1) {
          sigma[w] <- sigma[w] + sigma[v]
          pred[[w]] <- c(pred[[w]], v)
        }
      }
    }
    list(dist=dist, sigma=sigma, pred=pred)
  }

  for (s in seq_len(N)) {
    bfs_res <- bfs_paths(s)
    # Back-propagation of dependencies
    delta <- numeric(N)
    # Process nodes in reverse order of distance
    order_nodes <- order(bfs_res$dist, decreasing=TRUE)
    for (w in order_nodes) {
      if (is.infinite(bfs_res$dist[w])) next
      for (v in bfs_res$pred[[w]]) {
        if (bfs_res$sigma[w] > 0) {
          delta[v] <- delta[v] + (bfs_res$sigma[v] / bfs_res$sigma[w]) * (1 + delta[w])
        }
      }
      if (w != s) betweenness[w] <- betweenness[w] + delta[w]
    }
  }

  # Normalize
  betweenness / ((N-1)*(N-2))
}

#' Eigenvector centrality: PageRank-like, nodes connected to important nodes are important
eigenvector_centrality <- function(adj_mat, max_iter=100, tol=1e-8) {
  N <- nrow(adj_mat)
  x <- rep(1/N, N)
  for (iter in seq_len(max_iter)) {
    x_new <- adj_mat %*% x
    norm_x <- sqrt(sum(x_new^2))
    if (norm_x > 0) x_new <- x_new / norm_x
    if (max(abs(x_new - x)) < tol) break
    x <- x_new
  }
  x_new / max(x_new)
}

#' Compute all centrality measures
all_centrality <- function(adj_mat) {
  N <- nrow(adj_mat)
  asset_names <- rownames(adj_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", seq_len(N))

  degree_c <- degree_centrality(adj_mat)
  eigen_c  <- eigenvector_centrality(adj_mat)
  between_c <- if (N <= 30) betweenness_centrality(adj_mat) else rep(NA, N)

  df <- data.frame(
    asset     = asset_names,
    degree    = round(degree_c, 4),
    eigenvec  = round(eigen_c, 4),
    betweenness = round(between_c, 4)
  )
  df_sorted <- df[order(-df$degree), ]
  cat("=== Network Centrality Measures ===\n")
  print(df_sorted)
  invisible(df_sorted)
}

# -----------------------------------------------------------------------------
# 8. DCC-GARCH CORRELATION SPILLOVERS (simplified)
# -----------------------------------------------------------------------------

#' Simplified DCC-inspired dynamic correlation matrix
#' Uses rolling exponential smoothing of squared returns + cross-products
#' @param returns_mat T x N matrix of returns
#' @param lambda exponential smoothing parameter (RiskMetrics-style)
dcc_rolling_correlation <- function(returns_mat, lambda = 0.94) {
  T_obs <- nrow(returns_mat)
  N     <- ncol(returns_mat)

  # Initialize with sample covariance
  Q_t  <- cov(returns_mat[1:min(60, T_obs), ])
  Q_ts <- list(Q_t)

  for (t in 2:T_obs) {
    r_t <- returns_mat[t-1, ]
    Q_t <- lambda * Q_t + (1-lambda) * outer(r_t, r_t)
    Q_ts[[t]] <- Q_t
  }

  # Convert covariances to correlations
  corr_ts <- lapply(Q_ts, function(Q) {
    d <- sqrt(diag(Q))
    Q / outer(d, d)
  })

  # Average correlation over time
  avg_corr <- Reduce("+", corr_ts) / T_obs

  # Correlation spillover: change in avg correlation during stress periods
  avg_rets <- rowMeans(returns_mat)
  stress_idx <- avg_rets < quantile(avg_rets, 0.1)

  stress_corr  <- Reduce("+", corr_ts[stress_idx]) / max(sum(stress_idx), 1)
  normal_corr  <- Reduce("+", corr_ts[!stress_idx]) / max(sum(!stress_idx), 1)
  spillover <- stress_corr - normal_corr

  cat("=== DCC Correlation Spillovers ===\n")
  cat("Average correlation increase during stress:\n")
  print(round(spillover, 3))
  cat(sprintf("Mean spillover: %.4f\n", mean(spillover[upper.tri(spillover)])))

  invisible(list(corr_ts=corr_ts, avg_corr=avg_corr,
                 stress_corr=stress_corr, normal_corr=normal_corr,
                 spillover=spillover))
}

# -----------------------------------------------------------------------------
# 9. HUB COIN ANALYSIS
# -----------------------------------------------------------------------------

#' Identify hub coins: most systemically connected cryptos
#' Combines MST degree, betweenness centrality, and CoVaR
#' @param returns_mat T x N matrix
hub_coin_analysis <- function(returns_mat, alpha = 0.05) {
  N <- ncol(returns_mat)
  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", seq_len(N))

  cat("=== Hub Coin Analysis ===\n\n")

  # 1. MST-based network
  dist_mat <- returns_to_distance(returns_mat)
  mst_res  <- minimum_spanning_tree(dist_mat)
  mst_deg  <- mst_res$degree

  # 2. Full correlation network (threshold: |rho| > 0.3)
  cor_mat  <- cor(returns_mat, use="pairwise.complete.obs")
  adj_full <- (abs(cor_mat) > 0.3) * 1; diag(adj_full) <- 0
  if (N <= 30) bet_c <- betweenness_centrality(adj_full) else bet_c <- rep(NA, N)
  eig_c <- eigenvector_centrality(adj_full)

  # 3. MES for each coin vs mean market return
  r_mkt <- rowMeans(returns_mat)
  mes_vals <- sapply(seq_len(N), function(i) mes(returns_mat[,i], r_mkt, alpha))

  # Composite hub score: z-score and average
  z_score <- function(x) { if(sd(x, na.rm=T)==0) rep(0,length(x)) else (x - mean(x,na.rm=T)) / sd(x,na.rm=T) }
  hub_score <- z_score(mst_deg) + z_score(eig_c) - z_score(mes_vals)  # MES: higher = more systemic risk (negative direction)

  df <- data.frame(
    asset = asset_names,
    mst_degree = mst_deg,
    eigenvec_c = round(eig_c, 4),
    betweenness = round(bet_c, 4),
    MES = round(mes_vals, 5),
    hub_score = round(hub_score, 3)
  )
  df <- df[order(-df$hub_score), ]
  cat("Hub Coin Rankings:\n")
  print(df)

  cat(sprintf("\nTop Hub: %s\n", df$asset[1]))
  cat(sprintf("Most Systemically Risky (highest MES): %s\n",
              asset_names[which.max(mes_vals)]))

  invisible(df)
}

# -----------------------------------------------------------------------------
# 10. FULL NETWORK ANALYSIS PIPELINE
# -----------------------------------------------------------------------------

#' Complete financial network analysis
#' @param returns_mat T x N return matrix (each column = one crypto asset)
run_network_analysis <- function(returns_mat, system_idx = 1) {
  N <- ncol(returns_mat)
  T_obs <- nrow(returns_mat)
  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", seq_len(N))

  cat("=============================================================\n")
  cat("FINANCIAL NETWORK ANALYSIS\n")
  cat(sprintf("Assets: %s\n", paste(asset_names, collapse=", ")))
  cat(sprintf("Observations: %d\n\n", T_obs))

  # Distance matrix
  dist_mat <- returns_to_distance(returns_mat)

  # MST
  cat("--- Minimum Spanning Tree ---\n")
  mst_res <- minimum_spanning_tree(dist_mat)

  # PMFG (only for small N)
  if (N <= 15) {
    cat("\n--- PMFG ---\n")
    pmfg_res <- pmfg(dist_mat)
  }

  # Community detection
  cat("\n--- Community Detection ---\n")
  mst_adj <- mst_res$adjacency
  comm_res <- greedy_modularity(mst_adj)

  # Centrality
  cat("\n--- Network Centrality ---\n")
  cent_res <- all_centrality(mst_adj)

  # Systemic risk
  cat("\n--- Systemic Risk ---\n")
  sys_risk <- systemic_risk_matrix(returns_mat, system_idx)

  # Granger network (only for small N due to computation)
  if (N <= 12 && T_obs >= 100) {
    cat("\n--- Granger Causality Network ---\n")
    gc_res <- granger_network(returns_mat, p=3)
  }

  # DCC spillovers
  cat("\n--- DCC Correlation Spillovers ---\n")
  dcc_res <- dcc_rolling_correlation(returns_mat)

  # Hub analysis
  cat("\n--- Hub Coin Analysis ---\n")
  hub_res <- hub_coin_analysis(returns_mat)

  invisible(list(mst=mst_res, communities=comm_res, centrality=cent_res,
                 systemic=sys_risk, dcc=dcc_res, hubs=hub_res))
}

# =============================================================================
# EXAMPLE
# =============================================================================
# set.seed(42)
# n <- 500; N <- 10
# asset_names <- c("BTC","ETH","BNB","SOL","ADA","AVAX","MATIC","DOT","LINK","UNI")
# # Simulate correlated returns (BTC is market factor)
# btc <- cumsum(rnorm(n, 0, 0.02))
# returns_mat <- matrix(0, n, N)
# colnames(returns_mat) <- asset_names
# for (i in 1:N) {
#   beta_i <- runif(1, 0.4, 1.2)
#   returns_mat[, i] <- diff(c(0, beta_i*btc + rnorm(n, 0, 0.01*(1+runif(1)))))
# }
# result <- run_network_analysis(returns_mat, system_idx=1)

# =============================================================================
# EXTENDED NETWORK ANALYSIS: Directed Networks, Contagion Simulation,
# Rolling Network Dynamics, Sector Clustering, and Information Flow
# =============================================================================

# -----------------------------------------------------------------------------
# Directed Partial Correlation Network: asymmetric edges using partial correlations
# Identifies which assets have unique predictive relationships with others
# after controlling for common factors -- detects information flow direction
# -----------------------------------------------------------------------------
partial_correlation_network <- function(returns_mat, threshold = 0.1) {
  p <- ncol(returns_mat)
  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", 1:p)

  cov_mat <- cov(returns_mat)
  prec_mat <- tryCatch(solve(cov_mat), error = function(e) {
    solve(cov_mat + diag(1e-6, p))
  })

  # Partial correlation: pcor_ij = -prec_ij / sqrt(prec_ii * prec_jj)
  pcor_mat <- -prec_mat / outer(sqrt(diag(prec_mat)), sqrt(diag(prec_mat)))
  diag(pcor_mat) <- 1

  # Adjacency matrix (threshold)
  adj_mat <- abs(pcor_mat) > threshold
  diag(adj_mat) <- FALSE

  # Network statistics
  degree <- rowSums(adj_mat)

  # Clustering coefficient: fraction of neighbors that are also connected
  clust_coef <- numeric(p)
  for (i in 1:p) {
    neighbors <- which(adj_mat[i, ])
    k_i <- length(neighbors)
    if (k_i < 2) { clust_coef[i] <- 0; next }
    sub_adj <- adj_mat[neighbors, neighbors, drop=FALSE]
    clust_coef[i] <- sum(sub_adj) / (k_i * (k_i - 1))
  }

  # Identify hubs: nodes with partial correlation to many others
  data.frame(
    asset = asset_names,
    degree = degree,
    clustering_coef = clust_coef,
    avg_pcor_strength = rowMeans(abs(pcor_mat)) - 1/p
  )
}

# -----------------------------------------------------------------------------
# Rolling Network Centrality: track BTC's centrality over time
# During crises, BTC centrality increases as correlations rise
# Declining centrality signals emerging altcoin independence
# -----------------------------------------------------------------------------
rolling_network_centrality <- function(returns_mat, window = 60,
                                        centrality_type = "degree") {
  n <- nrow(returns_mat); p <- ncol(returns_mat)
  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", 1:p)

  centrality_history <- matrix(NA, n, p,
                                dimnames = list(NULL, asset_names))

  for (t in window:n) {
    sub <- returns_mat[(t-window+1):t, , drop=FALSE]
    dist_mat <- returns_to_distance(sub)

    if (centrality_type == "degree") {
      # Degree centrality: number of short-distance connections (< median)
      med_d <- median(dist_mat[upper.tri(dist_mat)])
      adj <- dist_mat < med_d; diag(adj) <- FALSE
      centrality_history[t, ] <- rowSums(adj)

    } else if (centrality_type == "closeness") {
      # Closeness: 1 / sum of distances to all others
      row_sums <- rowSums(dist_mat) - diag(dist_mat)
      centrality_history[t, ] <- (p - 1) / row_sums

    } else if (centrality_type == "strength") {
      # Strength: sum of correlation weights
      cor_mat <- cor(sub)
      centrality_history[t, ] <- rowSums(abs(cor_mat)) - 1
    }
  }

  list(
    centrality_history = centrality_history,
    avg_centrality = colMeans(centrality_history, na.rm=TRUE),
    centrality_trend = apply(centrality_history, 2, function(x) {
      x <- x[!is.na(x)]
      if (length(x) < 2) return(NA)
      coef(lm(x ~ seq_along(x)))[2]
    })
  )
}

# -----------------------------------------------------------------------------
# Financial Contagion Simulation: model how shocks propagate through network
# Uses DebtRank algorithm (Battiston et al. 2012) adapted for crypto correlations
# Nodes with high DebtRank amplify systemic shocks most
# -----------------------------------------------------------------------------
debtrank_simulation <- function(returns_mat, shock_asset_idx,
                                  shock_magnitude = -0.20,
                                  n_rounds = 10) {
  p <- ncol(returns_mat)
  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", 1:p)

  # Build correlation-based influence matrix
  cor_mat <- cor(returns_mat)
  # Influence W_ij: how much a shock to j propagates to i
  # Normalize rows so total influence <= 1
  W <- pmax(cor_mat, 0)  # only positive spillovers
  diag(W) <- 0
  W <- W / (rowSums(W) + 1)  # normalize

  # Initial distress vector: 0 = healthy, 1 = defaulted
  h <- rep(0, p)
  h[shock_asset_idx] <- abs(shock_magnitude)  # initial shock

  # Track which nodes have been activated
  active <- rep(FALSE, p); active[shock_asset_idx] <- TRUE
  default_seq <- list()

  for (round in 1:n_rounds) {
    h_new <- h
    for (i in 1:p) {
      if (active[i] && h[i] < 1) {
        # Propagate distress from i to its neighbors
        for (j in 1:p) {
          if (j != i && !active[j]) {
            h_new[j] <- min(1, h_new[j] + W[j, i] * h[i])
          }
        }
      }
    }
    # Activate newly distressed nodes
    newly_active <- which(h_new > 0.1 & !active)
    if (length(newly_active) > 0) {
      active[newly_active] <- TRUE
      default_seq[[round]] <- asset_names[newly_active]
    }
    if (max(abs(h_new - h)) < 1e-6) break
    h <- h_new
  }

  # DebtRank: sum of distress propagated normalized by initial shock
  debtrank <- (sum(h) - abs(shock_magnitude)) / abs(shock_magnitude)

  data.frame(
    asset = asset_names,
    final_distress = h,
    activated = active
  ) -> distress_df

  list(
    distress = distress_df,
    debtrank = debtrank,
    shock_asset = asset_names[shock_asset_idx],
    contagion_sequence = default_seq,
    systemic_loss_pct = sum(h) / p
  )
}

# -----------------------------------------------------------------------------
# Louvain Community Detection (approximation): hierarchical modularity optimization
# Groups assets into communities based on dense intra-group correlations
# In crypto: naturally finds BTC ecosystem, DeFi cluster, Layer-1 cluster
# -----------------------------------------------------------------------------
louvain_communities <- function(returns_mat, resolution = 1.0, n_init = 10) {
  p <- ncol(returns_mat)
  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", 1:p)

  # Weighted adjacency: use absolute correlation above median
  cor_mat <- cor(returns_mat); diag(cor_mat) <- 0
  threshold <- median(abs(cor_mat[upper.tri(cor_mat)]))
  W <- pmax(abs(cor_mat) - threshold, 0)

  total_weight <- sum(W) / 2
  strength <- rowSums(W)

  # Modularity for a given partition
  modularity <- function(membership) {
    m <- max(membership); Q <- 0
    for (c in 1:m) {
      idx <- which(membership == c)
      if (length(idx) < 2) next
      L_c  <- sum(W[idx, idx]) / 2
      D_c  <- sum(strength[idx])
      Q    <- Q + L_c / total_weight - resolution * (D_c / (2*total_weight))^2
    }
    Q
  }

  # Multiple random restarts of greedy search
  best_Q <- -Inf; best_membership <- 1:p

  for (trial in 1:n_init) {
    set.seed(trial)
    membership <- sample(1:max(2, floor(p/3)), p, replace=TRUE)

    for (iter in 1:50) {
      changed <- FALSE
      for (i in sample(1:p)) {
        current_Q <- modularity(membership)
        best_local_Q <- current_Q; best_c <- membership[i]

        # Try moving i to each neighboring community
        neighbor_comms <- unique(membership[W[i, ] > 0])
        for (c in c(neighbor_comms, max(membership)+1)) {
          membership_try <- membership; membership_try[i] <- c
          Q_try <- modularity(membership_try)
          if (Q_try > best_local_Q) {
            best_local_Q <- Q_try; best_c <- c; changed <- TRUE
          }
        }
        membership[i] <- best_c
      }
      if (!changed) break
    }

    # Relabel communities 1..K
    membership <- as.integer(factor(membership))
    Q_final <- modularity(membership)
    if (Q_final > best_Q) {
      best_Q <- Q_final; best_membership <- membership
    }
  }

  communities <- split(asset_names, best_membership)
  data.frame(
    asset = asset_names,
    community = best_membership,
    community_size = tapply(best_membership, best_membership, length)[best_membership]
  ) -> comm_df

  list(
    communities = communities,
    membership = best_membership,
    modularity = best_Q,
    n_communities = max(best_membership),
    community_df = comm_df
  )
}

# Extended network example:
# pcor_net <- partial_correlation_network(returns_mat, threshold=0.15)
# roll_cen <- rolling_network_centrality(returns_mat, window=60)
# contagion <- debtrank_simulation(returns_mat, shock_asset_idx=1, shock_magnitude=-0.30)
# louvain  <- louvain_communities(returns_mat)
