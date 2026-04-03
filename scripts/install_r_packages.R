#!/usr/bin/env Rscript
# Run this once to install all required R packages
pkgs <- c("PerformanceAnalytics", "quantmod", "rugarch", "rmgarch",
          "tseries", "boot", "ggplot2", "patchwork", "gt", "jsonlite",
          "xts", "zoo", "dplyr", "tidyr")
install.packages(pkgs, repos = "https://cran.rstudio.com/", dependencies = TRUE)
cat("All packages installed.\n")
