"""
SRFMViz — Research-grade visualization for the SRFM quant lab.

All functions return Plots.jl Plot objects. Backend-agnostic (GR/Plotly/PyPlot).
"""
module SRFMViz

using Plots
using Statistics
using Distributions
using Dates
using DataFrames

export plot_equity_curve, plot_bh_heatmap, plot_returns_distribution
export plot_drawdown, plot_correlation, plot_efficient_frontier
export plot_parameter_stability, plot_factor_exposures
export plot_mc_paths, plot_seasonality
export plot_rolling_metrics, plot_trade_analysis
export plot_regime_transitions, plot_bh_mass_series

# Default theme
const SRFM_COLORS = Dict(
    :bull     => :royalblue,
    :bear     => :firebrick,
    :neutral  => :goldenrod,
    :drift    => :lightgray,
    :equity   => :steelblue,
    :bench    => :gray,
    :drawdown => :tomato,
    :mass     => :purple,
    :positive => :seagreen,
    :negative => :crimson,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Equity Curve with Regime Overlay
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_equity_curve(equity, regimes; benchmark=nothing, title="") → Plot

Full equity curve plot with:
- Coloured regime bands overlaid (BH_BULL/BH_BEAR/BH_NEUTRAL/DRIFT)
- Optional benchmark overlay
- Drawdown shading below the curve
- Annotated Sharpe, max DD, CAGR in legend
"""
function plot_equity_curve(equity::Vector{Float64},
                            regimes::Vector{String};
                            benchmark::Union{Vector{Float64}, Nothing}=nothing,
                            title::String="Equity Curve")::Plots.Plot

    n = length(equity)
    xs = 1:n

    # Compute summary stats
    rets   = diff(log.(equity))
    sharpe = isempty(rets) ? NaN :
             (mean(rets) / max(std(rets), 1e-10)) * sqrt(252)

    peak = equity[1]; max_dd = 0.0
    for e in equity
        peak = max(peak, e)
        dd   = (peak - e) / max(peak, 1e-10)
        max_dd = max(max_dd, dd)
    end
    cagr = length(equity) > 1 ?
           (equity[end] / equity[1])^(252 / max(n - 1, 1)) - 1 : NaN

    # Background regime bands
    regime_colors = Dict(
        "BH_BULL"    => RGBA(0.2, 0.4, 0.8, 0.12),
        "BH_BEAR"    => RGBA(0.8, 0.2, 0.2, 0.12),
        "BH_NEUTRAL" => RGBA(0.8, 0.7, 0.0, 0.08),
        "DRIFT"      => RGBA(0.5, 0.5, 0.5, 0.03),
    )

    p = plot(
        title      = title,
        xlabel     = "Bar",
        ylabel     = "Portfolio Value",
        legend     = :topleft,
        size       = (1200, 600),
        grid       = true,
        gridalpha  = 0.3,
        background_color = :white,
    )

    # Shade regime bands
    i = 1
    while i <= n
        reg   = i <= length(regimes) ? regimes[i] : "DRIFT"
        color = get(regime_colors, reg, RGBA(0.5, 0.5, 0.5, 0.03))
        j     = i
        while j < n && (j >= length(regimes) || regimes[j] == reg)
            j += 1
        end
        vspan!(p, [i, j]; fillcolor=color, fillalpha=0.15, linewidth=0, label="")
        i = j + 1
    end

    # Equity curve
    label_str = @sprintf("Strategy (Sharpe=%.2f, MaxDD=%.1f%%, CAGR=%.1f%%)",
                          sharpe, max_dd * 100, cagr * 100)
    plot!(p, xs, equity;
          label=label_str,
          color=SRFM_COLORS[:equity],
          linewidth=2)

    # Benchmark
    if !isnothing(benchmark)
        nb = length(benchmark)
        bench_xs = 1:nb
        plot!(p, bench_xs, benchmark;
              label="Benchmark",
              color=SRFM_COLORS[:bench],
              linewidth=1.5,
              linestyle=:dash)
    end

    # Underwater fill
    peak_series = accumulate(max, equity)
    plot!(p, xs, equity;
          fillrange=peak_series,
          fillalpha=0.15,
          fillcolor=SRFM_COLORS[:drawdown],
          linewidth=0,
          label="Drawdown")

    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. BH Mass Heatmap
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_bh_heatmap(mass_matrix, symbols, dates) → Plot

Mass heatmap: rows = symbols, cols = time, colour = BH mass.
Cold = low mass (blue), hot = active BH (red/orange).
"""
function plot_bh_heatmap(mass_matrix::Matrix{Float64},
                          symbols::Vector{String},
                          dates::Vector{Date})::Plots.Plot

    n_sym, n_bars = size(mass_matrix)
    @assert length(symbols) == n_sym
    @assert length(dates) == n_bars

    # Normalise mass per symbol for visibility
    norm_mass = copy(mass_matrix)
    for i in 1:n_sym
        row_max = maximum(norm_mass[i, :])
        if row_max > 1e-10
            norm_mass[i, :] ./= row_max
        end
    end

    # Downsample dates for labels
    date_labels = string.(dates[round.(Int, range(1, n_bars, length=min(12, n_bars)))])

    p = heatmap(
        norm_mass,
        c              = :plasma,
        xlabel         = "Date",
        ylabel         = "Symbol",
        title          = "BH Mass Heatmap (normalised per instrument)",
        yticks         = (1:n_sym, symbols),
        xticks         = (round.(Int, range(1, n_bars, length=min(12, n_bars))), date_labels),
        size           = (1400, max(300, n_sym * 40)),
        colorbar_title = "Normalised Mass",
        clims          = (0, 1),
        xrotation      = 45,
    )

    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Returns Distribution
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_returns_distribution(returns; title="") → Plot

Returns histogram with:
- Fitted normal distribution overlay (red dashed)
- Fitted Student-t overlay (blue solid)
- VaR 1% / 5% vertical lines
- Annotated skewness, excess kurtosis, JB p-value
"""
function plot_returns_distribution(returns::Vector{Float64};
                                    title::String="Returns Distribution")::Plots.Plot

    n  = length(returns)
    m  = mean(returns); s = std(returns)
    sk = mean(((returns .- m) ./ s).^3)
    ku = mean(((returns .- m) ./ s).^4) - 3

    jb   = n / 6 * (sk^2 + ku^2 / 4)
    p_jb = 1 - cdf(Chisq(2), jb)

    p = histogram(
        returns,
        normalize  = :pdf,
        bins       = min(50, max(10, round(Int, sqrt(n)))),
        fillalpha  = 0.5,
        label      = "Returns",
        color      = SRFM_COLORS[:equity],
        title      = title,
        xlabel     = "Return",
        ylabel     = "Density",
        size       = (900, 500),
        legend     = :topright,
    )

    # Normal overlay
    x_range = range(m - 4s, m + 4s, length=300)
    normal_pdf = [pdf(Normal(m, s), x) for x in x_range]
    plot!(p, collect(x_range), normal_pdf;
          label="Normal fit",
          color=:crimson,
          linewidth=2,
          linestyle=:dash)

    # Student-t overlay (quick moment-based)
    nu = max(4.0, 6.0 / max(ku, 0.01) + 4)
    sigma_t = s * sqrt((nu - 2) / nu)
    t_pdf = [pdf(LocationScale(m, sigma_t, TDist(nu)), x) for x in x_range]
    plot!(p, collect(x_range), t_pdf;
          label=@sprintf("Student-t (ν=%.1f)", nu),
          color=:steelblue,
          linewidth=2)

    # VaR lines
    sorted_r = sort(returns)
    var_1  = sorted_r[max(1, round(Int, 0.01 * n))]
    var_5  = sorted_r[max(1, round(Int, 0.05 * n))]

    vline!(p, [var_1]; label="VaR 1%",  color=:black,   linewidth=1.5, linestyle=:dot)
    vline!(p, [var_5]; label="VaR 5%",  color=:gray,    linewidth=1.5, linestyle=:dot)
    vline!(p, [0.0];   label="Zero",     color=:darkgray, linewidth=1, linestyle=:solid)

    # Annotation
    annotate!(p, m + 2s, maximum(normal_pdf) * 0.9,
              text(@sprintf("Skew=%.2f\nKurt=%.2f\nJB p=%.3f", sk, ku, p_jb),
                   :left, 9))

    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Drawdown Chart
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_drawdown(equity; title="") → Plot

Drawdown chart with:
- Underwater drawdown curve (filled red)
- Major drawdown periods highlighted
- Recovery time annotations
"""
function plot_drawdown(equity::Vector{Float64};
                        title::String="Drawdown")::Plots.Plot

    n  = length(equity)
    xs = 1:n

    # Compute drawdown series
    dd = zeros(Float64, n)
    peak = equity[1]
    for i in 2:n
        peak   = max(peak, equity[i])
        dd[i]  = (peak - equity[i]) / max(peak, 1e-10)
    end

    max_dd = maximum(dd)
    max_dd_idx = argmax(dd)

    p = plot(
        xs, -dd .* 100,
        fillrange = 0,
        fillalpha = 0.5,
        fillcolor = SRFM_COLORS[:drawdown],
        linecolor = SRFM_COLORS[:drawdown],
        linewidth = 1.5,
        label     = @sprintf("Drawdown (max=%.1f%%)", max_dd * 100),
        title     = title,
        xlabel    = "Bar",
        ylabel    = "Drawdown (%)",
        legend    = :bottomleft,
        size      = (1200, 400),
        grid      = true,
        gridalpha = 0.3,
        yformatter = y -> @sprintf("%.0f%%", y),
    )

    # Mark max drawdown
    scatter!(p, [max_dd_idx], [-dd[max_dd_idx] * 100];
             marker=:star5, markersize=10,
             color=:black, label="Max DD")

    # Horizontal threshold lines
    for threshold in [-5.0, -10.0, -20.0, -30.0]
        if minimum(-dd .* 100) < threshold
            hline!(p, [threshold];
                   linewidth=1, linestyle=:dot,
                   color=:darkgray, label="")
        end
    end

    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Correlation Matrix Heatmap
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_correlation(corr, labels) → Plot

Annotated correlation heatmap with diverging colour scale (blue-white-red).
"""
function plot_correlation(corr::Matrix{Float64},
                           labels::Vector{String})::Plots.Plot

    n = size(corr, 1)
    @assert n == length(labels)

    p = heatmap(
        corr,
        c              = :RdBu,
        clims          = (-1, 1),
        xticks         = (1:n, labels),
        yticks         = (1:n, labels),
        xrotation      = 45,
        yflip          = true,
        title          = "Correlation Matrix",
        size           = (max(600, n * 60), max(550, n * 60)),
        colorbar_title = "ρ",
        annotate_cells = n <= 15,
    )

    # Overlay correlation values
    if n <= 15
        for i in 1:n, j in 1:n
            v = corr[i, j]
            color = abs(v) > 0.5 ? :white : :black
            annotate!(p, j, i,
                      text(@sprintf("%.2f", v), color, 8))
        end
    end

    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Efficient Frontier
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_efficient_frontier(frontier, optimal) → Plot

Efficient frontier scatter plot with:
- Frontier curve coloured by Sharpe ratio
- Optimal (max Sharpe) portfolio marked
- Min-variance portfolio marked
- Individual asset risk-return points
"""
function plot_efficient_frontier(frontier::Vector{NamedTuple},
                                  optimal::NamedTuple)::Plots.Plot

    risks    = [f.risk           for f in frontier]
    rets     = [f.achieved_return for f in frontier]
    sharpes  = [f.sharpe          for f in frontier]

    p = scatter(
        risks .* 100, rets .* 100,
        marker_z   = sharpes,
        c          = :viridis,
        label      = "Frontier portfolios",
        title      = "Efficient Frontier",
        xlabel     = "Annualised Volatility (%)",
        ylabel     = "Expected Return (%)",
        size       = (900, 650),
        colorbar_title = "Sharpe",
        markersize = 5,
        markerstrokewidth = 0,
        legend     = :topleft,
    )

    # Connect frontier points
    plot!(p, risks .* 100, rets .* 100;
          linewidth=2, color=:steelblue, label="")

    # Min variance portfolio (first frontier point)
    scatter!(p, [risks[1] * 100], [rets[1] * 100];
             marker=:diamond, markersize=12,
             color=:royalblue, label="Min Variance")

    # Optimal portfolio
    scatter!(p, [optimal.risk * 100], [optimal.achieved_return * 100];
             marker=:star5, markersize=14,
             color=:gold, label=@sprintf("Optimal (Sharpe=%.2f)", optimal.sharpe))

    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Walk-Forward Parameter Stability
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_parameter_stability(wf_results, param) → Plot

Show how the best parameter value evolves across walk-forward windows.
Also shows IS vs OOS Sharpe on secondary axis.
"""
function plot_parameter_stability(wf_results::DataFrame,
                                   param::Symbol)::Plots.Plot

    n = nrow(wf_results)
    windows = 1:n

    # Extract param values from best_params dict
    param_vals = [get(wf_results[i, :best_params], string(param), NaN) for i in 1:n]

    p = plot(
        layout    = (2, 1),
        size      = (1000, 700),
        legend    = :topleft,
    )

    # Top: parameter stability
    plot!(p[1],
          windows, param_vals,
          marker    = :circle,
          linewidth = 2,
          color     = SRFM_COLORS[:equity],
          label     = string("Best ", param),
          title     = @sprintf("Parameter Stability: %s", param),
          ylabel    = string(param),
          xlabel    = "")

    # Bottom: IS vs OOS Sharpe
    is_sharpes  = wf_results[!, :is_sharpe]
    oos_sharpes = wf_results[!, :oos_sharpe]

    plot!(p[2],
          windows, is_sharpes,
          marker    = :circle,
          linewidth = 2,
          color     = SRFM_COLORS[:bull],
          label     = "IS Sharpe",
          ylabel    = "Sharpe",
          xlabel    = "Window")

    plot!(p[2],
          windows, oos_sharpes,
          marker    = :square,
          linewidth = 2,
          color     = SRFM_COLORS[:bear],
          label     = "OOS Sharpe",
          linestyle = :dash)

    hline!(p[2], [0.0]; linewidth=1, color=:black, label="")

    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. Factor Exposure Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_factor_exposures(exposures) → Plot

Horizontal bar chart of factor betas with colour coding (positive=blue, negative=red).
"""
function plot_factor_exposures(exposures::NamedTuple)::Plots.Plot

    factor_names = string.(keys(exposures))
    values       = [v for v in values(exposures)]

    colors = [v >= 0 ? SRFM_COLORS[:positive] : SRFM_COLORS[:negative] for v in values]

    n = length(values)
    p = bar(
        factor_names, values,
        orientation   = :h,
        fillcolor     = colors,
        linewidth     = 0.5,
        linecolor     = :white,
        label         = "Factor β",
        title         = "Factor Exposures",
        xlabel        = "Loading",
        ylabel        = "Factor",
        size          = (800, max(350, n * 35)),
        legend        = false,
        xflip         = false,
    )

    vline!(p, [0.0]; linewidth=1.5, color=:black, label="")

    # Annotate values
    for (i, v) in enumerate(values)
        annotate!(p, v + sign(v) * 0.01, i,
                  text(@sprintf("%.3f", v), :left, 8))
    end

    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# 9. Monte Carlo Fan Chart
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_mc_paths(paths; percentiles=[5,25,50,75,95], show_all=false) → Plot

Monte Carlo fan chart with percentile bands and median highlighted.
"""
function plot_mc_paths(paths::Matrix{Float64};
                        percentiles::Vector{Int}=[5,25,50,75,95],
                        show_all::Bool=false,
                        n_display::Int=50,
                        title::String="Monte Carlo Paths")::Plots.Plot

    n_paths, n_steps = size(paths)
    xs = 0:(n_steps - 1)

    p = plot(
        title  = title,
        xlabel = "Step",
        ylabel = "Value",
        size   = (1000, 600),
        legend = :topleft,
        grid   = true,
        gridalpha = 0.3,
    )

    # Individual paths (thin, semi-transparent)
    if show_all || n_paths <= n_display
        n_show = min(n_paths, n_display)
        for i in 1:n_show
            plot!(p, collect(xs), paths[i, :];
                  linewidth=0.5,
                  alpha=0.15,
                  color=:steelblue,
                  label=i == 1 ? "Individual paths" : "")
        end
    end

    # Percentile bands
    sorted_by_t = hcat([sort(paths[:, t]) for t in 1:n_steps]...)  # n_paths × n_steps

    band_colors = [:royalblue, :steelblue, :lightblue]
    probs       = sort(percentiles)
    mid_idx     = length(probs) ÷ 2 + 1

    for k in 1:(length(probs) ÷ 2)
        lo = probs[k]
        hi = probs[end - k + 1]
        lo_idx = max(1, round(Int, lo / 100 * n_paths))
        hi_idx = min(n_paths, round(Int, hi / 100 * n_paths))

        lo_vals = sorted_by_t[lo_idx, :]
        hi_vals = sorted_by_t[hi_idx, :]

        color = band_colors[min(k, length(band_colors))]
        plot!(p, collect(xs), hi_vals;
              fillrange=lo_vals,
              fillalpha=0.25,
              fillcolor=color,
              linewidth=0,
              label="$(lo)%-$(hi)% band")
    end

    # Median
    med_idx = round(Int, 0.5 * n_paths)
    median_path = sorted_by_t[med_idx, :]
    plot!(p, collect(xs), median_path;
          linewidth=2.5, color=:darkblue,
          label="Median (50th)")

    # Starting value reference
    hline!(p, [paths[1, 1]]; linewidth=1, color=:black,
           linestyle=:dot, label="Start")

    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# 10. Seasonality Heatmap
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_seasonality(returns, dates) → Plot

Month-by-year heatmap of average returns (green=positive, red=negative).
"""
function plot_seasonality(returns::Vector{Float64},
                           dates::Vector{Date})::Plots.Plot

    n = min(length(returns), length(dates))
    rets   = returns[1:n]
    dates  = dates[1:n]

    years  = unique(year.(dates))
    months = 1:12
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    ny = length(years)
    M  = fill(NaN, ny, 12)

    for (i, y) in enumerate(years)
        for m in months
            mask = (year.(dates) .== y) .& (month.(dates) .== m)
            if any(mask)
                M[i, m] = mean(rets[mask]) * 100
            end
        end
    end

    max_abs = maximum(abs.(filter(!isnan, M)))

    p = heatmap(
        month_names, string.(years),
        M,
        c              = :RdYlGn,
        clims          = (-max_abs, max_abs),
        title          = "Seasonality Heatmap (Monthly Returns %)",
        xlabel         = "Month",
        ylabel         = "Year",
        size           = (900, max(300, ny * 35)),
        colorbar_title = "Return %",
    )

    # Annotate cells
    for (i, y) in enumerate(years)
        for m in months
            if !isnan(M[i, m])
                color = abs(M[i, m]) > max_abs * 0.5 ? :white : :black
                annotate!(p, m, i,
                          text(@sprintf("%.1f", M[i, m]), color, 7))
            end
        end
    end

    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# 11. Rolling Performance Metrics
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_rolling_metrics(returns; window=60) → Plot

4-panel rolling metrics: Sharpe, Volatility, Skewness, Beta.
"""
function plot_rolling_metrics(returns::Vector{Float64};
                               window::Int=60,
                               benchmark::Union{Vector{Float64}, Nothing}=nothing)::Plots.Plot

    n = length(returns)
    xs = window:n

    # Rolling Sharpe
    roll_sharpe = [begin
        w = returns[i-window+1:i]
        s = std(w)
        s < 1e-10 ? 0.0 : mean(w) / s * sqrt(252)
    end for i in window:n]

    # Rolling Volatility (annualised)
    roll_vol = [std(returns[i-window+1:i]) * sqrt(252) * 100 for i in window:n]

    # Rolling Skewness
    roll_skew = [begin
        w = returns[i-window+1:i]
        m = mean(w); s = std(w)
        s < 1e-10 ? 0.0 : mean(((w .- m) ./ s).^3)
    end for i in window:n]

    # Rolling Beta (if benchmark provided)
    roll_beta = if !isnothing(benchmark)
        nb = min(length(benchmark), n)
        [begin
            i <= nb && i >= window ?
            let a = returns[i-window+1:i], b = benchmark[i-window+1:min(i,nb)]
                nb2 = min(length(a), length(b))
                var_b = var(b[1:nb2])
                var_b < 1e-12 ? 0.0 : cov(a[1:nb2], b[1:nb2]) / var_b
            end : NaN
        end for i in window:n]
    else
        fill(NaN, length(xs))
    end

    p = plot(layout=(2,2), size=(1200, 800), legend=:topleft)

    plot!(p[1], xs, roll_sharpe;
          title="Rolling Sharpe (window=$window)",
          ylabel="Sharpe", xlabel="",
          color=SRFM_COLORS[:equity], linewidth=1.5, label="Sharpe")
    hline!(p[1], [0.0]; color=:black, linewidth=1, label="")

    plot!(p[2], xs, roll_vol;
          title="Rolling Volatility",
          ylabel="Ann. Vol (%)", xlabel="",
          color=SRFM_COLORS[:neutral], linewidth=1.5, label="Vol %")

    plot!(p[3], xs, roll_skew;
          title="Rolling Skewness",
          ylabel="Skewness", xlabel="Bar",
          color=SRFM_COLORS[:bear], linewidth=1.5, label="Skew")
    hline!(p[3], [0.0]; color=:black, linewidth=1, label="")

    if !all(isnan, roll_beta)
        plot!(p[4], xs, roll_beta;
              title="Rolling Beta",
              ylabel="Beta", xlabel="Bar",
              color=SRFM_COLORS[:bull], linewidth=1.5, label="Beta")
        hline!(p[4], [1.0]; color=:black, linewidth=1, linestyle=:dash, label="β=1")
    else
        plot!(p[4]; title="Beta (no benchmark)", xlabel="Bar")
    end

    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# 12. Trade Analysis
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_trade_analysis(trades_df) → Plot

4-panel trade analysis:
  1. PnL distribution
  2. MFE vs MAE scatter
  3. Duration distribution
  4. Cumulative PnL by regime
"""
function plot_trade_analysis(trades_df::DataFrame)::Plots.Plot

    p = plot(layout=(2,2), size=(1200, 800), legend=:topright)

    # 1. PnL Distribution
    if hasproperty(trades_df, :pnl)
        pnls = trades_df[!, :pnl] .* 100
        histogram!(p[1], pnls;
                   bins=30, normalize=:pdf,
                   fillalpha=0.6,
                   color=SRFM_COLORS[:equity],
                   label="PnL distribution",
                   title="Trade PnL (%)",
                   xlabel="PnL %", ylabel="Density")
        vline!(p[1], [0.0]; color=:black, linewidth=2, label="Break-even")
        vline!(p[1], [mean(pnls)];
               color=:red, linewidth=1.5, linestyle=:dash,
               label=@sprintf("Mean=%.2f%%", mean(pnls)))
    end

    # 2. MFE vs MAE
    if hasproperty(trades_df, :mfe) && hasproperty(trades_df, :mae)
        mfe = trades_df[!, :mfe] .* 100
        mae = trades_df[!, :mae] .* 100
        winners = hasproperty(trades_df, :pnl) ? trades_df[!, :pnl] .> 0 : fill(true, nrow(trades_df))

        scatter!(p[2], mae[.!winners], mfe[.!winners];
                 color=SRFM_COLORS[:bear], alpha=0.6,
                 label="Losers", markersize=4, markerstrokewidth=0)
        scatter!(p[2], mae[winners], mfe[winners];
                 color=SRFM_COLORS[:bull], alpha=0.6,
                 label="Winners", markersize=4, markerstrokewidth=0,
                 title="MFE vs MAE",
                 xlabel="MAE %", ylabel="MFE %")
    end

    # 3. Duration Distribution
    if hasproperty(trades_df, :duration)
        histogram!(p[3], trades_df[!, :duration];
                   bins=30, fillalpha=0.6,
                   color=SRFM_COLORS[:neutral],
                   label="Duration",
                   title="Trade Duration (bars)",
                   xlabel="Bars", ylabel="Count")
    end

    # 4. Cumulative PnL by Regime
    if hasproperty(trades_df, :pnl) && hasproperty(trades_df, :regime)
        regimes = unique(trades_df[!, :regime])
        for (k, reg) in enumerate(regimes)
            mask     = trades_df[!, :regime] .== reg
            reg_pnls = cumsum(trades_df[mask, :pnl] .* 100)
            color_k  = get(SRFM_COLORS, Symbol(lowercase(split(reg, "_")[end])), :steelblue)
            plot!(p[4], 1:length(reg_pnls), reg_pnls;
                  label=reg,
                  linewidth=2,
                  title="Cumulative PnL by Regime",
                  xlabel="Trade #", ylabel="Cumulative PnL %")
        end
        hline!(p[4], [0.0]; color=:black, linewidth=1, label="")
    end

    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# 13. BH Mass Series
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_bh_mass_series(prices, masses, active, bh_dir; title="") → Plot

3-panel plot:
  Top:    Price with BH active periods shaded
  Middle: BH mass with activation threshold line
  Bottom: Normalised beta (bar velocity)
"""
function plot_bh_mass_series(prices::Vector{Float64},
                              masses::Vector{Float64},
                              active::Vector{Bool},
                              bh_dir::Vector{Int};
                              cf::Float64=0.003,
                              bh_form::Float64=0.25,
                              title::String="BH Mass Dynamics")::Plots.Plot

    n  = length(prices)
    xs = 1:n

    betas = vcat(0.0, abs.(diff(log.(prices))))

    p = plot(layout=(3,1), size=(1200, 900), link=:x)

    # Top: Price
    plot!(p[1], xs, prices;
          title=title, ylabel="Price",
          linewidth=1.5, color=:darkblue,
          label="Price", legend=:topleft)

    # Shade active periods
    for i in 2:n
        if active[i] && !active[i-1]
            j = i
            while j < n && active[j]; j += 1; end
            color = bh_dir[i] == 1 ? RGBA(0.2, 0.4, 0.8, 0.2) :
                    bh_dir[i] == -1 ? RGBA(0.8, 0.2, 0.2, 0.2) :
                                      RGBA(0.8, 0.7, 0.0, 0.15)
            vspan!(p[1], [i, j]; fillcolor=color, linewidth=0, label="")
        end
    end

    # Middle: Mass
    plot!(p[2], xs, masses;
          ylabel="BH Mass",
          linewidth=1.5, color=SRFM_COLORS[:mass],
          label="Mass", fill=0, fillalpha=0.3)
    hline!(p[2], [bh_form];
           linewidth=1.5, linestyle=:dash,
           color=:red, label=@sprintf("Form threshold (%.3f)", bh_form))

    # Bottom: Beta
    bar!(p[3], xs, betas;
         ylabel="β", xlabel="Bar",
         fillcolor=ifelse.(betas .< cf, SRFM_COLORS[:bull], SRFM_COLORS[:bear]),
         linewidth=0, label="β")
    hline!(p[3], [cf];
           linewidth=1.5, color=:black, linestyle=:dash,
           label=@sprintf("CF (%.4f)", cf))

    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# 14. Regime Transitions
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_regime_transitions(regimes) → Plot

Stacked area chart showing fraction of time in each regime over rolling window.
"""
function plot_regime_transitions(regimes::Vector{String};
                                  window::Int=50)::Plots.Plot

    n = length(regimes)
    all_regimes = ["BH_BULL", "BH_BEAR", "BH_NEUTRAL", "DRIFT"]
    colors_reg  = [SRFM_COLORS[:bull], SRFM_COLORS[:bear],
                   SRFM_COLORS[:neutral], SRFM_COLORS[:drift]]

    xs = window:n
    fractions = Dict{String, Vector{Float64}}()

    for reg in all_regimes
        fractions[reg] = [mean(regimes[i-window+1:i] .== reg) for i in window:n]
    end

    p = plot(
        title  = "Regime Fractions (rolling window=$window)",
        xlabel = "Bar",
        ylabel = "Fraction",
        ylims  = (0, 1),
        size   = (1100, 500),
        legend = :topright,
    )

    cumul = zeros(length(xs))
    for (k, reg) in enumerate(all_regimes)
        frac = fractions[reg]
        plot!(p, collect(xs), cumul .+ frac;
              fillrange=cumul,
              fillalpha=0.6,
              fillcolor=colors_reg[k],
              linewidth=0,
              label=reg)
        cumul .+= frac
    end

    return p
end

end # module SRFMViz
