module AlternativeData

# ============================================================
# alternative_data.jl -- Alternative Data Signal Processing
# ============================================================
# Covers: options flow analysis, dark pool prints, unusual volume
# detection, satellite proxies, web traffic signals, NLP sentiment,
# credit card spend proxies, geolocation foot-traffic, shipping
# signals, and composite alpha construction.
# Pure stdlib Julia -- no external packages.
# ============================================================

using Statistics, LinearAlgebra

struct OptionsFlowRecord
    timestamp::Float64
    ticker::String
    strike::Float64
    expiry_days::Int
    call_or_put::Symbol
    volume::Int
    open_interest::Int
    implied_vol::Float64
    premium::Float64
    spot_price::Float64
    is_sweep::Bool
end

struct DarkPoolPrint
    timestamp::Float64
    ticker::String
    sz::Int
    price::Float64
    exchange::String
    lit_bid::Float64
    lit_ask::Float64
end

struct VolumeAnomaly
    ticker::String
    date_idx::Int
    actual_volume::Float64
    expected_volume::Float64
    z_score::Float64
    is_unusual::Bool
    direction::Symbol
end

struct SatelliteSignal
    asset::String
    signal_date::Float64
    metric::Symbol
    value::Float64
    baseline_avg::Float64
    baseline_std::Float64
    z_score::Float64
end

struct WebTrafficSignal
    company::String
    date::Float64
    visits::Float64
    unique_visitors::Float64
    avg_session_sec::Float64
    bounce_rate::Float64
    wow_pct::Float64
end

struct SentimentRecord
    timestamp::Float64
    source::Symbol
    score::Float64
    volume::Int
    ticker::String
end

struct CreditCardProxy
    ticker::String
    period::Int
    spend_yoy_pct::Float64
    txn_yoy_pct::Float64
    avg_ticket_yoy_pct::Float64
    panel_size::Int
end

struct ShippingSignal
    date::Float64
    route::String
    rate_usd::Float64
    utilisation_pct::Float64
    congestion_days::Float64
    wow_pct::Float64
end

# ---- 1. Options Flow ----

function options_flow_score(records::Vector{OptionsFlowRecord})
    bull = 0.0; bear = 0.0; sb = 0.0; ss = 0.0; nc = 0; np = 0
    for r in records
        n = r.premium * r.volume * 100.0
        if r.call_or_put == :call
            bull += n; nc += r.volume
            if r.is_sweep; sb += n; end
        else
            bear += n; np += r.volume
            if r.is_sweep; ss += n; end
        end
    end
    total = bull + bear + 1e-8
    score = (bull - bear) / total
    pcr   = (np + 1e-8) / (nc + 1e-8)
    sweep = (sb - ss) / (sb + ss + 1e-8)
    sig   = score > 0.1 ? :bullish : score < -0.1 ? :bearish : :neutral
    return (net_score=score, put_call_ratio=pcr, sweep_score=sweep,
            bull_notional=bull, bear_notional=bear, signal=sig)
end

function unusual_options_activity(records::Vector{OptionsFlowRecord},
                                   mult::Float64=3.0)::Vector{OptionsFlowRecord}
    return filter(r -> r.volume > mult * max(r.open_interest, 1), records)
end

function iv_rank(cur::Float64, hist::Vector{Float64})::Float64
    lo = minimum(hist); hi = maximum(hist)
    return (cur - lo) / (hi - lo + 1e-12) * 100.0
end

function iv_percentile(cur::Float64, hist::Vector{Float64})::Float64
    return count(v -> v < cur, hist) / length(hist) * 100.0
end

function gamma_exposure(records::Vector{OptionsFlowRecord}, spot::Float64)::Float64
    gex = 0.0
    for r in records
        m = r.strike / spot
        d = exp(-0.5 * (m - 1.0)^2 / (r.implied_vol^2 + 1e-8))
        gam = d / (spot * r.implied_vol * sqrt(r.expiry_days / 365.0 + 1e-6))
        sg  = r.call_or_put == :call ? 1.0 : -1.0
        gex += sg * gam * r.open_interest * 100.0
    end
    return gex
end

function iv_term_slope(niv::Float64, fiv::Float64, nd::Int, fd::Int)::Float64
    return (fiv - niv) / (fd - nd + 1e-8) * 365.0
end

function options_skew(otm_put_iv::Float64, atm_iv::Float64, otm_call_iv::Float64)
    put_skew  = otm_put_iv  - atm_iv
    call_skew = otm_call_iv - atm_iv
    risk_reversal = otm_call_iv - otm_put_iv
    butterfly     = (otm_put_iv + otm_call_iv) / 2.0 - atm_iv
    return (put_skew=put_skew, call_skew=call_skew,
            risk_reversal=risk_reversal, butterfly=butterfly)
end

# ---- 2. Dark Pool ----

function dp_classify(p::DarkPoolPrint)::Symbol
    mid = (p.lit_bid + p.lit_ask) / 2.0
    if p.price > p.lit_ask; return :above_offer
    elseif p.price < p.lit_bid; return :below_bid
    elseif p.price > mid; return :above_mid
    elseif p.price < mid; return :below_mid
    else; return :at_mid
    end
end

function dp_flow_imbalance(prints::Vector{DarkPoolPrint})::Float64
    buy = 0.0; sell = 0.0
    for p in prints
        cls = dp_classify(p)
        if cls in (:above_offer, :above_mid); buy  += p.sz * p.price
        else;                                 sell += p.sz * p.price
        end
    end
    return (buy - sell) / (buy + sell + 1e-8)
end

function dp_block_trades(prints::Vector{DarkPoolPrint}, min_sz::Int=10000)::Vector{DarkPoolPrint}
    return filter(p -> p.sz >= min_sz, prints)
end

function off_exchange_pct(total_vol::Float64, dp_vol::Float64)::Float64
    return dp_vol / (total_vol + 1e-8) * 100.0
end

function dp_price_discovery(prints::Vector{DarkPoolPrint}, mids::Vector{Float64})::Float64
    isempty(prints) && return 0.0
    n = min(length(prints), length(mids))
    x = [p.price for p in prints[1:n]]; y = mids[1:n]
    xb = mean(x); yb = mean(y)
    cov = sum((x .- xb) .* (y .- yb)) / (n - 1 + 1e-8)
    return cov / (std(x) * std(y) + 1e-8)
end

# ---- 3. Volume Anomaly ----

function volume_z_score(vols::Vector{Float64}, window::Int)::Vector{Float64}
    n = length(vols); z = fill(NaN, n)
    for i in (window+1):n
        h = vols[i-window:i-1]
        z[i] = (vols[i] - mean(h)) / (std(h) + 1e-8)
    end
    return z
end

function detect_unusual_volume(ticker::String, vols::Vector{Float64},
                                prices::Vector{Float64}, window::Int=20,
                                thresh::Float64=2.5)::Vector{VolumeAnomaly}
    z = volume_z_score(vols, window); n = length(vols); out = VolumeAnomaly[]
    for i in (window+1):n
        ev = mean(vols[max(1,i-window):i-1])
        dir = i > 1 && prices[i] > prices[i-1] ? :up :
              i > 1 && prices[i] < prices[i-1] ? :down : :flat
        push!(out, VolumeAnomaly(ticker, i, vols[i], ev, z[i], abs(z[i]) > thresh, dir))
    end
    return filter(r -> r.is_unusual, out)
end

function on_balance_volume(vols::Vector{Float64}, closes::Vector{Float64})::Vector{Float64}
    n = length(closes); obv = zeros(n)
    for i in 2:n
        if closes[i] > closes[i-1]; obv[i] = obv[i-1] + vols[i]
        elseif closes[i] < closes[i-1]; obv[i] = obv[i-1] - vols[i]
        else; obv[i] = obv[i-1]
        end
    end
    return obv
end

function volume_price_trend(vols::Vector{Float64}, prices::Vector{Float64})::Vector{Float64}
    n = length(prices); vpt = zeros(n)
    for i in 2:n
        vpt[i] = vpt[i-1] + vols[i] * (prices[i] - prices[i-1]) / (prices[i-1] + 1e-8)
    end
    return vpt
end

function amihud_illiquidity(rets::Vector{Float64}, dvols::Vector{Float64})::Float64
    return mean(abs.(rets) ./ (dvols .+ 1e-8)) * 252.0 * 1e6
end

# ---- 4. Satellite Signals ----

function parking_lot_proxy(car_count::Float64, avg_spend::Float64)::Float64
    return car_count * avg_spend
end

function satellite_aggregate(sigs::Vector{SatelliteSignal})
    isempty(sigs) && return (avg_z=0.0, trend=:flat, strength=0.0)
    zs = [s.z_score for s in sigs]; avgz = mean(zs)
    trend = avgz > 1 ? :bullish : avgz < -1 ? :bearish : :flat
    return (avg_z=avgz, trend=trend, strength=clamp(abs(avgz)/3.0, 0.0, 1.0))
end

function crop_yield_signal(ndvi::Vector{Float64}, hist_avg::Float64)
    cur = mean(ndvi); dev = (cur - hist_avg) / (hist_avg + 1e-8) * 100.0
    sig = dev > 5 ? :above_avg : dev < -5 ? :below_avg : :normal
    return (ndvi_avg=cur, pct_vs_hist=dev, signal=sig)
end

function vessel_count_signal(counts::Vector{Float64}, baseline::Float64, baseline_std::Float64)
    z = (mean(counts) - baseline) / (baseline_std + 1e-8)
    return (z_score=z, signal=z > 2 ? :high_traffic : z < -2 ? :low_traffic : :normal)
end

# ---- 5. Web Traffic ----

function web_traffic_momentum(recs::Vector{WebTrafficSignal}, window::Int=4)::Vector{Float64}
    n = length(recs); visits = [r.unique_visitors for r in recs]; mom = fill(NaN, n)
    for i in (window+1):n
        mom[i] = (visits[i] - visits[i-window]) / (visits[i-window] + 1e-8) * 100.0
    end
    return mom
end

function web_engagement(recs::Vector{WebTrafficSignal})::Vector{Float64}
    return [(1 - r.bounce_rate) * r.avg_session_sec / 60.0 for r in recs]
end

function app_alpha(dl::Vector{Float64}, sector_dl::Vector{Float64})::Float64
    co = length(dl) >= 4 ? (dl[end] - dl[end-3]) / (dl[end-3] + 1e-8) : 0.0
    sc = length(sector_dl) >= 4 ? (sector_dl[end] - sector_dl[end-3]) / (sector_dl[end-3] + 1e-8) : 0.0
    return co - sc
end

# ---- 6. Sentiment ----

function sentiment_aggregate(recs::Vector{SentimentRecord}, halflife::Float64=86400.0)::Float64
    isempty(recs) && return 0.0
    t_max = maximum(r.timestamp for r in recs); ws = 0.0; wt = 0.0
    for r in recs
        w = exp(-log(2) * (t_max - r.timestamp) / halflife) * r.volume
        ws += w * r.score; wt += w
    end
    return wt > 0 ? ws / wt : 0.0
end

function sentiment_dispersion(recs::Vector{SentimentRecord})::Float64
    scores = [r.score for r in recs]
    return isempty(scores) ? 0.0 : std(scores)
end

function news_spike_detect(counts::Vector{Int}, window::Int=5, thresh::Float64=3.0)::Vector{Bool}
    n = length(counts); spikes = fill(false, n)
    for i in (window+1):n
        h = Float64.(counts[i-window:i-1])
        spikes[i] = (counts[i] - mean(h)) / (std(h) + 1e-8) > thresh
    end
    return spikes
end

function rule_based_sentiment(pos::Int, neg::Int, intens::Int, neg_count::Int)::Float64
    raw = Float64(pos - neg)
    boosted = raw * (1.0 + 0.293 * intens)
    if neg_count % 2 == 1; boosted = -boosted; end
    return clamp(boosted / (4.0 * (pos + neg + 1)), -1.0, 1.0)
end

# ---- 7. Credit Card Spend ----

function cc_revenue_surprise(proxy::CreditCardProxy, consensus::Float64, std_dev::Float64)
    surprise = proxy.spend_yoy_pct - consensus; z = surprise / (std_dev + 1e-8)
    return (spend_yoy=proxy.spend_yoy_pct, surprise_pct=surprise, surprise_z=z,
            direction=surprise > 0 ? :beat : :miss)
end

function spend_velocity(proxies::Vector{CreditCardProxy})::Vector{Float64}
    n = length(proxies); if n < 2; return zeros(n); end
    spends = [p.spend_yoy_pct for p in proxies]
    return [i == 1 ? 0.0 : spends[i] - spends[i-1] for i in 1:n]
end

# ---- 8. Geolocation ----

function foot_traffic_index(counts::Vector{Float64}, baseline::Float64)::Vector{Float64}
    return counts ./ baseline .* 100.0
end

function trade_area_overlap(lat1::Float64, lon1::Float64, r1::Float64,
                             lat2::Float64, lon2::Float64, r2::Float64)::Float64
    dlat = deg2rad(lat2 - lat1); dlon = deg2rad(lon2 - lon1)
    a = sin(dlat/2)^2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon/2)^2
    d = 2 * 6371.0 * asin(sqrt(a))
    if d >= r1 + r2; return 0.0; end
    if d <= abs(r1 - r2); return 1.0; end
    a1 = r1^2 * acos(clamp((d^2+r1^2-r2^2)/(2*d*r1+1e-12), -1.0, 1.0))
    a2 = r2^2 * acos(clamp((d^2+r2^2-r1^2)/(2*d*r2+1e-12), -1.0, 1.0))
    a3 = 0.5 * sqrt(max(0.0, (-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2)))
    inter = a1 + a2 - a3
    return inter / (pi*r1^2 + pi*r2^2 - inter + 1e-8)
end

# ---- 9. Shipping ----

function bdi_momentum(rates::Vector{ShippingSignal}, window::Int=4)
    n = length(rates)
    if n < window + 1; return (momentum=NaN, z_score=NaN, signal=:insufficient_data); end
    rs = [r.rate_usd for r in rates]; base = rs[1:end-1]
    bmean = mean(base); bstd = std(base) + 1e-8; cur = rs[end]
    mom = (cur - bmean) / bmean * 100.0; z = (cur - bmean) / bstd
    sig = z > 1.5 ? :tightening : z < -1.5 ? :loosening : :stable
    return (momentum=mom, z_score=z, signal=sig, current=cur, base_avg=bmean)
end

function port_congestion_cost(actual::Float64, baseline::Float64, daily_rate::Float64)::Float64
    return max(0.0, actual - baseline) * daily_rate
end

# ---- 10. Signal Combination ----

function alt_data_composite(signals::Dict{Symbol,Float64},
                              weights::Dict{Symbol,Float64})::Float64
    s = 0.0; wt = 0.0
    for (k, v) in signals
        w = get(weights, k, 0.0); s += w * v; wt += abs(w)
    end
    return wt > 0 ? s / wt : 0.0
end

function signal_halflife(ic_series::Vector{Float64})::Float64
    n = length(ic_series); if n < 2; return Inf; end
    y = ic_series[2:end]; x = ic_series[1:end-1]
    xb = mean(x); yb = mean(y)
    rho = clamp(sum((x.-xb).*(y.-yb)) / (sum((x.-xb).^2)+1e-12), 0.001, 0.9999)
    return -log(2) / log(rho)
end

function cross_asset_ic(signal::Vector{Float64}, fwd_rets::Vector{Float64},
                         window::Int=60)::Float64
    n = min(length(signal), length(fwd_rets), window)
    x = signal[end-n+1:end]; y = fwd_rets[end-n+1:end]
    xb = mean(x); yb = mean(y)
    cov_val = sum((x.-xb).*(y.-yb)) / (n-1+1e-8)
    return cov_val / (std(x)*std(y)+1e-8)
end

# ---- 11. Event-Driven ----

function earnings_surprise(consensus::Float64, whisper::Float64, actual::Float64)
    vc = (actual-consensus) / (abs(consensus)+1e-8) * 100.0
    vw = (actual-whisper)   / (abs(whisper)+1e-8)   * 100.0
    qual = vc>0 && vw>0 ? :clean_beat : vc>0 ? :consensus_beat : vc<0 ? :miss : :in_line
    return (vs_consensus=vc, vs_whisper=vw, quality=qual)
end

function insider_flow(net_buys::Float64, net_sells::Float64, shares_out::Float64)
    net = net_buys - net_sells; pct = net / (shares_out+1e-8) * 100.0
    sig = pct > 0.1 ? :accumulation : pct < -0.1 ? :distribution : :neutral
    return (net_shares=net, pct_float=pct, signal=sig)
end

function short_interest_signal(short_sh::Float64, float_sh::Float64, adv::Float64)
    si = short_sh / (float_sh+1e-8) * 100.0; dtc = short_sh / (adv+1e-8)
    sent = si > 20 ? :heavily_shorted : si > 10 ? :moderate : :low
    return (short_pct=si, days_to_cover=dtc, sentiment=sent)
end

function sec_filing_lag(filing_date::Float64, earnings_date::Float64)::Float64
    return filing_date - earnings_date
end

# ---- Demo ----

function demo()
    println("=== AlternativeData Demo ===")
    flows = [
        OptionsFlowRecord(1.0,"SPY",420.0,30,:call,5000,2000,0.20,3.5,415.0,true),
        OptionsFlowRecord(2.0,"SPY",410.0,30,:put,1000,500,0.22,2.0,415.0,false),
        OptionsFlowRecord(3.0,"SPY",425.0,7,:call,8000,1000,0.18,1.2,415.0,true),
    ]
    sc = options_flow_score(flows)
    println("Options flow: score=", round(sc.net_score,digits=4), " signal=", sc.signal)
    println("P/C ratio: ", round(sc.put_call_ratio,digits=3))

    prints = [
        DarkPoolPrint(1.0,"AAPL",50000,182.5,"FINRA",182.3,182.7),
        DarkPoolPrint(2.0,"AAPL",20000,182.1,"FINRA",182.3,182.7),
        DarkPoolPrint(3.0,"AAPL",100000,183.0,"FINRA",182.3,182.7),
    ]
    println("DP imbalance: ", round(dp_flow_imbalance(prints),digits=4))
    println("Block trades: ", length(dp_block_trades(prints,50000)))

    vols = [1.0e6+0.1e6*sin(Float64(i)) for i in 1:20]; push!(vols, 5.0e6)
    prices = cumsum([0.01*sin(Float64(i)) for i in 1:length(vols)]) .+ 100.0
    anoms = detect_unusual_volume("XYZ", vols, prices, 15, 2.5)
    println("Unusual volume events: ", length(anoms))

    sents = [SentimentRecord(1000.0,:twitter,0.3,500,"TSLA"),
             SentimentRecord(2000.0,:news,0.7,200,"TSLA"),
             SentimentRecord(3000.0,:reddit,-0.1,800,"TSLA")]
    println("Sentiment: ", round(sentiment_aggregate(sents,3600.0),digits=4))
    println("Dispersion: ", round(sentiment_dispersion(sents),digits=4))

    ships = [ShippingSignal(Float64(i),"ASIA-EU",1500.0+50*i,85.0,2.5,3.0) for i in 1:8]
    bdi = bdi_momentum(ships)
    println("BDI momentum: ", round(bdi.momentum,digits=2), "% | ", bdi.signal)

    sk = options_skew(0.25, 0.20, 0.18)
    println("25d RR: ", round(sk.risk_reversal*100,digits=2), "% | BF: ", round(sk.butterfly*100,digits=2), "%")

    si = short_interest_signal(5e6,50e6,2e6)
    println("Short interest: ", round(si.short_pct,digits=2), "% DTC=", round(si.days_to_cover,digits=2))
end

# ---- Additional Alternative Data Functions ----

function options_open_interest_trend(oi_series::Vector{Int}, window::Int=5)::Symbol
    if length(oi_series) < window + 1; return :insufficient_data; end
    recent = oi_series[end-window+1:end]; older = oi_series[end-2*window+1:end-window]
    pct_chg = (mean(recent) - mean(older)) / (mean(older) + 1e-8) * 100.0
    return pct_chg > 10 ? :building : pct_chg < -10 ? :unwinding : :stable
end

function dark_pool_session_summary(prints::Vector{DarkPoolPrint})
    if isempty(prints)
        return (total_sz=0, buy_vol=0.0, sell_vol=0.0, block_count=0, imbalance=0.0)
    end
    total_sz = sum(p.sz for p in prints)
    imbal = dp_flow_imbalance(prints)
    blocks = length(dp_block_trades(prints, 10000))
    buy_vol = sum(p.sz * p.price for p in prints if dp_classify(p) in (:above_offer, :above_mid))
    sell_vol = sum(p.sz * p.price for p in prints if !(dp_classify(p) in (:above_offer, :above_mid)))
    return (total_sz=total_sz, buy_vol=buy_vol, sell_vol=sell_vol, block_count=blocks, imbalance=imbal)
end

function satellite_coverage_ratio(observed_sites::Int, total_sites::Int)::Float64
    return observed_sites / (total_sites + 1e-8) * 100.0
end

function web_retention_rate(new_visitors::Float64, returning_visitors::Float64)::Float64
    total = new_visitors + returning_visitors + 1e-8
    return returning_visitors / total * 100.0
end

function social_media_viral_coefficient(shares_per_post::Float64,
                                         conversion_rate::Float64)::Float64
    return shares_per_post * conversion_rate
end

function alternative_data_freshness_score(data_lag_days::Float64,
                                            max_acceptable_lag::Float64=5.0)::Float64
    return clamp(1.0 - data_lag_days / max_acceptable_lag, 0.0, 1.0)
end

function cross_validate_alt_signal(signal::Vector{Float64},
                                    returns::Vector{Float64},
                                    n_folds::Int=5)::Vector{Float64}
    n = length(signal); fold_size = n div n_folds; ics = Float64[]
    for fold in 1:n_folds
        val_start = (fold-1)*fold_size + 1; val_end = min(fold*fold_size, n)
        val_s = signal[val_start:val_end]; val_r = returns[val_start:val_end]
        if length(val_s) > 2
            push!(ics, cross_asset_ic(val_s, val_r, length(val_s)))
        end
    end
    return ics
end

function geospatial_cluster_signal(lats::Vector{Float64}, lons::Vector{Float64},
                                    values::Vector{Float64}, radius_km::Float64=5.0)::Vector{Float64}
    n = length(lats); cluster_scores = zeros(n)
    for i in 1:n
        count = 0; total = 0.0
        for j in 1:n
            dlat = deg2rad(lats[j] - lats[i])
            dlon = deg2rad(lons[j] - lons[i])
            a = sin(dlat/2)^2 + cos(deg2rad(lats[i]))*cos(deg2rad(lats[j]))*sin(dlon/2)^2
            d = 2*6371.0*asin(sqrt(a))
            if d <= radius_km; count += 1; total += values[j]; end
        end
        cluster_scores[i] = count > 0 ? total/count : 0.0
    end
    return cluster_scores
end

function esg_controversy_signal(controversy_score::Float64, industry_avg::Float64,
                                  industry_std::Float64)::NamedTuple
    z = (controversy_score - industry_avg) / (industry_std + 1e-8)
    risk = z > 2 ? :high_risk : z > 1 ? :elevated : z < -1 ? :below_avg : :normal
    return (z_score=z, relative_risk=risk, percentile=0.5*(1+erf(z/sqrt(2)))*100)
end

function patent_filing_signal(filings_12m::Int, filings_prior_12m::Int,
                                sector_avg_growth::Float64)::NamedTuple
    growth = (filings_12m - filings_prior_12m) / (filings_prior_12m + 1e-8) * 100.0
    excess = growth - sector_avg_growth
    signal = excess > 20 ? :strong_innovation : excess > 5 ? :above_avg :
             excess < -20 ? :declining : :stable
    return (yoy_growth=growth, excess_growth=excess, signal=signal)
end

function job_posting_signal(current_postings::Int, prior_postings::Int,
                              sector_growth::Float64)::NamedTuple
    growth = (current_postings - prior_postings) / (prior_postings + 1e-8) * 100.0
    alpha = growth - sector_growth
    return (posting_growth=growth, alpha_vs_sector=alpha,
            signal=alpha > 10 ? :hiring_acceleration : alpha < -10 ? :hiring_slowdown : :neutral)
end

function supply_chain_disruption_score(delivery_delays::Float64,
                                        supplier_diversity::Int,
                                        inventory_days::Float64)::Float64
    delay_score = clamp(delivery_delays / 30.0, 0.0, 1.0) * 40.0
    diversity_score = clamp(1.0 - supplier_diversity / 10.0, 0.0, 1.0) * 30.0
    inventory_score = clamp(1.0 - inventory_days / 90.0, 0.0, 1.0) * 30.0
    return delay_score + diversity_score + inventory_score
end

function macro_nowcast_signal(coincident_indicators::Vector{Float64},
                               weights::Vector{Float64})::Float64
    z_scores = cross_section_zscore(coincident_indicators)
    w = abs.(weights) ./ (sum(abs.(weights)) + 1e-8)
    return dot(z_scores, w)
end

function cross_section_zscore(x::Vector{Float64})::Vector{Float64}
    mu = mean(x); sig = std(x) + 1e-8
    return clamp.((x .- mu) ./ sig, -3.0, 3.0)
end


# ---- Alternative Data Utilities (continued) ----

function google_trends_momentum(trends::Vector{Float64}, window::Int=4)::Float64
    n = length(trends); if n < window + 1; return 0.0; end
    recent = trends[end-window+1:end]; prior = trends[end-2*window+1:end-window]
    return (mean(recent) - mean(prior)) / (mean(prior) + 1e-8) * 100.0
end

function email_receipt_signal(email_counts::Vector{Int}, baseline::Float64)::Float64
    return (mean(Float64.(email_counts)) - baseline) / (baseline + 1e-8) * 100.0
end

function second_party_data_quality(coverage::Float64, freshness_days::Float64,
                                    accuracy_estimate::Float64)::Float64
    cov_score = clamp(coverage, 0.0, 1.0)
    fresh_score = clamp(1.0 - freshness_days/30.0, 0.0, 1.0)
    acc_score = clamp(accuracy_estimate, 0.0, 1.0)
    return (cov_score + fresh_score + acc_score) / 3.0 * 100.0
end


# ---- Alternative Data Utilities (continued) ----

function google_trends_momentum(trends::Vector{Float64}, window::Int=4)::Float64
    n = length(trends); if n < window + 1; return 0.0; end
    recent = trends[end-window+1:end]; prior = trends[end-2*window+1:end-window]
    return (mean(recent) - mean(prior)) / (mean(prior) + 1e-8) * 100.0
end

function email_receipt_signal(email_counts::Vector{Int}, baseline::Float64)::Float64
    return (mean(Float64.(email_counts)) - baseline) / (baseline + 1e-8) * 100.0
end

function second_party_data_quality(coverage::Float64, freshness_days::Float64,
                                    accuracy_estimate::Float64)::Float64
    cov_score = clamp(coverage, 0.0, 1.0)
    fresh_score = clamp(1.0 - freshness_days/30.0, 0.0, 1.0)
    acc_score = clamp(accuracy_estimate, 0.0, 1.0)
    return (cov_score + fresh_score + acc_score) / 3.0 * 100.0
end


# ---- Alternative Data Utilities (continued) ----

function google_trends_momentum(trends::Vector{Float64}, window::Int=4)::Float64
    n = length(trends); if n < window + 1; return 0.0; end
    recent = trends[end-window+1:end]; prior = trends[end-2*window+1:end-window]
    return (mean(recent) - mean(prior)) / (mean(prior) + 1e-8) * 100.0
end

function email_receipt_signal(email_counts::Vector{Int}, baseline::Float64)::Float64
    return (mean(Float64.(email_counts)) - baseline) / (baseline + 1e-8) * 100.0
end

function second_party_data_quality(coverage::Float64, freshness_days::Float64,
                                    accuracy_estimate::Float64)::Float64
    cov_score = clamp(coverage, 0.0, 1.0)
    fresh_score = clamp(1.0 - freshness_days/30.0, 0.0, 1.0)
    acc_score = clamp(accuracy_estimate, 0.0, 1.0)
    return (cov_score + fresh_score + acc_score) / 3.0 * 100.0
end


# ============================================================
# SECTION 2B: BLOCKCHAIN ON-CHAIN ANALYTICS
# ============================================================

function detect_whale_transactions(values_usd::Vector{Float64},
                                   labels_from::Vector{String},
                                   labels_to::Vector{String},
                                   times::Vector{Int};
                                   threshold_usd::Float64=1e6)
    whale_idx = findall(v -> v >= threshold_usd, values_usd)
    exchange_inflows  = [values_usd[i] for i in whale_idx if labels_to[i] == "exchange"]
    exchange_outflows = [values_usd[i] for i in whale_idx if labels_from[i] == "exchange"]
    net = sum(exchange_inflows) - sum(exchange_outflows)
    total_whale = sum(values_usd[whale_idx])
    sell_pressure = sum(exchange_inflows) / (sum(exchange_inflows)+sum(exchange_outflows)+1e-10)
    return (whale_count=length(whale_idx), total_whale_usd=total_whale,
            net_exchange_flow=net, sell_pressure=sell_pressure)
end

function nvt_zscore_signal(market_cap::Vector{Float64}, tx_volume::Vector{Float64};
                            window::Int=90)
    nvt = market_cap ./ (tx_volume .+ 1e-10)
    n = length(nvt); signal = zeros(n)
    for i in window:n
        w = nvt[max(1,i-window+1):i]
        signal[i] = (nvt[i] - mean(w)) / (std(w) + 1e-10)
    end
    return (nvt=nvt, signal=signal)
end

function mvrv_zscore(market_cap::Vector{Float64}, realized_cap::Vector{Float64})
    mvrv = market_cap ./ (realized_cap .+ 1e-10)
    mu = mean(mvrv); sigma = std(mvrv)
    return (mvrv=mvrv, zscore=(mvrv .- mu) ./ (sigma .+ 1e-10))
end

function hash_ribbon_signal(hr_30d::Vector{Float64}, hr_60d::Vector{Float64})
    n = min(length(hr_30d), length(hr_60d))
    sig = zeros(n)
    for i in 2:n
        if hr_30d[i] > hr_60d[i] && hr_30d[i-1] <= hr_60d[i-1]; sig[i] = 1.0
        elseif hr_30d[i] < hr_60d[i] && hr_30d[i-1] >= hr_60d[i-1]; sig[i] = -1.0
        end
    end
    return sig
end

function stock_to_flow_model(stock::Float64, annual_flow::Float64)
    sf = stock / (annual_flow + 1e-10)
    return (sf_ratio=sf, predicted_price=exp(14.6 + 3.3*log(sf+1e-10)))
end

# ============================================================
# SECTION 3B: OPTIONS SIGNALS
# ============================================================

function compute_put_call_ratio(put_oi::Float64, call_oi::Float64,
                                 put_vol::Float64, call_vol::Float64)
    return (oi_pcr=put_oi/(call_oi+1e-10), vol_pcr=put_vol/(call_vol+1e-10),
            combined=(put_oi/(call_oi+1e-10) + put_vol/(call_vol+1e-10))/2)
end

function iv_rank(current_iv::Float64, iv_history::Vector{Float64})
    lo = minimum(iv_history); hi = maximum(iv_history)
    return 100.0*(current_iv-lo)/(hi-lo+1e-10)
end

function volatility_risk_premium(realized_vol::Vector{Float64}, implied_vol::Vector{Float64})
    vrp = implied_vol .- realized_vol
    return (vrp=vrp, mean_vrp=mean(vrp), signal=mean(vrp)>0 ? 1.0 : -1.0)
end

function gamma_exposure(gammas::Vector{Float64}, open_interests::Vector{Float64},
                         spot::Float64; contract_size::Float64=1.0)
    return sum(gammas .* open_interests .* contract_size .* spot^2 .* 0.01)
end

function composite_options_signal(pcr::Float64, risk_rev::Float64,
                                   ivr::Float64, term_slope::Float64)
    pcr_s = clamp(-(pcr-1.0)*2.0,-1.0,1.0)
    rr_s  = clamp(risk_rev*10.0,-1.0,1.0)
    ivr_s = clamp((50.0-ivr)/50.0,-1.0,1.0)
    ts_s  = clamp(-term_slope*5.0,-1.0,1.0)
    return (pcr_s + rr_s + ivr_s + ts_s)/4.0
end

# ============================================================
# SECTION 4B: FUTURES TERM STRUCTURE
# ============================================================

struct FuturesTermStructure
    tenors::Vector{Float64}
    prices::Vector{Float64}
    spot::Float64
    implied_yields::Vector{Float64}
end

function build_term_structure(spot::Float64, fut_prices::Vector{Float64},
                               tenors_days::Vector{Float64}, rf::Float64)
    n = length(fut_prices)
    implied_yields = [T>0 ? rf-log(fut_prices[i]/spot)/(tenors_days[i]/365) : rf
                      for (i,T) in enumerate(tenors_days./365)]
    return FuturesTermStructure(tenors_days, fut_prices, spot, implied_yields)
end

function contango_backwardation_index(ts::FuturesTermStructure)
    length(ts.prices)<2 && return 0.0
    return -(ts.prices[2]-ts.prices[1])/(ts.prices[1]+1e-10)
end

function roll_yield(near_price::Float64, far_price::Float64,
                    near_days::Float64, far_days::Float64)
    d = far_days - near_days
    d <= 0 && return 0.0
    return (near_price-far_price)/far_price * 365.0/d
end

function futures_basis_zscore(basis::Vector{Float64}; window::Int=30)
    n = length(basis); z = zeros(n)
    for i in window:n
        w = basis[max(1,i-window+1):i]
        z[i] = (basis[i]-mean(w))/(std(w)+1e-10)
    end
    return z
end

function open_interest_momentum(oi::Vector{Float64}, prices::Vector{Float64};
                                  window::Int=5)
    n = length(oi); sig = zeros(n)
    for i in window+1:n
        oi_c = (oi[i]-oi[i-window])/(oi[i-window]+1e-10)
        px_c = (prices[i]-prices[i-window])/(prices[i-window]+1e-10)
        sig[i] = sign(px_c)*min(abs(oi_c),1.0)
    end
    return sig
end

# ============================================================
# SECTION 5B: ORDER BOOK MICROSTRUCTURE
# ============================================================

struct OrderBook
    bids::Vector{Tuple{Float64,Float64}}
    asks::Vector{Tuple{Float64,Float64}}
    timestamp::Float64
end

function order_book_imbalance(ob::OrderBook; depth::Int=5)
    bv = sum(s for (_,s) in ob.bids[1:min(depth,length(ob.bids))])
    av = sum(s for (_,s) in ob.asks[1:min(depth,length(ob.asks))])
    return (bv-av)/(bv+av+1e-10)
end

function mid_price(ob::OrderBook)
    (isempty(ob.bids)||isempty(ob.asks)) && return NaN
    return (ob.bids[1][1]+ob.asks[1][1])/2
end

function weighted_mid_price(ob::OrderBook)
    (isempty(ob.bids)||isempty(ob.asks)) && return NaN
    bp,bs=ob.bids[1]; ap,as_=ob.asks[1]
    return (bp*as_+ap*bs)/(bs+as_+1e-10)
end

function vpin_estimator(buy_vols::Vector{Float64}, sell_vols::Vector{Float64};
                         bucket_size::Int=50)
    n = length(buy_vols); nb = n÷bucket_size
    vpin = [abs(sum(buy_vols[(b-1)*bucket_size+1:b*bucket_size]) -
                sum(sell_vols[(b-1)*bucket_size+1:b*bucket_size])) /
            (sum(buy_vols[(b-1)*bucket_size+1:b*bucket_size]) +
             sum(sell_vols[(b-1)*bucket_size+1:b*bucket_size]) + 1e-10)
            for b in 1:nb]
    return (vpin=vpin, mean_vpin=isempty(vpin) ? 0.0 : mean(vpin))
end

function kyle_lambda(dp::Vector{Float64}, order_flow::Vector{Float64})
    xb=mean(order_flow); yb=mean(dp)
    return sum((order_flow.-xb).*(dp.-yb))/(sum((order_flow.-xb).^2)+1e-10)
end

function amihud_illiquidity(returns::Vector{Float64}, volumes::Vector{Float64})
    return mean(abs.(returns)./(volumes.+1e-10))
end

function roll_spread_estimator(prices::Vector{Float64})
    dp = diff(prices); n = length(dp)
    c = sum(dp[1:end-1].*dp[2:end])/(n-1+1e-10)
    return c<0 ? 2*sqrt(-c) : 0.0
end

# ============================================================
# SECTION 6B: IC ANALYSIS
# ============================================================

function spearman_ic(signal::Vector{Float64}, fwd_ret::Vector{Float64})
    n = length(signal)
    sr = Float64.(sortperm(sortperm(signal)))
    rr = Float64.(sortperm(sortperm(fwd_ret)))
    return 1.0 - 6*sum((sr.-rr).^2)/(n*(n^2-1)+1e-10)
end

function rolling_ic(signals::Matrix{Float64}, returns::Vector{Float64}; window::Int=60)
    T,K = size(signals); ic = fill(NaN,T,K)
    for t in window:T, k in 1:K
        ic[t,k] = spearman_ic(signals[t-window+1:t,k], returns[t-window+1:t])
    end
    return ic
end

function icir(ic_series::Vector{Float64})
    v = filter(!isnan, ic_series)
    isempty(v) && return 0.0
    return mean(v)/(std(v)+1e-10)
end

function pca_signal_combination(signals::Matrix{Float64}; n_comp::Int=3)
    T,K = size(signals)
    mu = [mean(signals[:,k]) for k in 1:K]
    X = signals .- mu'
    C = X'*X./(T-1)
    comps = zeros(K,n_comp); res = copy(C)
    for j in 1:n_comp
        v = randn(K); v ./= (norm(v)+1e-10)
        for _ in 1:100
            vn = res*v; vn ./= (norm(vn)+1e-10)
            norm(vn-v)<1e-8 && break; v=vn
        end
        lam = dot(v,res*v); comps[:,j]=v; res.-=lam.*(v*v')
    end
    return (scores=X*comps, loadings=comps, centers=mu)
end

function ic_weighted_combination(signals::Matrix{Float64}, ics::Vector{Float64})
    ic_pos = max.(ics,0.0); total=sum(ic_pos)
    total<1e-10 && return zeros(size(signals,1))
    return signals*(ic_pos./total)
end

function signal_neutralization(signal::Vector{Float64}, sector_ids::Vector{Int},
                                 market_cap::Vector{Float64})
    neu = copy(signal)
    for s in unique(sector_ids)
        idx = findall(==(s),sector_ids)
        length(idx)<2 && continue
        neu[idx] .-= mean(signal[idx])
    end
    mc_total = sum(market_cap); mc_total<1e-10 && return neu
    neu .-= sum(neu.*market_cap)/mc_total
    return neu
end

function build_alpha_model(signals::Matrix{Float64}, fwd_ret::Vector{Float64};
                            lam::Float64=1e-3)
    T,K = size(signals); X = hcat(ones(T),signals)
    beta = (X'*X+lam*I(K+1))\(X'*fwd_ret)
    fitted = X*beta; resid = fwd_ret.-fitted
    return (beta=beta, r2=1.0-var(resid)/(var(fwd_ret)+1e-10))
end

# ============================================================
# SECTION 7B: REGIME DETECTION & ORTHOGONALIZATION
# ============================================================

function identify_regimes_hmm(returns::Vector{Float64}; n_regimes::Int=2, max_iter::Int=50)
    n = length(returns)
    mu_k = [quantile(returns, k/(n_regimes+1)) for k in 1:n_regimes]
    sig_k = fill(std(returns), n_regimes)
    A = fill(1.0/n_regimes, n_regimes, n_regimes)
    gmat = zeros(n, n_regimes)
    gpdf(x,m,s) = exp(-0.5*((x-m)/s)^2)/(sqrt(2π)*s+1e-15)

    for _ in 1:max_iter
        alpha = zeros(n,n_regimes); beta = zeros(n,n_regimes)
        for k in 1:n_regimes; alpha[1,k]=gpdf(returns[1],mu_k[k],sig_k[k])/n_regimes; end
        for t in 2:n
            for k in 1:n_regimes
                alpha[t,k]=gpdf(returns[t],mu_k[k],sig_k[k])*
                            sum(alpha[t-1,j]*A[j,k] for j in 1:n_regimes)
            end
            s=sum(alpha[t,:]); s>0 && (alpha[t,:]./=s)
        end
        beta[n,:].=1.0
        for t in n-1:-1:1
            for j in 1:n_regimes
                beta[t,j]=sum(A[j,k]*gpdf(returns[t+1],mu_k[k],sig_k[k])*beta[t+1,k]
                               for k in 1:n_regimes)
            end
            s=sum(beta[t,:]); s>0 && (beta[t,:]./=s)
        end
        for t in 1:n
            s=sum(alpha[t,k]*beta[t,k] for k in 1:n_regimes)
            for k in 1:n_regimes; gmat[t,k]=alpha[t,k]*beta[t,k]/(s+1e-15); end
        end
        for k in 1:n_regimes
            g=gmat[:,k]; tot=sum(g)+1e-10
            mu_k[k]=sum(g.*returns)/tot
            sig_k[k]=sqrt(sum(g.*(returns.-mu_k[k]).^2)/tot)+1e-6
        end
    end
    states = [argmax(gmat[t,:]) for t in 1:n]
    return (states=states, mu=mu_k, sigma=sig_k)
end

function orthogonalize_signal(target::Vector{Float64}, controls::Matrix{Float64})
    res = copy(target)
    for k in 1:size(controls,2)
        c = controls[:,k]
        res .-= dot(res,c)/(dot(c,c)+1e-10).*c
    end
    return res
end

function regime_conditional_performance(returns::Vector{Float64},
                                         signal::Vector{Float64},
                                         states::Vector{Int}; n_regimes::Int=2)
    Dict(r => begin
        idx=findall(==(r),states); length(idx)<5 && continue
        ret_r=returns[idx]; sig_r=signal[idx]
        (ic=spearman_ic(sig_r,ret_r),
         sharpe=mean(ret_r)/(std(ret_r)+1e-10)*sqrt(252),
         n=length(idx))
    end for r in 1:n_regimes if length(findall(==(r),states))>=5)
end

# ============================================================
# SECTION 8B: MACRO & SATELLITE SIGNALS
# ============================================================

function google_trends_momentum(volume::Vector{Float64}; fast::Int=4, slow::Int=26)
    n=length(volume)
    f=[mean(volume[max(1,i-fast+1):i]) for i in 1:n]
    s=[mean(volume[max(1,i-slow+1):i]) for i in 1:n]
    return (fast_ma=f, slow_ma=s, signal=(f.-s)./(s.+1e-10))
end

function ship_tracking_signal(ships_at_sea::Vector{Float64}, ships_in_port::Vector{Float64})
    ratio = ships_at_sea./(ships_at_sea.+ships_in_port.+1e-10)
    mu=mean(ratio); sigma=std(ratio)
    return tanh.((ratio.-mu)./(sigma.+1e-10))
end

function crop_ndvi_signal(ndvi::Vector{Float64}; window::Int=4)
    n=length(ndvi)
    ma=[mean(ndvi[max(1,i-window+1):i]) for i in 1:n]
    return -tanh.((ma.-mean(ndvi))./(std(ndvi).+1e-10))
end

function electricity_consumption_signal(consumption::Vector{Float64}; period::Int=52)
    n=length(consumption)
    adj=[i>period ? consumption[i]/(consumption[i-period]+1e-10)-1.0 : 0.0
         for i in 1:n]
    v=filter(!iszero,adj)
    isempty(v) && return zeros(n)
    return tanh.((adj.-mean(v))./(std(v).+1e-10))
end

function credit_card_spending_signal(spending::Vector{Float64}; yoy_window::Int=52)
    n=length(spending)
    yoy=[(spending[i]-spending[max(1,i-yoy_window)])/(spending[max(1,i-yoy_window)]+1e-10)
         for i in 1:n]
    return tanh.((yoy.-mean(yoy))./(std(yoy).+1e-10))
end

function earnings_revision_signal(current_est::Vector{Float64}, prior_est::Vector{Float64})
    rev=(current_est.-prior_est)./(abs.(prior_est).+1e-10)
    return tanh.((rev.-mean(rev))./(std(rev).+1e-10))
end

function fear_greed_index(vol_z::Float64, mom_z::Float64, vol_flow_z::Float64,
                           sent_z::Float64, pcr_z::Float64)
    raw = -vol_z*0.25+mom_z*0.25+vol_flow_z*0.2+sent_z*0.2-pcr_z*0.1
    return clamp(50.0+raw*10.0, 0.0, 100.0)
end

# ============================================================
# SECTION 9B: COMPOSITE SIGNAL CONSTRUCTION
# ============================================================

struct CompositeSignal
    name::String
    components::Vector{String}
    weights::Vector{Float64}
    values::Vector{Float64}
    zscore::Vector{Float64}
    confidence::Float64
end

function build_composite_signal(signals::Matrix{Float64}, names::Vector{String},
                                  ic_weights::Vector{Float64}; zscore_window::Int=252)
    T,K = size(signals)
    w = max.(ic_weights,0.0); s=sum(w); s<1e-10 && (w=ones(K)./K); w./=sum(w)
    std_sigs = zeros(T,K)
    for k in 1:K
        mu=mean(signals[:,k]); sg=std(signals[:,k])
        std_sigs[:,k]=(signals[:,k].-mu)./(sg.+1e-10)
    end
    comp = std_sigs*w; zs = zeros(T)
    for t in zscore_window:T
        hw=comp[max(1,t-zscore_window+1):t]
        zs[t]=(comp[t]-mean(hw))/(std(hw)+1e-10)
    end
    return CompositeSignal("composite", names, w, comp, zs, min(1.0, T/(2*zscore_window)))
end

function backtest_signal(signal::Vector{Float64}, returns::Vector{Float64};
                          tc::Float64=0.001)
    T=min(length(signal),length(returns))
    pos=sign.(signal[1:T]); pnl=zeros(T)
    for t in 2:T
        pnl[t]=pos[t-1]*returns[t]-abs(pos[t]-pos[t-1])*tc
    end
    cum=cumsum(pnl); sr=mean(pnl)/(std(pnl)+1e-10)*sqrt(252)
    peak=cum[1]; mdd=0.0
    for v in cum; peak=max(peak,v); mdd=max(mdd,peak-v); end
    hr=count(p->p>0, pnl[2:end])/(T-1)
    return (pnl=pnl, cum_pnl=cum, sharpe=sr, max_drawdown=mdd, hit_rate=hr)
end

function signal_combination_ensemble(signals::Matrix{Float64},
                                       fwd_returns::Vector{Float64};
                                       method::Symbol=:ic_weight)
    T,K = size(signals)
    if method == :ic_weight
        ics = [spearman_ic(signals[:,k], fwd_returns) for k in 1:K]
        ic_pos = max.(ics,0.0); tot=sum(ic_pos)
        w = tot<1e-10 ? ones(K)./K : ic_pos./tot
    elseif method == :ridge
        lam=1e-2; w=(signals'*signals+lam*I(K))\(signals'*fwd_returns)
        w=max.(w,0.0); tot=sum(w); w=tot<1e-10 ? ones(K)./K : w./tot
    else
        w = ones(K)./K
    end
    return (weights=w, combined=signals*w)
end

# ============================================================
# DEMO
# ============================================================

function demo_alternative_data_all()
    println("=== Full Alternative Data Demo ===")

    # Whale detection
    vals=abs.(randn(100)).*1e6
    lf=rand(["exchange","whale","unknown"],100)
    lt=rand(["exchange","whale","unknown"],100)
    wh=detect_whale_transactions(vals,lf,lt,collect(1:100); threshold_usd=5e5)
    println("Whales: ", wh.whale_count, " NetFlow: ", round(wh.net_exchange_flow, sigdigits=3))

    # Options
    pcr=compute_put_call_ratio(1.2e6,1.0e6,800e3,700e3)
    println("PCR combined: ", round(pcr.combined, digits=3))

    # Futures
    ts=build_term_structure(30000.0,[30150.0,30400.0,30800.0],[7.0,30.0,90.0],0.05)
    println("Contango idx: ", round(contango_backwardation_index(ts), digits=4))

    # OB
    ob=OrderBook([(29990.0-i*10,rand()*5) for i in 0:4],
                  [(30010.0+i*10,rand()*5) for i in 0:4], 0.0)
    println("OB imbalance: ", round(order_book_imbalance(ob), digits=3))

    # IC
    sig=randn(200); ret=0.3*sig.+0.7*randn(200)
    println("Spearman IC: ", round(spearman_ic(sig,ret), digits=3))

    # HMM
    hmm=identify_regimes_hmm(randn(300).*0.01)
    println("HMM mu: ", round.(hmm.mu, sigdigits=3))

    # Fear & Greed
    println("Fear&Greed: ", round(fear_greed_index(-0.5,1.2,0.8,0.6,-0.3), digits=1))

    # Composite backtest
    sigs=randn(500,3); fwd=0.1*sigs[:,1].+randn(500)*0.15
    cs=build_composite_signal(sigs,["a","b","c"],[0.4,0.3,0.3])
    bt=backtest_signal(cs.values, fwd)
    println("Backtest Sharpe: ", round(bt.sharpe,digits=2), " HitRate: ", round(bt.hit_rate,digits=2))
end

end # module AlternativeData
