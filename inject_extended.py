"""
inject_extended.py
Adds longer, substantive plain-English explanations to every major section
that currently lacks one, then regenerates the PDF.
"""
import re, os

IN_HTML  = r"C:/Users/Matthew/Desktop/srfm-experiments/event_horizon_book_oxidized.html"
OUT_HTML = r"C:/Users/Matthew/Desktop/srfm-experiments/event_horizon_book_oxidized.html"
OUT_PDF  = r"C:/Users/Matthew/Desktop/srfm-experiments/event_horizon_book.pdf"

def box(text):
    return (
        '\n<div class="nonphd"><div class="nonphd-label">Plain English</div>'
        + text +
        '</div>\n'
    )

def p(text):
    return f'<p>{text}</p>\n'

HEADING_CALLOUTS = {

    "1.1 The Efficient Market Hypothesis and Its Limits": box(
        p("The Efficient Market Hypothesis (EMH) says, in plain terms: you cannot consistently "
          "beat the market because prices already reflect all available information. "
          "If a stock is underpriced, traders will immediately buy it until the price rises to fair value. "
          "By the time you read a news headline, the price has already moved.") +
        p("In practice this creates three versions of the claim. The weak form says past prices contain "
          "no useful information: charts are noise. The semi-strong form says publicly available "
          "information is already priced in: fundamental analysis adds no edge. "
          "The strong form says even private information is priced in: insider trading is pointless.") +
        p("The empirical record is uncomfortable for all three versions. Momentum works: assets that "
          "went up last month tend to keep going up for a few more months. Value works: cheap assets "
          "outperform expensive ones over long horizons. Crashes happen that no 'efficient' pricing "
          "model could justify. The 2008 financial crisis was not a surprise in hindsight; "
          "the warning signals were measurable. The EMH could not see them because it was not looking "
          "at topology, microstructure, or network connectivity. This program is built to see exactly those things.")
    ),

    "1.2 Gaussian Returns: A Convenient Fiction": box(
        p("The bell curve, or normal distribution, says that extreme events are astronomically rare. "
          "If daily stock returns were truly normally distributed, a one-day drop of 20 percent "
          "should occur roughly once every several billion years. "
          "The US market has experienced such drops several times in the last century.") +
        p("The Gaussian assumption is not just theoretically wrong; it is operationally dangerous. "
          "Value-at-Risk models built on normal returns told banks in 2008 that their positions "
          "were safe. Standard option pricing models underprice deep out-of-the-money puts by "
          "factors of ten or more because they assign near-zero probability to catastrophic scenarios "
          "that actually happen fairly regularly.") +
        p("The alternative is fat-tailed distributions: the Student-T, the Generalized Pareto, "
          "the stable Levy distribution. These assign genuine probability mass to extreme events. "
          "The entire architecture of this program, from the Student-T HMM in Phase II "
          "to the Generalized Pareto tail fitting in Phase III, is built on the refusal to "
          "pretend that financial returns are Gaussian. That single design choice changes every "
          "downstream calculation.")
    ),

    "1.3 The Hidden Architecture of Financial Crises": box(
        p("A financial crisis does not begin on the day prices collapse. "
          "The research throughout this program shows that measurable structural changes "
          "precede the visible crash by hundreds of trading bars. "
          "The sequence is consistent: topology contracts, causal connections densify, "
          "the network becomes over-connected, and only then do prices fall.") +
        p("Think of it like a building before it collapses. The steel framework bends and creaks "
          "for a long time before the roof falls. Engineers can measure the stress in the structure "
          "while there is still time to act. The seven phases of this program are the financial "
          "equivalent of stress sensors placed at strategic points in the market's architecture.") +
        p("The key insight is that the architecture of crisis is not random. It follows a sequence: "
          "first the geometry changes (Ricci curvature and topology), then the information channels "
          "collapse (transfer entropy and Granger density spike), then the microstructure heats up "
          "(Hawkes intensity surges), and only at the end do prices visibly break. "
          "Each layer is measurable independently. Combined into one Grand Unified Score, "
          "they form an early warning system that fires well before the tape shows anything unusual.")
    ),

    "1.4 The Eight Regimes of Market Dynamics": box(
        p("Markets do not behave the same way all the time. Any trader knows this intuitively: "
          "some periods feel like everything moves together, others feel choppy and random, "
          "some feel like a slow grind upward, others like a free fall. "
          "The eight regimes identified in Phase I are a formal taxonomy of these states.") +
        p("Each regime has a signature in the signal space: a characteristic Hawkes intensity level, "
          "a characteristic Ricci curvature, a characteristic transfer entropy pattern. "
          "Knowing which regime the market is in right now tells you which strategy to deploy. "
          "A mean-reversion strategy that works perfectly in a choppy regime will lose money "
          "steadily in a trending regime. The Mixture-of-Experts gating network in Phase IV "
          "is the automated version of this regime-aware strategy switching.")
    ),

    "2.1 Topological Data Analysis: Reading Shape in Data": box(
        p("Statistics describes data through numbers: the mean, the variance, the correlation. "
          "Topology describes data through shape: are the points clustered in one blob or several? "
          "Do they form a ring? A surface with holes? These shapes contain information that "
          "averages and correlations completely miss.") +
        p("Imagine plotting each day's returns for thirty assets as a point in a thirty-dimensional "
          "space. On normal days, the points form a loose cloud. As a crisis approaches, "
          "the cloud compresses: all assets start moving together and the cloud flattens "
          "into a lower-dimensional pancake. TDA detects this compression before any "
          "individual correlation exceeds an alarm threshold.") +
        p("The specific tool used here is persistent homology. It works by connecting nearby points "
          "with edges and triangles as you gradually increase a distance parameter, tracking which "
          "clusters form and which holes appear. Features that persist across a wide range of "
          "distance parameters are structurally real. Features that flicker in and out are noise. "
          "The 'persistence landscape' converts this into a single curve that can be compared "
          "across time windows, giving a time series of market shape.")
    ),

    "2.2 Graph Theory and Network Measures": box(
        p("A graph in mathematics is simply a set of nodes connected by edges. "
          "In a financial network, each node is an asset and each edge represents a relationship: "
          "correlation, Granger causality, shared sector exposure, or any other measurable link.") +
        p("Graph theory gives us powerful tools to ask questions about this network. "
          "Which node is most central, meaning whose removal would disconnect the most other nodes? "
          "That is betweenness centrality. Which node is most 'important' in the sense of being "
          "linked to by other important nodes? That is PageRank. How tightly clustered is the "
          "network overall? That is Ricci curvature.") +
        p("The financial insight is that markets are not a collection of independent assets: "
          "they are a network, and network structure determines how shocks propagate. "
          "A highly centralized network with one dominant hub is brittle: "
          "hit the hub and the whole system falls. A distributed network is resilient. "
          "Measuring network structure in real time is therefore a direct measure of systemic fragility.")
    ),

    "2.3 Information Theory: Transfer Entropy and Mutual Information": box(
        p("Shannon entropy measures uncertainty. If you know a coin is fair, "
          "its entropy is high: you have no idea which way it will land. "
          "If the coin is weighted to land heads 99 percent of the time, its entropy is low: "
          "you already know the answer before flipping.") +
        p("Mutual information extends this: it measures how much knowing the outcome of "
          "one variable reduces your uncertainty about another. Two assets with high mutual "
          "information are informationally coupled: knowing one tells you something about the other. "
          "Unlike correlation, mutual information catches nonlinear relationships "
          "and does not assume any particular functional form.") +
        p("Transfer entropy goes further by adding the dimension of time. "
          "It asks: does knowing Asset A's past reduce your uncertainty about Asset B's future, "
          "beyond what B's own past already tells you? "
          "If yes, information is flowing from A to B. This directionality is crucial: "
          "in the run-up to the 2008 crisis, information flowed from the housing derivatives "
          "market into equities, not the other way around. "
          "Transfer entropy would have detected that one-way flow.")
    ),

    "2.4 Stochastic Processes: Hawkes, HJB, and Extreme Values": box(
        p("A stochastic process is a mathematical model for something that evolves randomly over time. "
          "Financial prices are a canonical example: they move up and down in ways that "
          "cannot be predicted exactly but do follow statistical patterns.") +
        p("The three stochastic frameworks used here each address a different aspect of market behavior. "
          "The Hawkes process addresses clustering: trades arrive in bursts, not uniformly. "
          "The Hamilton-Jacobi-Bellman equation addresses control: given stochastic price dynamics, "
          "what is the mathematically optimal exit strategy? "
          "Extreme Value Theory addresses tails: given that most of the time is unremarkable, "
          "how should we model the rare but catastrophic exceptions?") +
        p("Together they cover the three questions a practitioner actually needs to answer: "
          "when is activity clustering (Hawkes), when should I exit (HJB), "
          "and how bad could it get (EVT). None of these questions can be answered well "
          "with the simple random-walk models that underpin most retail trading platforms.")
    ),

    "2.5 Reinforcement Learning Architecture": box(
        p("Reinforcement learning (RL) is a framework for training an agent to make good decisions "
          "through trial and error. Unlike supervised learning, which requires labelled examples "
          "of correct answers, RL agents learn by doing: they take actions, observe outcomes, "
          "receive a reward or penalty, and gradually adjust their behaviour to maximize "
          "cumulative reward over time.") +
        p("In the trading context, the agent is the strategy. The state is everything the agent "
          "can observe: current signal values, recent price history, portfolio position. "
          "The action is the trade decision: go long, go short, or stay flat. "
          "The reward is the risk-adjusted return from that decision. "
          "Over tens of thousands of simulated trading days, the agent learns which state "
          "configurations reliably lead to profitable actions and which do not.") +
        p("The specific architectures used (D3QN, DDQN, TD3, PPO) differ mainly in how they "
          "handle the trade-off between exploration (trying new things to learn) and exploitation "
          "(doing what worked before). TD3 is designed for continuous action spaces "
          "and is particularly good at controlling position size. PPO is robust and stable, "
          "making it the standard choice for the position-sizing role in the ensemble.")
    ),

    "4.3 Factor Zoo Decay and the Bayesian Credibility Debate": box(
        p("The 'Factor Zoo' is a term for the hundreds of stock market anomalies published in "
          "academic journals: momentum, value, quality, low volatility, profitability, and so on. "
          "Each paper claims its factor predicts returns. Most of them do not survive out of sample.") +
        p("Why? Two reasons. First, data mining: with enough variables and enough slicing of "
          "historical data, any researcher can find patterns that look significant but are pure noise. "
          "Second, arbitrage: once a factor is published, professional traders pile into it, "
          "and the act of trading eliminates the very mispricing the factor was exploiting.") +
        p("The Bayesian Credibility Debate is this program's solution. Instead of committing to "
          "one factor or one model, a panel of agents each advocates for a different signal. "
          "Their voting weight is proportional to their recent accuracy, updated every period. "
          "A factor that used to work but has decayed loses weight automatically and gradually "
          "rather than catastrophically, because the system keeps reassessing the evidence. "
          "It is the quantitative equivalent of a scientific community that updates its beliefs "
          "as new data arrives rather than defending its prior papers.")
    ),

    "5.3 Wormhole Contagion Network": box(
        p("In normal markets, the correlation between most pairs of assets is modest. "
          "A utility stock and a semiconductor stock might have a 0.3 correlation: "
          "they vaguely move together because they are both equities, "
          "but their day-to-day movements are largely independent.") +
        p("During a crisis, this changes dramatically. Correlations spike across the board: "
          "everything falls together regardless of sector, geography, or fundamental story. "
          "This is the 'wormhole' effect: a sudden high-correlation link appears between assets "
          "that normally have nothing to do with each other, as if a shortcut through the "
          "market's space has opened up.") +
        p("The wormhole edges are crisis accelerants. When contagion can jump directly from, "
          "say, US high-yield bonds to European equities without needing to travel through "
          "the normal chain of intermediaries, it spreads faster and further than any model "
          "that assumes stable network structure would predict. "
          "Detecting the appearance of wormhole edges in real time is therefore an early warning "
          "of the 'correlation-one' regime that precedes systemic crashes.")
    ),

    "6.3 The Zero-Dimension Arbitrage Window": box(
        p("The Zero-Dimension Arbitrage Window (ZDIM) is one of the most counterintuitive "
          "results in the program, so it is worth explaining carefully.") +
        p("Normally, if two assets are highly correlated, we expect there to be an identifiable "
          "causal reason: they share a common driver, or one causes the other. "
          "The ZDIM is the opposite: high correlation with no measurable causal connection. "
          "The Granger edges have vanished, but the prices are still moving together.") +
        p("Why does this happen? It occurs when a third, unobserved factor is simultaneously "
          "driving both assets. The causal network looks empty because the actual cause is not "
          "in the data. Both assets are responding to the same invisible signal.") +
        p("The trading implication is significant. In normal correlation-driven co-movement, "
          "a pairs trade is risky because you are betting against a real causal force. "
          "In the ZDIM, there is no causal force maintaining the relationship: "
          "both assets are just shadows of something else. "
          "When the hidden driver disappears, the correlation will snap back to its "
          "fundamental level very quickly. The ZDIM identifies exactly those moments "
          "when a mean-reversion trade has a structural, not just statistical, basis.")
    ),

    "9.1 Synthetic On-Chain Signal Stream": box(
        p("Decentralized Finance (DeFi) operates on public blockchains. "
          "Every transaction, every trade, every dollar moved between wallets is permanently "
          "recorded and publicly readable. This is a complete inversion of traditional finance, "
          "where order flow is private and institutional positioning is only partially disclosed.") +
        p("The three on-chain signals synthesized here represent the most actionable categories "
          "of blockchain data. DEX (Decentralized Exchange) volume spikes signal that "
          "sophisticated traders are repositioning in cryptocurrency markets, often a leading "
          "indicator of volatility in correlated traditional assets. "
          "Whale net flow measures the net movement of large wallet addresses in and out of "
          "stable coins: accumulation of stable coins by large holders often precedes "
          "risk-off moves in equities. LP depth volatility tracks how much liquidity "
          "providers are withdrawing from automated market makers, a direct measure of "
          "professional risk appetite in real time.") +
        p("None of these signals are available in traditional market data vendors. "
          "Their potential value comes precisely from the fact that most equity traders do not "
          "monitor them, creating an information gap that could be exploited if the cross-domain "
          "correlations identified here are present in live data.")
    ),

    "10.1 The Feature Hypercube: Normalizing Fifteen Signals": box(
        p("The fifteen signals in the Grand Unified Model come from completely different mathematical "
          "universes. Hawkes intensity is measured in events per unit time. "
          "Ricci curvature is a dimensionless number that can be positive or negative. "
          "Transfer entropy is measured in bits. Granger density is a fraction between zero and one. "
          "You cannot simply add these together: it would be like adding meters to kilograms.") +
        p("The percentile transform solves this by asking, for each signal at each moment: "
          "'where does today's value rank among the last 500 observations?' "
          "If today's Hawkes intensity is higher than 90 percent of the last 500 days, "
          "it becomes 0.90. If today's Ricci curvature is at the 30th percentile, it becomes 0.30. "
          "Now all fifteen signals are on the same zero-to-one scale, and they can be "
          "meaningfully combined, compared, and fed into a neural network together.") +
        p("This is distribution-free normalization: it makes no assumptions about the "
          "underlying distribution of any signal. It is also robust to outliers: "
          "a single extreme reading shifts only a few percentile ranks rather than "
          "distorting an entire mean or standard deviation calculation. "
          "The Feature Hypercube is the prerequisite for everything else in Phase VII.")
    ),

    "10.3 Grand Unified Strategy and Best-of-All Switching": box(
        p("The Grand Unified Agent is a three-layer neural network that takes all fifteen "
          "normalized signals as input and outputs a trading decision: long, short, or flat. "
          "It is trained online, updating its weights continuously as new data arrives, "
          "which means it adapts to the current regime rather than relying on a static "
          "model trained on historical data.") +
        p("But even this agent is not trusted unconditionally. The Best-of-All switching mechanism "
          "runs a competition in real time. At every bar, it compares the recent Sharpe ratio "
          "of the Grand Unified Agent against the recent Sharpe of the best-performing "
          "single-signal strategy from the preceding period. If a simpler strategy is currently "
          "outperforming the unified model, the system routes trades through that simpler strategy.") +
        p("This is a form of model humility: the most sophisticated model does not always win. "
          "In regimes where a single signal (say, Hawkes intensity alone) is extremely predictive, "
          "adding thirteen other noisy signals to it reduces performance. "
          "The Best-of-All switch detects this situation and steps back to the simpler model "
          "automatically. The result is a strategy that is both sophisticated when complexity "
          "helps and disciplined enough to use simple tools when simple tools are what the "
          "current regime rewards.")
    ),

    "11.1 The Seven-Layer Model of Crisis Formation": box(
        p("The seven-layer model is the central theoretical contribution of this program. "
          "It says that financial crises do not arrive randomly: they build through a consistent "
          "sequence of structural changes, each measurable in a different signal layer, "
          "each occurring at a predictable lead time before the price collapse.") +
        p("Layer 1 is the earliest warning, firing roughly 800 bars before the crisis. "
          "The optimal stopping region shifts: the HJB equation says the market's risk-return "
          "trade-off has changed even though prices look calm. "
          "Layer 2, around 700 bars out, sees the Ricci curvature of the network begin rising: "
          "assets are becoming more connected. Layer 3 sees the tail distribution fatten: "
          "extreme events are becoming more likely.") +
        p("By Layers 4 and 5 (300 to 500 bars out), the causal network is densifying and "
          "the multifractal spectrum is narrowing, meaning the market's self-similarity "
          "across time scales is breaking down. Layer 6 (around 100 bars out) brings the "
          "Hawkes intensity surge: order clustering is accelerating. "
          "Layer 7 is the crash itself.") +
        p("The practical message is simple: by the time Layer 6 fires, you have already "
          "missed five earlier warning layers. A system monitoring all seven layers "
          "in parallel can, in principle, begin reducing risk exposure hundreds of bars "
          "before the event that every other market participant calls a 'surprise.'")
    ),

    "11.2 The Pre-Crisis Diagnostic Checklist": box(
        p("The pre-crisis checklist operationalizes the seven-layer model into a concrete "
          "monitoring tool. Each item on the checklist corresponds to one or more signals "
          "exceeding a historically calibrated threshold. "
          "When more items are checked, the probability of a crisis within the next "
          "N bars rises according to the empirically estimated relationship.") +
        p("Think of it as a doctor's checklist before surgery: individually, "
          "elevated blood pressure is concerning but not disqualifying. "
          "Elevated blood pressure combined with abnormal clotting factors and a "
          "family history of cardiac events together cross a threshold that changes the decision. "
          "The Singularity Score is the numerical summary of how many items are checked "
          "and how far they deviate from normal.")
    ),

    "11.3 Topology is Not Optional: Why Geometry Precedes Statistics": box(
        p("The central argument of this section is that geometric and topological signals "
          "are not an exotic addition to a conventional statistical framework: "
          "they are more fundamental. Statistics summarizes what has happened. "
          "Topology describes the structural conditions under which things can happen.") +
        p("A simple example: two assets with 0.9 correlation are statistically very similar. "
          "But knowing their correlation tells you nothing about whether that correlation "
          "is maintained by a direct causal link, an indirect common cause, or pure coincidence. "
          "Topology and causal graph theory can distinguish between these three cases. "
          "The practical difference is enormous: a direct causal link is stable and tradeable, "
          "a coincidental correlation can vanish overnight.") +
        p("The program shows empirically that topological signals (Ricci curvature, "
          "persistent homology, causal density) precede statistical signals (volatility spikes, "
          "correlation jumps) by hundreds of bars. This temporal ordering is not an accident: "
          "the geometry of the market's state space changes first, and the statistics that "
          "practitioners normally monitor are downstream consequences of that geometric change. "
          "Monitoring only the downstream statistics is like watching the smoke while "
          "ignoring the fire.")
    ),

    "12.1 Data Requirements and Preprocessing": box(
        p("The program runs entirely on a matrix of synthetic daily returns for thirty assets "
          "simulated over 1,000 bars, with a calibrated crisis event injected at bar 825. "
          "For a live implementation, the direct equivalent would be daily total-return series "
          "for a universe of liquid instruments: index ETFs, sector ETFs, or individual stocks.") +
        p("The preprocessing steps are minimal by design: compute log returns from prices, "
          "winsorize at the 1st and 99th percentile to prevent single outliers from "
          "corrupting rolling statistics, and ensure all series start from the same date. "
          "No more complex pre-processing than this is needed or recommended: "
          "overly elaborate preprocessing introduces its own assumptions and can "
          "inadvertently leak future information into past computations.")
    ),

    "12.2 Avoiding Lookahead Bias": box(
        p("Lookahead bias is the single most common reason a backtest looks excellent "
          "but live trading loses money. It happens when a model is trained or calibrated "
          "using information that would not have been available at the time the trade was made.") +
        p("The classic example: you normalize returns using the mean and standard deviation "
          "of the full historical sample. But the standard deviation of a sample that includes "
          "tomorrow's crash is different from the standard deviation known before the crash. "
          "Your model implicitly 'knew' the crash was coming because its normalization "
          "parameters were contaminated by it.") +
        p("The Project Event Horizon design avoids this through a single strict rule: "
          "every computation at bar t uses only data from bars 1 through t-1. "
          "Rolling windows for normalization, for model fitting, for threshold calibration: "
          "all use only the past. This seems obvious but is surprisingly easy to violate "
          "in complex multi-signal pipelines, and the program's architecture was specifically "
          "designed to make this violation structurally impossible rather than just unlikely.")
    ),

    "12.3 Transaction Costs and Market Impact": box(
        p("The synthetic experimental results do not incorporate transaction costs. "
          "In live markets, every trade has a cost: the bid-ask spread, "
          "exchange fees, and market impact (the price moving against you as you buy or sell). "
          "For a high-frequency system, these costs dominate. "
          "For a daily-rebalancing system like this program, they are smaller but still significant.") +
        p("A rough rule of thumb for liquid equity ETFs is 2 to 5 basis points (0.02 to 0.05 percent) "
          "per round-trip trade. A strategy that turns over its entire portfolio once per week "
          "is paying roughly 1 to 2.5 percent per year in costs before seeing any alpha. "
          "The reported Sharpe ratios in this program should therefore be discounted accordingly "
          "when estimating live performance.")
    ),

    "12.4 Rolling Window Choices and Computational Budget": box(
        p("Every rolling computation in this program involves a choice of window length: "
          "how many past bars to include. Longer windows give more stable estimates but "
          "react slowly to regime changes. Shorter windows are responsive but noisy.") +
        p("The program uses a 500-bar window as its primary lookback throughout. "
          "At daily frequency this is roughly two years of data, which is long enough "
          "to estimate most statistics reliably but short enough to adapt within a "
          "market cycle. The specific value was chosen pragmatically and would need "
          "re-optimization for different asset classes or frequencies.") +
        p("The computational budget is dominated by the N-by-N Granger causality matrix, "
          "which requires fitting N squared bivariate VARs at each bar. "
          "For N equals 30 assets, this is 900 regressions per bar. "
          "At 1,000 bars, this is 900,000 regressions in total. "
          "On a modern laptop this runs in a few minutes. "
          "Scaling to 500 assets would require parallelization or GPU acceleration.")
    ),

    "12.5 From Synthetic to Live Data: Key Differences": box(
        p("The synthetic environment is a controlled laboratory. The real market is not. "
          "Several structural differences deserve careful attention before deploying anything "
          "from this program in a live setting.") +
        p("Non-stationarity is the biggest challenge. The synthetic data is generated by a "
          "relatively stable process with one crisis event. Real markets change their character "
          "continuously: the correlations of 2005 are not the correlations of 2015. "
          "Any model trained on a fixed historical window will drift out of calibration. "
          "The Page-Hinkley drift detection and online learning components of this program "
          "are specifically designed to address this, but they should be stress-tested "
          "against historical regime changes before live deployment.") +
        p("Survivorship bias is a second concern. The synthetic universe contains 30 assets "
          "that all survive the full 1,000-bar period. In real markets, "
          "companies go bankrupt, get acquired, and get delisted. "
          "A realistic backtesting universe must include assets that were removed from indices "
          "at various points, or the historical returns will be systematically overstated.") +
        p("Finally, DeFi on-chain signals currently require manual pipeline construction "
          "to extract from public blockchain nodes or data providers. "
          "The latency, reliability, and cost of this data infrastructure are non-trivial "
          "and should be assessed as a first step in any live implementation.")
    ),

    "13.2 Limitations and Caveats": box(
        p("The honest summary of what this program does not prove is as important as "
          "what it does. Every result reported here is in-sample or on synthetic data. "
          "There is no out-of-sample test on live markets, and the reported Sharpe ratios "
          "and crisis-detection precisions will not be reproduced exactly in live trading.") +
        p("The crisis event in the synthetic data was injected by the researchers. "
          "This means the model was effectively trained and tested on data generated by "
          "a known process. Real crises are caused by events and mechanisms that are "
          "not known in advance and may differ qualitatively from the synthetic scenario.") +
        p("The on-chain DeFi signals are the most speculative component. "
          "The 15-bar lead time of whale accumulation as a predictor of equity volatility "
          "is a synthetic calibration result. Whether it holds in real markets, "
          "where the causal mechanism would need to involve actual capital flows between "
          "DeFi and TradFi, is an open empirical question that this program does not resolve.")
    ),

    "13.3 Open Research Questions": box(
        p("The most important open question is whether the seven-layer crisis formation model "
          "holds across different asset classes, time periods, and crisis types. "
          "The 2008 credit crisis, the 2020 pandemic crash, and the 2022 rate shock "
          "each had different structural origins. Does the topology-first, statistics-second "
          "ordering hold for all of them? That is a testable hypothesis requiring "
          "real historical data and careful retrospective analysis.") +
        p("A second open question is the optimal way to combine the fifteen signals. "
          "The linear combination in the Singularity Score and the neural network in the "
          "Grand Unified Agent are two answers. Gaussian Processes, ensemble trees, "
          "and attention-based transformer models might do better. "
          "The signal discovery work is complete; the optimal fusion architecture is not.")
    ),

    "13.4 The Road to Live Markets": box(
        p("A realistic path from this research to a live trading system involves "
          "four sequential steps. First: validate the signal discovery results on "
          "real historical data. This means replicating the phase-by-phase signal analysis "
          "on actual equity return data covering at least one major crisis period.") +
        p("Second: build the data pipeline for all signal inputs, including the on-chain "
          "DeFi feeds, which do not come from standard market data vendors. "
          "Third: run a forward-looking paper trading period of at least twelve months "
          "before committing real capital, using the live data pipeline but without "
          "actual execution. This surfaces latency issues, data quality problems, "
          "and model degradation that only appear in real conditions.") +
        p("Fourth: size positions conservatively on first deployment, targeting a "
          "fraction of the Kelly-optimal size, and monitor the Page-Hinkley drift "
          "detector aggressively. If the detector fires within the first few months, "
          "the model has already encountered a regime it was not trained for, "
          "and the system should scale down automatically until recalibration is complete.")
    ),

    "A.1 Vietoris-Rips Persistent Homology": box(
        p("The Vietoris-Rips complex at radius r connects every pair of data points "
          "that are within distance r of each other with an edge. "
          "As r increases from zero to infinity, the complex grows from isolated points "
          "to a fully connected network. Persistent homology tracks which topological "
          "features appear and disappear during this growth.") +
        p("A connected component appears when two isolated points first become connected. "
          "A loop appears when three or more points form a cycle without filling in. "
          "Both features eventually disappear: components merge into larger components, "
          "loops fill in with triangles. The 'birth' and 'death' of each feature, "
          "recorded as a (birth radius, death radius) pair, forms the persistence diagram. "
          "Features with large persistence (death minus birth is large) are structurally meaningful. "
          "Features with small persistence are noise.")
    ),

    "A.2 Granger Causality F-Test": box(
        p("The Granger causality test fits two linear regressions. The restricted model "
          "predicts Asset Y using only Y's own past values. The unrestricted model "
          "predicts Y using both Y's own past and Asset X's past. "
          "If the unrestricted model fits significantly better, X Granger-causes Y.") +
        p("The F-statistic compares the residual sum of squares of the two models. "
          "A large F-statistic, corresponding to a small p-value, rejects the null hypothesis "
          "that X's past adds no predictive power. The threshold for declaring Granger causality "
          "is typically p less than 0.05, adjusted for multiple comparisons when "
          "running all N-squared pairs simultaneously.")
    ),

    "A.4 HJB Optimal Stopping": box(
        p("The HJB equation for optimal stopping says that the value of holding a position "
          "at state x and time t must equal the maximum of two alternatives: "
          "the value of stopping immediately (taking the current profit), "
          "or the value of waiting one more instant (the expected future value discounted back).") +
        p("Solving this equation backwards in time from the terminal condition gives a "
          "stopping boundary: a curve in the (position, time) space that separates "
          "'keep holding' from 'exit now.' Any position that crosses this boundary "
          "should be closed. The boundary is not fixed: it shifts as drift and volatility "
          "estimates are updated, making it a dynamic exit rule rather than a static stop-loss.")
    ),

    "A.6 Multifractal DFA Legendre Transform": box(
        p("MF-DFA estimates how the variance of residuals scales with the length of the "
          "analysis window for different statistical moments (the q-values). "
          "For a simple random walk, all moments scale with the same exponent: "
          "the scaling is monofractal. For a multifractal process, different moments "
          "scale with different exponents, revealing that the process behaves differently "
          "at different amplitudes.") +
        p("The Legendre transform converts the scaling exponent function h(q) into the "
          "multifractal spectrum f(alpha): a curve describing how many different "
          "'flavors' of scaling are present in the data. A wide f(alpha) curve means "
          "the process has rich multifractal structure. A narrow curve approaching a "
          "spike means the process is becoming monofractal, which in the context of "
          "market returns is associated with crisis regimes where everything moves "
          "in a single dominant pattern.")
    ),

    "A.7 Transfer Entropy via Histogram Estimator": box(
        p("Transfer entropy from X to Y is defined as the conditional mutual information "
          "between X's past and Y's future, given Y's own past. "
          "Computing it requires estimating joint probability distributions over "
          "continuous variables, which is done here with a histogram: "
          "the state space is divided into bins and the probability of each bin "
          "is estimated by the frequency of observations falling into it.") +
        p("The number of bins is the critical tuning parameter. Too few bins and "
          "the distribution is over-smoothed: true structure is washed out. "
          "Too many bins and most bins have zero observations: the estimate is noisy. "
          "The default of 10 bins per dimension is a practical compromise for "
          "typical financial return distributions with 500 observations.")
    ),

    "A.8 Ollivier-Ricci Curvature Spectral Proxy": box(
        p("Computing exact Ollivier-Ricci curvature on a large graph requires solving "
          "an optimal transport problem for every edge, which is computationally expensive. "
          "The spectral proxy used here approximates it using the second smallest eigenvalue "
          "of the graph Laplacian, sometimes called the Fiedler value or algebraic connectivity.") +
        p("The Fiedler value measures how difficult it is to cut a graph into two disconnected "
          "pieces: a high Fiedler value means the graph is well-connected and hard to split, "
          "which corresponds to positive (crowded) curvature. "
          "A low Fiedler value means the graph is close to disconnecting, "
          "corresponding to negative (sparse) curvature. "
          "This proxy captures the curvature signal at a fraction of the computational cost.")
    ),

    "A.9 PageRank on Weighted Directed Graphs": box(
        p("PageRank assigns each node a score based on the scores of the nodes that link to it. "
          "A node linked to by many high-score nodes gets a high score itself. "
          "The algorithm is iterative: start with equal scores, propagate scores along edges, "
          "and repeat until the scores converge.") +
        p("On a weighted directed Granger causality graph, PageRank identifies assets "
          "that are 'caused by' many other important assets: they are downstream recipients "
          "of causal influence from across the network. "
          "Paradoxically, high PageRank in this context means high systemic risk: "
          "the asset absorbs shocks from many directions simultaneously, "
          "making it likely to move violently in a coordinated market stress event.")
    ),

    "A.10 Betweenness Centrality": box(
        p("Betweenness centrality measures how often a node sits on the shortest path "
          "between two other nodes. A node with high betweenness is a bridge: "
          "remove it, and many other pairs of nodes can no longer communicate efficiently.") +
        p("In a financial contagion network, a high-betweenness node is a transmission hub: "
          "stress that would normally stay contained in one sector spreads through the "
          "hub to every other sector it connects. "
          "Identifying these hubs before a crisis is the goal: "
          "monitoring or hedging the hub's exposure can interrupt the contagion pathway "
          "before it reaches the rest of the network.")
    ),

    "A.12 Beta Distribution Bayesian Update": box(
        p("The Beta distribution is the natural probability model for a rate or proportion "
          "between zero and one. In the Bayesian Debate System, each agent's credibility "
          "score is modelled as a Beta distribution with parameters alpha (number of correct "
          "predictions) and beta (number of incorrect predictions).") +
        p("After each round, if the agent's prediction was correct, alpha increases by one; "
          "if incorrect, beta increases by one. The agent's effective credibility weight "
          "is the mean of its Beta distribution: alpha divided by (alpha plus beta). "
          "This is a principled, automatic way of tracking track records: "
          "a new agent with no history starts at 0.5 (equal chance of being right or wrong), "
          "and its weight converges toward its true accuracy rate over time as evidence accumulates.")
    ),
}

print("Reading HTML...", flush=True)
with open(IN_HTML, "r", encoding="utf-8") as f:
    html = f.read()

print("Injecting extended plain-English boxes...", flush=True)
injected = 0
for heading_text, callout in HEADING_CALLOUTS.items():
    pattern = r'(<h[23][^>]*>' + re.escape(heading_text) + r'</h[23]>)'
    new_html, n = re.subn(pattern, r'\1' + callout, html, count=1)
    if n:
        html = new_html
        injected += 1
    else:
        print(f"  [MISS] {heading_text[:60]}")

print(f"  {injected}/{len(HEADING_CALLOUTS)} boxes injected.", flush=True)

print("Writing HTML...", flush=True)
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)
print(f"  {OUT_HTML}", flush=True)

print("Converting to PDF with Playwright...", flush=True)
try:
    from playwright.sync_api import sync_playwright
    file_url = "file:///" + IN_HTML.replace("\\", "/")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(file_url, wait_until="networkidle", timeout=180000)
        page.pdf(
            path=OUT_PDF,
            format="Letter",
            margin={"top": "1in", "bottom": "1in", "left": "1in", "right": "1in"},
            print_background=True,
        )
        browser.close()
    size_mb = os.path.getsize(OUT_PDF) / 1_048_576
    print(f"  PDF written: {OUT_PDF}  ({size_mb:.1f} MB)", flush=True)
except Exception as e:
    print(f"  PDF error: {e}", flush=True)

# Copy to desktop
import shutil
DESK = r"C:/Users/Matthew/Desktop"
shutil.copy(OUT_HTML, os.path.join(DESK, "event_horizon_book_oxidized.html"))
shutil.copy(OUT_PDF,  os.path.join(DESK, "event_horizon_book.pdf"))
print("Copied to Desktop.", flush=True)
print("Done.", flush=True)
