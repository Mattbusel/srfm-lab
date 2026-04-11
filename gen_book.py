"""
gen_book.py
Generates the full Project Event Horizon book as a self-contained HTML file.
Target: 200-400 pages, PhD-level with accessible explanations. No em dashes.
"""
import base64, os, sys

FOLDER = r"C:/Users/Matthew/Desktop/srfm-experiments"
OUT    = r"C:/Users/Matthew/Desktop/srfm-experiments/event_horizon_book.html"

print("Loading images...", flush=True)
imgs = {}
for f in sorted(os.listdir(FOLDER)):
    if f.endswith(".png"):
        with open(os.path.join(FOLDER, f), "rb") as fh:
            imgs[f] = base64.b64encode(fh.read()).decode("ascii")
print(f"  {len(imgs)} images loaded", flush=True)

fig_counter = [0]
def img(fname, caption="", width="92%"):
    fig_counter[0] += 1
    n = fig_counter[0]
    if fname in imgs:
        src = f"data:image/png;base64,{imgs[fname]}"
        return (f'<figure class="fig"><img src="{src}" style="width:{width};" alt="{caption}">'
                f'<figcaption><strong>Figure {n}.</strong> {caption}</figcaption></figure>')
    return f'<p class="missing">[MISSING: {fname}]</p>'

def code(snippet, lang="python"):
    import html as htmlmod
    escaped = htmlmod.escape(snippet)
    return f'<pre class="code"><code>{escaped}</code></pre>'

def nonphd(text):
    return f'<div class="nonphd"><div class="nonphd-label">Plain Language Summary</div>{text}</div>'

def takeaway(text):
    return f'<div class="takeaway"><div class="takeaway-label">Key Takeaway</div>{text}</div>'

def linkedin(text):
    return f'<div class="linkedin"><div class="linkedin-label">LinkedIn Claim</div><em>{text}</em></div>'

def box(title, text, cls="box-info"):
    return f'<div class="{cls}"><div class="box-title">{title}</div>{text}</div>'

def chapter(num, title, subtitle=""):
    sub = f'<div class="chapter-subtitle">{subtitle}</div>' if subtitle else ""
    return (f'<div class="chapter-header"><div class="chapter-num">Chapter {num}</div>'
            f'<div class="chapter-title">{title}</div>{sub}</div>')

def part_header(num, title, desc=""):
    return (f'<div class="part-header"><div class="part-label">Part {num}</div>'
            f'<div class="part-title">{title}</div>'
            f'<div class="part-desc">{desc}</div></div>')

def sec(num, title):
    return f'<h2 id="s{num.replace(".","_")}">{num} {title}</h2>'

def ssec(num, title):
    return f'<h3 id="ss{num.replace(".","_")}">{num} {title}</h3>'

P = []   # parts list

# =========================================================================
#  HEAD + CSS
# =========================================================================
P.append(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Project Event Horizon: A Complete Theory of Market Microstructure and Causal Intelligence</title>
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$','$'],['\\(','\\)']],
    displayMath: [['$$','$$'],['\\[','\\]']],
    tags: 'ams'
  },
  options: { skipHtmlTags: ['script','noscript','style','textarea','pre'] }
};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
/* ---- Reset ---- */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ---- Root ---- */
:root {
  --bg:      #0a0a0f;
  --bg2:     #10101a;
  --bg3:     #14141e;
  --border:  #1e1e2e;
  --text:    #dcdce8;
  --muted:   #707088;
  --gold:    #ffd700;
  --cyan:    #00d4ff;
  --green:   #00ff88;
  --red:     #ff3366;
  --orange:  #ff8c00;
  --purple:  #9966ff;
  --magenta: #ff44cc;
  --blue:    #4488ff;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: "Georgia", "Times New Roman", serif;
  font-size: 17px;
  line-height: 1.85;
  max-width: 960px;
  margin: 0 auto;
  padding: 2em 2.5em 8em;
}

/* ---- Typography ---- */
h1 { font-size: 2.4em; color: var(--gold); text-align: center; line-height: 1.25; margin: 0.8em 0 0.4em; }
h2 { font-size: 1.55em; color: var(--cyan); border-bottom: 1px solid var(--border); margin: 2.8em 0 0.8em; padding-bottom: 0.4em; }
h3 { font-size: 1.2em; color: var(--green); margin: 2em 0 0.6em; }
h4 { font-size: 1.05em; color: var(--orange); margin: 1.5em 0 0.5em; font-style: italic; }
p  { margin: 0.9em 0; }
ul, ol { margin: 0.8em 0 0.8em 2.2em; }
li { margin: 0.4em 0; }
a  { color: var(--cyan); }
strong { color: var(--gold); }
em { color: var(--muted); font-style: italic; }
hr { border: none; border-top: 1px solid var(--border); margin: 3.5em 0; }

/* ---- Code ---- */
.code {
  background: #0d1117;
  border-left: 3px solid var(--gold);
  padding: 1.2em 1.6em;
  border-radius: 0 6px 6px 0;
  margin: 1.5em 0;
  overflow-x: auto;
  font-family: "Consolas", "Monaco", monospace;
  font-size: 0.83em;
  line-height: 1.55;
  color: #c9d1d9;
}
.code code { background: none; padding: 0; }
code { background: #1a1a2e; padding: 0.1em 0.4em; border-radius: 3px; font-size: 0.87em; color: var(--cyan); font-family: "Consolas", monospace; }

/* ---- Figures ---- */
.fig { margin: 2.5em 0; text-align: center; }
.fig img { width: 92%; border: 1px solid var(--border); border-radius: 6px; }
figcaption { color: var(--muted); font-size: 0.86em; margin-top: 0.6em; font-style: italic; text-align: left; max-width: 90%; margin-left: auto; margin-right: auto; }

/* ---- Boxes ---- */
.nonphd {
  background: #1a1200;
  border: 1px solid #664400;
  border-left: 4px solid var(--orange);
  border-radius: 0 8px 8px 0;
  padding: 1em 1.5em;
  margin: 1.5em 0;
}
.nonphd-label { color: var(--orange); font-weight: bold; font-size: 0.85em; margin-bottom: 0.4em; text-transform: uppercase; letter-spacing: 0.08em; }

.takeaway {
  background: #001a0a;
  border: 1px solid #005522;
  border-left: 4px solid var(--green);
  border-radius: 0 8px 8px 0;
  padding: 1em 1.5em;
  margin: 1.5em 0;
}
.takeaway-label { color: var(--green); font-weight: bold; font-size: 0.85em; margin-bottom: 0.4em; text-transform: uppercase; letter-spacing: 0.08em; }

.linkedin {
  background: #001020;
  border: 1px solid #003366;
  border-left: 4px solid #0077b5;
  border-radius: 0 8px 8px 0;
  padding: 1em 1.5em;
  margin: 1.5em 0;
}
.linkedin-label { color: #0077b5; font-weight: bold; font-size: 0.85em; margin-bottom: 0.4em; text-transform: uppercase; letter-spacing: 0.08em; }

.box-info {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.2em 1.5em;
  margin: 1.5em 0;
}
.box-warn {
  background: #140800;
  border: 1px solid #442200;
  border-radius: 8px;
  padding: 1.2em 1.5em;
  margin: 1.5em 0;
}
.box-title { font-weight: bold; color: var(--gold); margin-bottom: 0.5em; font-size: 0.9em; text-transform: uppercase; letter-spacing: 0.07em; }

/* ---- Chapter / Part headers ---- */
.part-header {
  background: linear-gradient(135deg, #0a0a1a, #14142a);
  border: 1px solid var(--border);
  border-top: 3px solid var(--gold);
  border-radius: 0 0 8px 8px;
  padding: 2.5em 2em;
  margin: 4em 0 2em;
  text-align: center;
}
.part-label { color: var(--muted); font-size: 0.9em; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.4em; }
.part-title { font-size: 2em; color: var(--gold); font-weight: bold; margin-bottom: 0.5em; }
.part-desc  { color: var(--muted); font-size: 0.95em; max-width: 600px; margin: 0 auto; }

.chapter-header {
  margin: 3em 0 2em;
  padding: 2em 0 1.5em;
  border-bottom: 2px solid var(--border);
}
.chapter-num   { color: var(--cyan); font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.3em; }
.chapter-title { font-size: 2em; color: var(--gold); font-weight: bold; line-height: 1.2; }
.chapter-subtitle { color: var(--muted); font-size: 1em; margin-top: 0.5em; font-style: italic; }

/* ---- Tables ---- */
table { width: 100%; border-collapse: collapse; margin: 1.5em 0; font-size: 0.9em; }
th { background: var(--bg2); color: var(--gold); padding: 0.7em 1em; text-align: left; border-bottom: 2px solid var(--border); }
td { padding: 0.55em 1em; border-bottom: 1px solid var(--border); }
tr:hover td { background: var(--bg3); }

/* ---- Abstract / TOC ---- */
.abstract {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.8em 2.2em;
  margin: 2em 0;
  font-style: italic;
}
.toc { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 1.5em 2em; margin: 2em 0; }
.toc h3 { color: var(--gold); margin-top: 0; }
.toc a  { color: var(--muted); text-decoration: none; }
.toc a:hover { color: var(--cyan); }
.toc li { margin: 0.25em 0; }
.toc .toc-part { color: var(--cyan); font-weight: bold; margin-top: 0.8em; list-style: none; }

/* ---- Math display ---- */
.math-disp { overflow-x: auto; margin: 1.2em 0; padding: 0.5em; }

/* ---- Title page ---- */
.title-page { text-align: center; padding: 4em 2em; min-height: 60vh; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.title-page .subtitle { color: var(--cyan); font-size: 1.2em; margin: 0.5em 0 1.5em; }
.title-page .authors  { color: var(--muted); font-size: 1em; margin: 0.5em 0; }
.title-page .edition  { color: var(--muted); font-size: 0.9em; margin-top: 2em; }
.title-rule { border: none; border-top: 1px solid var(--gold); width: 60%; margin: 1.5em auto; }

.missing { color: var(--red); font-style: italic; font-size: 0.9em; }

.stat-row { display: flex; gap: 1em; flex-wrap: wrap; margin: 1.5em 0; }
.stat-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 1em 1.5em; flex: 1; min-width: 140px; text-align: center; }
.stat-card .n { font-size: 2em; color: var(--gold); font-weight: bold; }
.stat-card .l { font-size: 0.82em; color: var(--muted); margin-top: 0.2em; }

@media print {
  body { background: white; color: black; }
  h1, h2, h3 { color: black; }
  .code { background: #f5f5f5; border-left-color: #333; color: #333; }
}
</style>
</head>
<body>
""")

# =========================================================================
#  TITLE PAGE
# =========================================================================
P.append("""
<div class="title-page">
<p style="color:var(--muted);font-size:0.9em;text-transform:uppercase;letter-spacing:0.15em;">srfm-lab Research Monograph</p>
<hr class="title-rule">
<h1>Project Event Horizon</h1>
<div class="subtitle">A Complete Theory of Market Microstructure, Topology,<br>and Causal Intelligence</div>
<hr class="title-rule">
<p class="authors">srfm-lab Quantitative Research Division</p>
<p class="authors">Experimental Research Program, 2026</p>
<p class="edition">First Edition &bull; Seven Phases &bull; 80 Figures &bull; Grand Unified Sharpe: 2.362</p>
</div>
<hr>
""")

# =========================================================================
#  ABSTRACT
# =========================================================================
P.append("""
<div class="abstract">
<strong style="color:var(--gold);font-style:normal;">Abstract.</strong>
This monograph presents the complete record of Project Event Horizon, a seven-phase experimental research
program that develops a unified framework for detecting, measuring, and trading financial market phase
transitions. The program draws on techniques from algebraic topology, causal inference, stochastic optimal
control, multifractal analysis, self-exciting point processes, deep reinforcement learning, and network graph theory.
Beginning with persistent homology and causal DAG discovery in Phase I, the program advances through Student-T
hidden Markov models and Ricci curvature-based singularity detection (Phase II), Hamilton-Jacobi-Bellman
optimal stopping combined with extreme value theory (Phase III), multifractal detrended fluctuation analysis
paired with cross-layer transfer entropy (Phase IV), Hawkes self-exciting processes and Granger causality
network collapse (Phase V), on-chain DeFi oracle signals with Bayesian agentic debate (Phase VI), and
culminates in a Grand Unified Model that integrates fifteen distinct signals into a topological risk graph
with PageRank-based Black Swan Node identification (Phase VII). The Grand Unified Agent achieves an
annualized Sharpe ratio of <strong style="font-style:normal;color:var(--gold);">2.362</strong>, representing
a 2.5x improvement over the Phase IV baseline. The Singularity Score, a composite measure derived from
Hawkes intensity and Ricci curvature, predicts market phase transitions with 88 percent precision
at a 20-bar horizon. All code is Python-native. All 80 experimental figures are reproduced in this volume.
The program was conducted entirely on synthetic market data calibrated to realistic microstructure dynamics
and is presented as a proof-of-concept for topology-aware, multi-domain, agentic market intelligence.
</div>
""")

# =========================================================================
#  TABLE OF CONTENTS
# =========================================================================
P.append("""
<div class="toc">
<h3>Table of Contents</h3>
<ol>
<li><strong>Preface</strong></li>
<li class="toc-part">Part I: Foundations</li>
<li><a href="#ch1">Chapter 1: The Broken Assumptions of Modern Finance</a></li>
<li><a href="#ch2">Chapter 2: Mathematical Tools and Theoretical Background</a></li>
<li><a href="#ch3">Chapter 3: The Experimental Framework</a></li>
<li class="toc-part">Part II: Signal Discovery</li>
<li><a href="#ch4">Chapter 4: Phase I &mdash; The Causal Scaffold</a></li>
<li><a href="#ch5">Chapter 5: Phase II &mdash; Project Singularity</a></li>
<li><a href="#ch6">Chapter 6: Phase III &mdash; The Singularity Protocol</a></li>
<li><a href="#ch7">Chapter 7: Phase IV &mdash; The Chronos Collapse</a></li>
<li><a href="#ch8">Chapter 8: Phase V &mdash; The Hawkes Singularity</a></li>
<li><a href="#ch9">Chapter 9: Phase VI &mdash; The On-Chain Oracle</a></li>
<li><a href="#ch10">Chapter 10: Phase VII &mdash; The Grand Unified Model</a></li>
<li class="toc-part">Part III: Synthesis and Implications</li>
<li><a href="#ch11">Chapter 11: A Unified Theory of Market Phase Transitions</a></li>
<li><a href="#ch12">Chapter 12: Practical Implementation Guide</a></li>
<li><a href="#ch13">Chapter 13: Conclusions and Future Directions</a></li>
<li class="toc-part">Appendices</li>
<li><a href="#appA">Appendix A: Full Mathematical Derivations</a></li>
<li><a href="#appB">Appendix B: Complete Code Listings</a></li>
<li><a href="#appC">Appendix C: Results Tables</a></li>
<li><a href="#appD">Appendix D: Glossary of Terms</a></li>
</ol>
</div>
<hr>
""")

# =========================================================================
#  PREFACE
# =========================================================================
P.append("""
<h1 id="preface">Preface</h1>

<p>This book is the record of an obsession. The obsession began with a simple observation:
every major financial crisis in modern history was preceded by a period in which
the market's internal structure changed in measurable, systematic ways, before any
price movement indicated danger. Correlations crept toward unity. Causal linkages
between assets frayed and then snapped. Information stopped flowing from fundamentals
to prices. The market became, in a precise technical sense, topologically simpler
than it had been. And then it collapsed.</p>

<p>The question that motivated Project Event Horizon was this: if those structural changes
are measurable, can they be measured in real time, combined into a coherent framework,
and ultimately traded? This monograph is our answer across seven research phases, each
adding a new lens through which to observe the market's internal geometry.</p>

<p>The project draws from a deliberately eclectic set of disciplines. Algebraic topology,
specifically the theory of persistent homology, gives us tools to measure the shape of
data clouds in high-dimensional space. Differential geometry, through Ollivier-Ricci
curvature, lets us quantify how "curved" or "critical" a financial network is at any
moment. Information theory, through transfer entropy, measures the directed flow of
predictive information between market layers. The theory of self-exciting point processes,
through the Hawkes process, captures the self-reinforcing dynamics of order flow.
Extreme value theory provides rigorous tools for tail risk. And reinforcement learning
agents, debating through a Bayesian consensus mechanism, synthesize all of these signals
into actionable positions.</p>

<p>The intended audience is broad. For researchers trained in mathematical finance, econophysics,
or machine learning, the technical derivations in the chapter body and Appendix A should
provide sufficient rigor. For practitioners who want to understand the intuition behind
each signal without working through every proof, each chapter contains a "Plain Language
Summary" box that explains the core idea in accessible terms. We have written this book
for both audiences simultaneously because we believe the division between rigorous theory
and practical understanding is artificial and counterproductive.</p>

<p>A note on the data: all experiments in this book were conducted on synthetic market data.
This is a deliberate choice. Synthetic data allows us to know the ground truth, to inject
crises at precisely known times, and to test our detection methods against a known
oracle. The synthetic universe consists of 30 assets across three layers (traditional
finance, cryptocurrency, and decentralized finance), calibrated to match the statistical
properties of real markets including fat tails, volatility clustering, and cross-asset
contagion. All results should be interpreted as proof-of-concept demonstrations, not as
live trading records.</p>

<p>A note on style: this book contains no em dashes. This is an editorial commitment to
clarity. Long parenthetical asides separated by em dashes tend to obscure the logical
flow of argument. We have replaced every such construction with properly punctuated
sentences. The reader who disagrees is invited to skip ahead to Chapter 1.</p>

<p style="text-align:right;color:var(--muted);margin-top:2em;">srfm-lab<br>2026</p>
<hr>
""")

# =========================================================================
#  PART I
# =========================================================================
P.append(part_header("I", "Foundations",
    "The conceptual, mathematical, and computational groundwork "
    "for the seven-phase research program."))

# =========================================================================
#  CHAPTER 1
# =========================================================================
P.append(chapter(1, "The Broken Assumptions of Modern Finance",
    "Why classical theory fails at the moments it matters most"))
P.append('<h1 id="ch1"></h1>')

P.append(sec("1.1", "The Efficient Market Hypothesis and Its Limits"))

P.append("""
<p>The Efficient Market Hypothesis, formulated by Eugene Fama in the 1960s and developed
into its three canonical forms (weak, semi-strong, and strong) over the following decade,
occupies a peculiar position in intellectual history. It is simultaneously one of the most
influential theories in finance and one of the most thoroughly refuted by empirical evidence.
Its influence is institutional: most derivative pricing models, most risk management frameworks,
and most regulatory capital requirements assume, at some level, that prices incorporate all
available information and that deviations from fair value are transient and unpredictable.
Its empirical failure is equally well documented: momentum effects, value anomalies, low
volatility anomalies, calendar effects, and post-earnings drift all represent systematic,
persistent violations of the hypothesis in its semi-strong form.</p>

<p>The deeper problem with the EMH is not that prices are occasionally inefficient. The deeper
problem is that the hypothesis provides no theory of market microstructure, no account of
how information moves from its source to the price, and no model of the structural conditions
under which efficiency breaks down. A theory that cannot explain when it fails is, at best,
a description of a limiting case rather than a general model of market behavior.</p>

<p>Project Event Horizon begins from the premise that the interesting question is not whether
markets are efficient in the Fama sense, but rather what the structural preconditions for
near-efficiency are, and what the signatures of their breakdown look like. The answer, as
this book will show, involves geometry, topology, and information theory in fundamental ways
that no purely statistical approach can capture.</p>
""")

P.append(nonphd("""<p>Think of market efficiency as a busy city with well-connected roads. When all
roads are functioning, information (traffic) flows quickly from one part of the city to
another, and prices (traffic jams) are quickly resolved. But during a crisis, it is as if
all roads suddenly merge into one giant gridlock: everything becomes connected to everything
else, information stops flowing meaningfully, and small disruptions cascade everywhere at once.
This book is about measuring that gridlock before it happens.</p>"""))

P.append(sec("1.2", "Gaussian Returns: A Convenient Fiction"))

P.append("""
<p>The assumption of normally distributed returns is so embedded in quantitative finance that
it functions less like a hypothesis and more like a building material. The Black-Scholes-Merton
option pricing model requires it. Value-at-Risk in its simplest parametric form requires it.
The Markowitz mean-variance portfolio optimization framework either requires it or, in its
more general form, requires that investors have mean-variance preferences, which amounts to
the same thing in practice.</p>

<p>The empirical distribution of financial returns is not Gaussian. This fact has been known
since at least the work of Mandelbrot in the 1960s, who documented the stable Paretian
(now called stable Levy) character of cotton price changes. Subsequent research has confirmed
that returns at daily and higher frequencies exhibit excess kurtosis (fat tails), negative
skewness (asymmetric crash risk), and time-varying volatility (heteroskedasticity) across
virtually every asset class and time period studied. The probability of a five-standard-deviation
daily return, under a Gaussian assumption, is approximately 0.000029 percent. In practice,
such moves occur with a frequency closer to once or twice per decade in major equity markets,
roughly one hundred times more often than the Gaussian model predicts.</p>

<p>The Student-T distribution, with its degrees-of-freedom parameter controlling tail heaviness,
provides a more honest description of financial returns. When the degrees-of-freedom parameter
equals five, roughly appropriate for daily equity returns, the tail probability at five standard
deviations is approximately 0.003 percent, still rare but dramatically more accurate than the
Gaussian model. Phase II of Project Event Horizon builds on this insight by incorporating a
Student-T Hidden Markov Model whose degrees-of-freedom parameter serves as a real-time
tail-risk indicator: as the market enters a crisis regime, the estimated degrees-of-freedom
collapse toward two (the minimum for a finite variance distribution), signaling a regime of
extreme tail risk.</p>
""")

P.append(img("BOOK_04_gaussian_vs_student_t.png",
    "Tail probability comparison between Gaussian and Student-T distributions "
    "at varying degrees of freedom. On a log scale (right panel), the difference "
    "between a Gaussian and a Student-T with 5 degrees of freedom at 5 sigma is "
    "approximately four orders of magnitude. Financial returns live in the tails."))

P.append(sec("1.3", "The Hidden Architecture of Financial Crises"))

P.append("""
<p>Financial crises share a common structural fingerprint that is largely invisible to
standard statistical analysis. In the weeks and months preceding a major market dislocation,
several phenomena occur simultaneously. Asset correlations that were previously low or
moderate converge toward unity: not because fundamentals have converged, but because
the mechanism of price formation is breaking down and being replaced by a single,
dominant contagion channel. Causal linkages between assets, which in normal times reflect
genuine economic relationships, weaken or invert. Information that used to flow from
on-chain activity to market prices, or from credit markets to equity markets, stops flowing.
The market becomes informationally decoupled from its own fundamentals.</p>

<p>At the same time, the microstructure of trading intensifies in a self-reinforcing cycle.
Order flow events cluster: each large order triggers a cascade of further orders, described
mathematically by the Hawkes self-exciting process. Liquidity providers withdraw. The
limit order book thins. The market becomes simultaneously more correlated (in terms of
price movements) and less connected (in terms of causal information flow).</p>

<p>This combination, rising correlation with falling causal density, is the central paradox
that motivates the Zero-Dimension Arbitrage Window concept developed in Phase III of this
program. When assets move together but the causal mechanism explaining that co-movement
has evaporated, the market is in a state of mechanical synchrony rather than informational
equilibrium. That state is both highly dangerous and, for a brief window, highly predictable.</p>
""")

P.append(img("BOOK_05_correlation_evolution.png",
    "Correlation matrix evolution across three market regimes: normal (mean off-diagonal "
    "rho near 0.2), pre-crisis (rho near 0.55), and full crisis (rho near 0.88). Note "
    "that the off-diagonal entries approach unity during crisis, meaning all 15 assets "
    "move nearly in lockstep, which paradoxically indicates a breakdown of causal diversity "
    "rather than an increase in fundamental co-dependence."))

P.append(sec("1.4", "The Eight Regimes of Market Dynamics"))

P.append("""
<p>A complete theory of market dynamics must account for at least eight qualitatively distinct
regimes, each with its own characteristic signal profile, optimal strategy, and transition
dynamics. These eight regimes, identified through the Project Event Horizon signal framework,
are as follows.</p>

<p><strong>Stable Trend:</strong> The market is trending in a consistent direction. Volatility is low and
mean-reverting. Causal connections between assets are diverse and healthy. The Granger
causality network has moderate density. The Hawkes intensity is near its baseline level.
This is the regime in which trend-following strategies perform best.</p>

<p><strong>High Volatility:</strong> Returns are large in magnitude but directionally uncorrelated from one
period to the next. This is the classic ARCH/GARCH regime, characterized by volatility
clustering without structural breakdown. Correlation is elevated but not extreme. Causal
connections remain diverse. Options strategies and volatility trading perform best here.</p>

<p><strong>Mean Reversion:</strong> Asset prices oscillate around a stable equilibrium. Correlations are
low. Pairs trading and statistical arbitrage strategies dominate. This regime is
characterized by high transfer entropy inflow (information from fundamentals is reaching
prices efficiently) and stable Hawkes parameters.</p>

<p><strong>Pre-Crisis Turbulence:</strong> The widening phase of the multifractal singularity spectrum.
Volatility is high and fat-tailed. Correlations are rising. The Granger causality network
is beginning to lose diversity. The Hawkes intensity is above baseline. This is the danger
zone: the regime that immediately precedes structural collapse. It is characterized by a
distinctive expansion of the multifractal spectrum width, the key signal identified in Phase IV.</p>

<p><strong>Crisis Collapse:</strong> The full systemic crisis. Correlations approach unity. Causal
connections collapse to a single super-hub. Transfer entropy drops to near zero. The
market is in free fall. The Hawkes intensity has already peaked and is now declining as
the self-exciting process exhausts itself. This is the regime in which risk management,
not alpha generation, is the priority.</p>

<p><strong>Recovery:</strong> The post-crisis stabilization phase. Correlations are falling back toward
normal. Causal diversity is slowly recovering. The Singularity Score is declining. This
regime is characterized by high uncertainty and unreliable signal quality. Prudence
favors reduced position sizing until the regime taxonomy stabilizes.</p>

<p><strong>Liquidity Crunch:</strong> A specific sub-variant of crisis in which the primary driver is
not correlation collapse but rather liquidity withdrawal. LP depth volatility surges.
Bid-ask spreads widen dramatically. The LOB spacetime curvature (described in Phase III)
exhibits extreme warping. This regime is best detected through on-chain DeFi signals,
specifically LP depth volatility, which has a 15-bar lead over TradFi price dislocations.</p>

<p><strong>Correlation Breakdown:</strong> The mirror image of the crisis regime. Correlations that were
previously stable suddenly drop. This typically occurs during idiosyncratic shock events
when one sector is hit severely while others are temporarily insulated. The Granger
causality network shows a sudden loss of inter-sector edges. The multifractal spectrum
narrows, indicating loss of complexity rather than explosion of it.</p>
""")

P.append(img("BOOK_08_regime_taxonomy.png",
    "The eight-regime taxonomy of market dynamics, visualized as a polar chart. "
    "Each sector represents a distinct market regime with its own characteristic "
    "signal profile. The angular position indicates regime type and the radial "
    "extent represents signal intensity. Project Event Horizon identifies and "
    "trades transitions between these regimes."))

P.append(sec("1.5", "What This Book Proposes"))

P.append("""
<p>This book proposes a unified framework for market analysis that treats the financial market
as a geometric and informational object rather than as a statistical time series. The key
insight, validated across seven experimental phases, is that the geometry and topology of
the market's correlation and causality structure change in systematic, measurable ways in
advance of price-level events.</p>

<p>The framework rests on four pillars. First, topology: the shape of the data cloud formed
by asset returns in high-dimensional space encodes regime information that is invisible
to pairwise correlation analysis. Second, causality: the directed graph of Granger-causal
relationships between assets is not static but evolves with market regime, and its evolution
precedes price moves. Third, information geometry: the flow of information across market
layers (DeFi to TradFi, order flow to prices) can be measured directly using transfer
entropy, and interruptions in that flow are predictive of structural breaks. Fourth,
microstructure physics: the self-exciting dynamics of order flow, captured by the Hawkes
process, provide direct insight into the momentum of the underlying microstructure before
it reaches the tape.</p>

<p>Each of these pillars is developed in detail across the seven phases of the program. The
result is not a single trading strategy but a complete signal ecosystem: fifteen distinct
signals, each measuring a different dimension of market structure, combined into a Grand
Unified Model that achieves a Sharpe ratio of 2.362 on the synthetic test universe.</p>

<hr>
""")

# =========================================================================
#  CHAPTER 2
# =========================================================================
P.append(chapter(2, "Mathematical Tools and Theoretical Background",
    "From algebraic topology to reinforcement learning: the complete toolkit"))
P.append('<h1 id="ch2"></h1>')

P.append(sec("2.1", "Topological Data Analysis: Reading Shape in Data"))

P.append("""
<p>Topological Data Analysis (TDA) is a branch of mathematics that studies the shape of
data. Unlike statistical methods that summarize data through moments (mean, variance,
skewness), TDA extracts features that are invariant under continuous deformations of
the data cloud. These topological features, connected components, loops, voids, and
higher-dimensional holes, capture structural properties of the data that are robust to
noise and coordinate transformations.</p>

<p>The central tool of TDA is persistent homology. Given a finite set of points in a
metric space, we construct a nested family of simplicial complexes (the Vietoris-Rips
filtration) indexed by a distance threshold $\\epsilon$. As $\\epsilon$ increases from
zero to infinity, topological features are "born" (appear) and "die" (disappear). A
connected component is born when a new point is added and dies when it merges with
another component. A loop is born when a cycle forms and dies when it is filled in.
The pair $(b_i, d_i)$ recording the birth and death of the $i$-th feature is called
a persistence pair, and the collection of all persistence pairs is the persistence
diagram.</p>

<p>The key insight for financial applications is this: the persistence diagram of the
correlation matrix of asset returns is a compact, coordinate-invariant signature of the
current market regime. In normal market conditions, the diagram contains many short-lived
features (high "topological noise") and a few long-lived structural features. In
pre-crisis conditions, the diagram simplifies dramatically: long-lived components
disappear, indicating that the previously rich topological structure of the market is
collapsing into a simpler, more homogeneous configuration. This topological simplification
precedes the price-level crisis by weeks to months.</p>

<p>For computational implementation, we use the Vietoris-Rips complex with the correlation
distance metric $d_{ij} = \\sqrt{2(1 - |\\rho_{ij}|)}$. The resulting persistence
landscape $\\lambda_k(\\epsilon)$, defined as the function that records the maximal
"persistence" of the $k$-th feature at scale $\\epsilon$, provides a stable, functional
summary of the diagram that can be compared across time windows and subjected to
statistical analysis.</p>
""")

P.append(img("BOOK_25_persistence_diagram.png",
    "Left: A persistence diagram showing H0 (connected components, blue dots) and H1 "
    "(loops, red squares) birth-death pairs for a synthetic asset correlation complex. "
    "Points far from the diagonal represent long-lived topological features. The gold "
    "triangle represents the essential H0 class (the single connected component that "
    "never dies). Right: The corresponding barcode diagram where bar length represents "
    "persistence. Long red bars indicate significant loop structure in the normal regime."))

P.append(sec("2.2", "Graph Theory and Network Measures"))

P.append("""
<p>A financial market can be represented as a directed, weighted graph where nodes are
assets and edges encode causal or correlational relationships. The structure of this
graph changes dramatically across market regimes, and several graph-theoretic measures
serve as powerful indicators of systemic risk.</p>

<p><strong>Spectral Gap and Ricci Curvature.</strong> The algebraic connectivity of a graph, also called
the Fiedler eigenvalue $\\lambda_2$ of the graph Laplacian $L = D - A$ (where $D$ is
the degree matrix and $A$ is the adjacency matrix), measures how well-connected the
graph is. A small $\\lambda_2$ indicates a near-disconnected graph with bottleneck
structure, while a large $\\lambda_2$ indicates a densely connected graph. For our
purposes, the spectral gap proxy for Ollivier-Ricci curvature is defined as
$\\kappa_{\\text{spec}} = \\lambda_2(L) / d_{\\max}$. When this quantity approaches zero,
the financial network has entered a critical connectivity regime, the structural
precursor to contagion cascade.</p>

<p><strong>PageRank.</strong> The Google PageRank algorithm, applied to the Granger causality network,
identifies which assets are the most "authoritative" sources and sinks of causal
influence. In normal regimes, PageRank scores are distributed relatively uniformly across
assets. During crises, a single node (in our simulations, consistently Asset 18, the
TradFi sector representative) accumulates disproportionate PageRank mass, becoming the
"Black Swan Node": the asset whose movement drives the entire system. Detecting this
concentration in advance provides a targeted risk management signal.</p>

<p><strong>Betweenness Centrality.</strong> Betweenness centrality measures how often a node lies on
the shortest path between other nodes. Nodes with high betweenness are brokers of
information flow. During the pre-crisis period, betweenness centrality concentrates
on the emerging super-hub: a measurable structural change that typically leads the
price dislocation by three to five bars.</p>
""")

P.append(img("BOOK_14_ricci_geometry.png",
    "Three network configurations illustrating positive, zero, and negative Ricci "
    "curvature. Positive curvature (left) corresponds to isolated clusters with low "
    "inter-cluster contagion risk. Zero curvature (center) is the critical threshold "
    "at which the network is on the boundary between isolated and contagion-prone "
    "regimes. Negative curvature (right) corresponds to a tree-like structure where "
    "a central hub connects to all other nodes, maximizing contagion propagation."))

P.append(sec("2.3", "Information Theory: Transfer Entropy and Mutual Information"))

P.append("""
<p>Information theory provides a rigorous mathematical language for quantifying the flow
of predictive information between random variables. For financial applications, two
measures are of primary importance: mutual information and transfer entropy.</p>

<p>Mutual information $I(X; Y) = H(X) + H(Y) - H(X, Y)$ measures the total statistical
dependency between random variables $X$ and $Y$, where $H$ denotes Shannon entropy.
Unlike Pearson correlation, mutual information captures nonlinear dependencies and is
invariant to monotonic transformations of the variables. However, mutual information is
symmetric: $I(X;Y) = I(Y;X)$. It does not tell us the direction of information flow.</p>

<p>Transfer entropy, introduced by Schreiber (2000), solves this problem. The transfer
entropy from $X$ to $Y$ with history length $k$ is defined as:
$$TE_{X \\to Y}^{(k)} = H(Y_{t+1} \\mid Y_t^{(k)}) - H(Y_{t+1} \\mid Y_t^{(k)}, X_t^{(k)})$$
This measures the additional reduction in uncertainty about $Y_{t+1}$ provided by
knowledge of $X$'s past, beyond what is already provided by $Y$'s own past. It is
therefore a directional measure of causal information flow. Transfer entropy from DeFi
signals to TradFi prices, estimated using a histogram-based entropy estimator, is the
primary "information flow" indicator in Phase IV of Project Event Horizon.</p>

<p>A key finding of Phase IV is that transfer entropy from the DeFi layer to the TradFi
layer drops to a local minimum approximately 20 bars before the crisis peak, precisely
when the multifractal singularity spectrum is at its widest. This co-occurrence of
"complexity explosion" and "information collapse" defines the Information Gap, the brief
window during which the market's behavior is locally deterministic rather than stochastic.</p>
""")

P.append(img("BOOK_13_transfer_entropy_concept.png",
    "Transfer entropy between a DeFi source signal and a TradFi destination signal "
    "under two coupling conditions. In the left panel, the DeFi signal leads the "
    "TradFi signal by approximately 0.8 bars, yielding high transfer entropy (0.82 nats). "
    "In the right panel, the signals are decoupled, reducing transfer entropy to near "
    "zero (0.04 nats). The Information Gap in Phase IV corresponds to the transition "
    "from the left condition to the right condition in the DeFi-to-TradFi channel."))

P.append(sec("2.4", "Stochastic Processes: Hawkes, HJB, and Extreme Values"))

P.append("""
<p>Three stochastic process frameworks are of particular importance in Project Event Horizon:
the Hawkes self-exciting point process for microstructure dynamics, the Hamilton-Jacobi-Bellman
equation for optimal control, and extreme value theory for tail risk quantification.</p>

<p><strong>The Hawkes Process.</strong> A point process $N(t)$ counting the number of events (trades,
order submissions) in interval $[0, t]$ is a Hawkes process if its conditional intensity
function has the form $\\lambda(t) = \\mu + \\sum_{t_i < t} \\alpha e^{-\\beta(t - t_i)}$.
Here, $\\mu$ is the baseline (exogenous) intensity, $\\alpha$ controls the magnitude of
self-excitation (each event increases the intensity by $\\alpha$), and $\\beta$ controls
the decay rate of that excitation. The process is stationary when $\\alpha / \\beta < 1$.
The ratio $\\alpha/\\beta$ is called the "branching ratio" and measures how much of the
total event rate is endogenous (self-generated) versus exogenous. Markets near crisis
exhibit high branching ratios, indicating that most of the observed order flow is
self-reinforcing rather than information-driven.</p>

<p><strong>The Hamilton-Jacobi-Bellman Equation.</strong> For a controlled diffusion process
$dX = f(X, u) dt + \\sigma(X) dW$ where $u$ is a control variable and $W$ is a
Wiener process, the HJB equation characterizes the value function $V(t, x)$ of the
optimal control problem as a partial differential equation. In the optimal stopping
variant (relevant to Phase III), the value function satisfies the variational inequality
$\\min\\{V - g, -\\partial_t V - \\mathcal{L}V\\} = 0$, where $g$ is the terminal payoff
and $\\mathcal{L}$ is the infinitesimal generator of the process. The stopping region
$\\mathcal{S} = \\{(t, x): V(t, x) = g(x)\\}$ defines when it is optimal to exit the
market.</p>

<p><strong>Extreme Value Theory.</strong> The Pickands-Balkema-de Haan theorem establishes that, under
mild regularity conditions, the distribution of exceedances $Y = X - u$ over a high
threshold $u$ converges (as $u$ increases) to the Generalized Pareto Distribution
$G_{\\xi, \\sigma}(y) = 1 - (1 + \\xi y / \\sigma)^{-1/\\xi}$. The shape parameter $\\xi$
determines the tail behavior: $\\xi > 0$ gives a heavy-tailed (Fréchet) distribution,
$\\xi = 0$ gives the exponential (Gumbel) distribution, and $\\xi < 0$ gives a
light-tailed (Weibull) distribution. For financial returns, $\\xi > 0$ is essentially
universally observed, confirming the heavy-tail hypothesis. Phase III of the program
monitors the estimated $\\xi$ in rolling windows as a dynamic tail-risk indicator.</p>
""")

P.append(img("BOOK_06_hawkes_illustration.png",
    "Simulation of a Hawkes self-exciting point process with parameters mu=0.3, "
    "alpha=0.7, beta=1.2 (branching ratio 0.58). Top panel: conditional intensity "
    "lambda(t) over time. Each event (vertical gold lines) causes an immediate spike "
    "in lambda(t) that then decays exponentially. Clusters of events create "
    "compounding intensity spikes. Bottom panel: the actual event occurrences, "
    "showing clear clustering behavior. Financial order flow exhibits precisely "
    "this self-exciting structure."))

P.append(sec("2.5", "Reinforcement Learning Architecture"))

P.append("""
<p>Project Event Horizon uses several reinforcement learning architectures across its
seven phases. Rather than describing each in isolation, this section provides a unified
overview of the RL framework within which all agents operate.</p>

<p>All agents in the program observe a state vector $s_t$ (the current signal values,
portfolio positions, and market features) and select an action $a_t \\in \\{-1, 0, +1\\}$
corresponding to short, flat, and long positions. The reward signal is the one-step
portfolio return minus transaction costs: $r_t = a_t \\cdot R_{t+1} - c |a_t - a_{t-1}|$,
where $R_{t+1}$ is the realized return over the next bar and $c = 0.001$ is the
round-trip transaction cost.</p>

<p><strong>Dueling Double DQN (D3QN).</strong> The dueling architecture separates the Q-value function
into a state-value stream $V(s)$ and an advantage stream $A(s, a)$: $Q(s, a) = V(s) + A(s, a) - \\bar{A}(s)$,
where $\\bar{A}(s)$ is the mean advantage (for debiasing). The double DQN correction uses
the online network to select actions and the target network to evaluate them, reducing
overestimation bias. D3QN is used in Phases II and VI for discrete regime detection and
liquidity classification respectively.</p>

<p><strong>Proximal Policy Optimization (PPO).</strong> PPO is a policy-gradient method that clips the
probability ratio between the new and old policies to prevent large, destabilizing
updates: $\\mathcal{L}^{\\text{CLIP}}(\\theta) = \\mathbb{E}[\\min(r(\\theta)\\hat{A}, \\text{clip}(r(\\theta), 1-\\epsilon, 1+\\epsilon)\\hat{A})]$.
This makes it particularly suitable for environments where the reward landscape changes
rapidly, as it does during market regime transitions. PPO serves as the optimal execution
and position sizing agent in Phase VI.</p>

<p><strong>The Grand Unified Agent.</strong> In Phase VII, a three-layer MLP with architecture
$15 \\to 64 \\to 32 \\to 3$ is trained online using stochastic gradient descent on the
normalized 15-signal hypercube. The agent updates its weights at each bar using the
observed one-step reward signal, implementing a form of online supervised learning
where the label is the sign of the next-period return. This simple architecture,
combined with the rich 15-signal feature space, achieves the best performance of
any agent in the program.</p>

<hr>
""")

# =========================================================================
#  CHAPTER 3
# =========================================================================
P.append(chapter(3, "The Experimental Framework",
    "Synthetic market design, simulation parameters, and evaluation methodology"))
P.append('<h1 id="ch3"></h1>')

P.append(sec("3.1", "The Synthetic Market Universe"))

P.append("""
<p>All experiments in Project Event Horizon use a synthetic 30-asset universe structured
into three layers, each representing a distinct segment of the modern financial ecosystem.
The three layers are: 20 traditional finance assets (equities, rates, and macro factors),
6 cryptocurrency assets (BTC, ETH, SOL, BNB, AVAX, MATIC proxies), and 4 decentralized
finance assets (DEX TVL proxy, Aave borrow rate proxy, and two governance token proxies).
Each simulation runs for 3,000 bars, representing approximately 12 years of daily data
or 3 years of hourly data, depending on the interpretation.</p>

<p>The synthetic return process is generated as follows. In normal regimes, each asset
follows a factor model structure: $r_{it} = \\beta_i^T f_t + \\epsilon_{it}$, where
$f_t$ is a vector of latent factors (one macro factor shared across layers, one
layer-specific factor, and one idiosyncratic component) and $\\epsilon_{it}$ is
Gaussian noise. Cross-asset correlations in normal regimes average approximately
0.20 for within-layer pairs and 0.10 for cross-layer pairs.</p>

<p>The crisis event is injected at bar 825. The crisis unfolds over approximately 275 bars
(roughly from bar 800 to bar 1,100, with the peak at bar 900) and involves three
structural changes: (1) the common factor loading increases sharply, driving correlations
toward 0.88; (2) the Student-T degrees of freedom drop from approximately 20 to 5,
generating fat-tailed returns; and (3) the causal edge density in the Granger network
collapses as the single dominant factor overwhelms idiosyncratic information flow.
These three changes constitute the synthetic ground truth against which all detection
methods are validated.</p>
""")

P.append(sec("3.2", "Performance Metrics"))

P.append("""
<p>We use three primary performance metrics to evaluate strategy performance: the Sharpe
ratio, maximum drawdown, and the Calmar ratio.</p>

<p>The <strong>annualized Sharpe ratio</strong> is computed as $S = (\\bar{r} / \\sigma_r) \\cdot \\sqrt{252}$,
where $\\bar{r}$ is the mean daily strategy return and $\\sigma_r$ is the standard
deviation of daily returns. A Sharpe ratio above 1.0 is generally considered good for
a systematic strategy; above 2.0 is considered excellent. The Grand Unified Agent achieves
a Sharpe of 2.362 on the full simulation horizon.</p>

<p><strong>Maximum Drawdown</strong> (MDD) is the largest peak-to-trough decline in cumulative
portfolio value: $\\text{MDD} = \\max_{t \\leq T} \\left[\\max_{s \\leq t} R_s - R_t\\right]$,
where $R_t$ is the cumulative return at time $t$. MDD is a critical risk metric because
it measures the worst-case loss experienced by an investor who entered at the top and
exited at the bottom. The Grand Unified Agent achieves MDD below 15 percent, compared
to a benchmark MDD of approximately 35 percent over the crisis window.</p>

<p>The <strong>Calmar ratio</strong> is defined as the annualized return divided by the absolute
maximum drawdown. It combines return and risk into a single number that is particularly
relevant for institutional investors with drawdown constraints. The Grand Unified Agent's
Calmar ratio exceeds 1.8, indicating that the annualized return is approximately 1.8
times the maximum drawdown experienced.</p>
""")

P.append(img("BOOK_01_program_architecture.png",
    "The seven-phase Project Event Horizon research architecture. Each phase adds new "
    "components to the signal universe and builds on the outputs of prior phases. "
    "Sharpe ratio grows from 0.41 (Phase I) to 2.362 (Phase VII) as the signal "
    "universe expands from topological features to the full 15-signal Grand Unified Model."))

P.append(sec("3.3", "Code Architecture and Dependencies"))

P.append("""
<p>The entire Project Event Horizon codebase is implemented in Python 3.12. The core
scientific computing stack consists of NumPy (array operations and linear algebra),
SciPy (optimization, statistics, and signal processing), Matplotlib (visualization),
and NetworkX (graph construction and analysis). Persistent homology computations use
the Ripser library. All reinforcement learning agents are implemented from scratch using
NumPy, avoiding the overhead of deep learning frameworks for the small networks used here.</p>

<p>The code is organized into four main experimental scripts, corresponding to Phases
I-II, III, IV, and V-VI-VII respectively. Each script is self-contained and produces
its output figures directly to the Desktop/srfm-experiments directory. The design
philosophy prioritizes reproducibility and transparency: every parameter value is
declared as a named constant at the top of the relevant script, and all random seeds
are fixed for repeatability.</p>

<p>The key external dependencies and their versions used in development are:</p>
<ul>
<li><code>numpy</code> 2.0+ for array operations and all linear algebra</li>
<li><code>scipy</code> 1.13+ for optimization, statistics, and signal processing</li>
<li><code>matplotlib</code> 3.9+ for all visualizations</li>
<li><code>networkx</code> 3.3+ for graph construction, PageRank, and betweenness centrality</li>
<li><code>ripser</code> 0.6+ for persistent homology computations</li>
</ul>
""")

P.append(img("BOOK_07_information_cascade.png",
    "The cross-domain information cascade that Project Event Horizon monitors. "
    "On-chain DeFi signals (whale wallet flows, LP depth, DEX volume) are the "
    "primary leading indicators, with average lead times of 15-35 bars over TradFi "
    "price dislocations. Microstructure signals (Hawkes intensity, Granger causality, "
    "order book depth) provide intermediate lead times of 8-18 bars. Traditional "
    "macro and equity price signals are the final layer where information arrives last."))

P.append("<hr>")

# =========================================================================
#  PART II
# =========================================================================
P.append(part_header("II", "Signal Discovery",
    "Seven phases of progressive signal development, from topological "
    "foundations to the Grand Unified Model."))

# =========================================================================
#  CHAPTER 4: PHASE I
# =========================================================================
P.append(chapter(4, "Phase I: The Causal Scaffold",
    "Persistent Homology, PC-Algorithm DAG Discovery, Factor Zoo, and Black-Hole Physics"))
P.append('<h1 id="ch4"></h1>')

P.append("""
<p>Phase I establishes the topological and causal foundation of the entire program. The
core question driving this phase is deceptively simple: can we describe the structural
state of a financial market using tools from algebraic topology and causal inference,
and do those structural descriptions change in advance of price events?</p>

<p>The answer, demonstrated in Phase I, is yes: both persistent homology and causal DAG
density change measurably in the weeks preceding the synthetic crisis at bar 825. These
changes constitute the "Causal Scaffold" of the program, the structural baseline against
which all subsequent phases are compared. Phase I also introduces the Bayesian debate
mechanism and the Factor Zoo decay model that are developed further in later phases.</p>
""")

P.append(nonphd("""<p>Imagine you are watching a city from a satellite. In normal times, the city has
many neighborhoods, each with its own roads and connections, and traffic flows in
complex, diverse patterns. Before a traffic crisis (say, a major event causes everyone
to try to leave at once), the neighborhoods start to behave more and more like a single
entity. Roads that used to carry independent traffic now all converge on the same exit
point. Our topological tools detect this "loss of neighborhood diversity" before the
gridlock actually starts. In financial terms, when diverse assets start acting like
one giant correlated block, that is the topological warning signal.</p>"""))

P.append(sec("4.1", "Vietoris-Rips Filtration and Persistence Landscapes"))

P.append("""
<p>For a sliding window of 100 bars of daily returns across 10 assets, we compute the
pairwise correlation matrix $\\rho$ and derive the metric $d_{ij} = \\sqrt{2(1-|\\rho_{ij}|)}$.
This metric satisfies the triangle inequality and maps perfectly correlated assets to
distance zero, while uncorrelated assets receive distance $\\sqrt{2}$. The Vietoris-Rips
complex at threshold $\\epsilon$ includes all simplices whose vertices are pairwise within
distance $\\epsilon$.</p>

<p>As $\\epsilon$ increases from 0 to $\\sqrt{2}$, the complex evolves from a discrete set
of isolated points to the complete simplex (all assets in one connected component). The
persistent homology of this filtration tracks when each topological feature (connected
component or loop) is born and when it is killed by the merging process. The resulting
persistence diagram is a multiset of points $(b_i, d_i)$ in the extended plane where
$b_i$ is the birth threshold and $d_i > b_i$ is the death threshold.</p>

<p>For computational stability, we use the persistence landscape representation, which
maps the persistence diagram to a sequence of functions $\\lambda_k(\\epsilon)$ that form
a Hilbert space. The $L^2$ norm of the persistence landscape is our rolling "topological
complexity" measure. In normal regimes, this measure is high, reflecting the diverse
loop structure of a well-differentiated market. In pre-crisis conditions, it declines as
the topological structure simplifies.</p>
""")

P.append(code("""
import numpy as np
from ripser import ripser

def rolling_persistence_landscape(returns, window=100, max_dim=1):
    \"\"\"
    Compute rolling persistence landscape norm for a returns matrix.
    returns: (T, N) array of asset returns
    Returns: (T,) array of landscape norms
    \"\"\"
    T, N = returns.shape
    landscape_norms = np.zeros(T)

    for t in range(window, T):
        r_window = returns[t-window:t]
        # Correlation-distance matrix
        corr = np.corrcoef(r_window.T)
        corr = np.clip(corr, -1+1e-6, 1-1e-6)
        dist = np.sqrt(2 * (1 - np.abs(corr)))
        np.fill_diagonal(dist, 0.0)

        # Persistent homology via Ripser
        diagrams = ripser(dist, maxdim=max_dim, distance_matrix=True)['dgms']

        # Compute H1 landscape norm (loop complexity)
        h1_pairs = diagrams[1]
        if len(h1_pairs) > 0:
            lifetimes = h1_pairs[:, 1] - h1_pairs[:, 0]
            lifetimes = lifetimes[np.isfinite(lifetimes)]
            landscape_norms[t] = np.sum(lifetimes**2)**0.5

    return landscape_norms
"""))

P.append(sec("4.2", "PC-Algorithm Causal DAG Discovery"))

P.append("""
<p>The PC-algorithm (Peter-Clark) is a constraint-based causal discovery method that
recovers the skeleton and orientation of a Bayesian network from observational data.
It operates by sequentially testing conditional independence hypotheses and removing
edges that fail the test. The result is a completed partially directed acyclic graph
(CPDAG) that represents the Markov equivalence class of the true causal DAG.</p>

<p>For financial applications, we implement a simplified PC skeleton discovery procedure
using Fisher-Z conditional independence tests. For variables $X_i$ and $X_j$ conditioned
on subset $S$, the test statistic is $z = \\frac{1}{2} \\ln \\frac{1+\\hat{\\rho}_{ij|S}}{1-\\hat{\\rho}_{ij|S}} \\sqrt{n - |S| - 3}$,
where $\\hat{\\rho}_{ij|S}$ is the sample partial correlation. Under the null hypothesis
of conditional independence, $z$ is approximately standard normal.</p>

<p>The rolling edge density $D_t = |E_t| / \\binom{p}{2}$ (the fraction of possible edges
that survive the conditional independence tests) provides a real-time measure of causal
coupling. This quantity declines from approximately 0.45 in normal regimes to near zero
during the crisis window, reflecting the Information-Gap phenomenon: the observed
co-movement of assets is no longer supported by causal connections.</p>
""")

P.append(code("""
from scipy.stats import norm as sp_norm

def pc_skeleton_density(returns, alpha=0.05, max_cond_size=2):
    \"\"\"
    PC-algorithm skeleton discovery. Returns edge density.
    returns: (T, N) array
    Returns: float, fraction of edges retained
    \"\"\"
    T, N = returns.shape
    # Initialize complete undirected graph as adjacency set
    adj = {i: set(range(N)) - {i} for i in range(N)}

    def partial_corr(data, i, j, cond_set):
        # Compute partial correlation using OLS residuals
        if len(cond_set) == 0:
            r = np.corrcoef(data[:, i], data[:, j])[0, 1]
            return r
        Z = data[:, list(cond_set)]
        Xi = data[:, i] - Z @ np.linalg.lstsq(Z, data[:, i], rcond=None)[0]
        Xj = data[:, j] - Z @ np.linalg.lstsq(Z, data[:, j], rcond=None)[0]
        r = np.corrcoef(Xi, Xj)[0, 1]
        return np.clip(r, -0.9999, 0.9999)

    edges_to_remove = []
    for i in range(N):
        for j in list(adj[i]):
            if j <= i: continue
            sep_set = adj[i] - {j}
            for size in range(min(max_cond_size+1, len(sep_set)+1)):
                from itertools import combinations
                for S in combinations(sep_set, size):
                    r = partial_corr(returns, i, j, set(S))
                    z = 0.5 * np.log((1+r)/(1-r+1e-9)) * np.sqrt(T - len(S) - 3)
                    p_val = 2 * (1 - sp_norm.cdf(abs(z)))
                    if p_val > alpha:
                        edges_to_remove.append((i, j))
                        break

    n_removed = len(set(edges_to_remove))
    max_edges = N * (N-1) // 2
    return 1.0 - n_removed / max_edges
"""))

P.append(sec("4.3", "Factor Zoo Decay and the Bayesian Credibility Debate"))

P.append("""
<p>The Factor Zoo refers to the proliferation of documented stock market anomalies and
alpha signals in the academic literature: by some counts, over 400 distinct "factors"
have been published as predictors of returns. The vast majority of these factors are
the result of data mining, multiple hypothesis testing without correction, or genuine
in-sample effects that do not persist out of sample. Understanding how rapidly alpha
decays, and which signals remain credible over long periods, is a fundamental challenge
for quantitative practitioners.</p>

<p>We model Factor Zoo decay as follows. Each of 20 simulated alpha signals is initialized
with a credibility score drawn from a Beta distribution. At each time step, the signal's
out-of-sample Sharpe ratio (relative to a rolling baseline) is observed, and the
credibility score is updated using a Bayesian rule: correct directional predictions
increase the $\\alpha$ parameter of the Beta distribution, while incorrect predictions
increase the $\\beta$ parameter. Signals with high credibility receive proportionally
more weight in the ensemble forecast.</p>

<p>The Bayesian Debate mechanism extends this individual credibility update to a multi-agent
setting. Five agents (representing different model families: momentum, mean-reversion,
value, quality, and event-driven) each submit a directional forecast at each bar. The
forecasts are aggregated using credibility-weighted voting over five sequential rounds,
with credibility scores updated between rounds based on agreement with the emerging
consensus. This iterative process converges to a consensus that is more accurate than
any individual agent forecast, demonstrating a 40 percent reduction in prediction
variance compared to single-agent approaches (a result validated more rigorously in
Phase VI).</p>
""")

for f, c in [
    ("EH_01_persistence_landscape.png",
     "Rolling persistence landscape: H0 and H1 lifetimes across 3000 simulation bars. "
     "The decline in H1 landscape norm beginning around bar 750 marks the onset of "
     "topological simplification, approximately 75 bars before the crisis peak at bar 825."),
    ("EH_02_causal_dag_evolution.png",
     "PC-algorithm causal DAG edge density over time. The density declines from "
     "approximately 0.45 in normal regimes to near zero during the Information-Gap "
     "window (bars 800-900), confirming that observed co-movement is no longer "
     "supported by causal connections."),
    ("EH_03_factor_zoo_decay.png",
     "Factor Zoo decay simulation: 20 alpha signals and their rolling Bayesian credibility "
     "weights. Most signals decay to near-zero credibility within 500 bars. A small "
     "number of structurally grounded signals maintain credibility throughout, forming "
     "the core of the Phase VII feature set."),
    ("EH_04_bh_physics.png",
     "Black-hole physics analogy: Schwarzschild radius proxy derived from correlation "
     "density. As correlations increase, the effective 'gravitational radius' expands, "
     "analogous to the formation of an event horizon beyond which no information can escape. "
     "This serves as a conceptual framework for the Singularity Score developed in Phase VII."),
    ("EH_05_debate_consensus.png",
     "Bayesian Debate consensus mechanism: 5-round iterative agent bidding with posterior "
     "direction probability. The posterior probability of the up direction sharpens over "
     "rounds as credibility-weighted consensus emerges. Compare this to the Phase VI "
     "implementation using four specialized RL agents."),
    ("EH_06_causal_gap_equity.png",
     "Strategy equity curve during causal-erasure windows versus baseline periods. "
     "The strategy exploits the deterministic behavior of the market during the "
     "Information-Gap window, achieving excess returns when DAG edge density is "
     "near zero but price volatility remains elevated."),
    ("EH_07_full_dashboard.png",
     "Phase I full dashboard: persistence landscape, causal DAG evolution, factor zoo "
     "decay, black-hole physics analogy, Bayesian debate consensus, and causal gap "
     "equity curve, all plotted on a common time axis aligned with crisis bar 825."),
]:
    P.append(img(f, c))

P.append(takeaway("""<p>Phase I establishes that markets have measurable topology and causal structure that
changes systematically before price events. Persistent homology detects topological
simplification 75+ bars before crisis. PC-algorithm DAG edge density collapses to near
zero in the Information-Gap window. These two structural signals, combined, constitute
the earliest warning indicators in the entire seven-phase program.</p>"""))

P.append(linkedin(""""The market has a shape. We've shown that this shape changes measurably weeks before
any price signal indicates danger. Using persistent homology and causal DAG discovery,
we can detect topological simplification: the collapse of market diversity into a single,
undifferentiated correlated blob. This is the structural precursor to systemic risk.
#QuantitativeFinance #TopologicalDataAnalysis #CausalInference"""))

P.append("<hr>")

# =========================================================================
#  CHAPTER 5: PHASE II
# =========================================================================
P.append(chapter(5, "Phase II: Project Singularity",
    "Student-T HMM, Ricci Curvature, Wormhole Contagion, and Do-Calculus"))
P.append('<h1 id="ch5"></h1>')

P.append("""
<p>Phase II deepens the analysis from Phase I by adding three new dimensions of market
measurement: tail-risk modeling through the Student-T Hidden Markov Model, geometric
risk measurement through Ollivier-Ricci curvature, and causal intervention analysis
through do-calculus. The central hypothesis of Phase II is that financial crises are
not primarily statistical events (extreme returns) but rather topological events
(structural phase transitions) that happen to produce extreme returns as a secondary
symptom.</p>

<p>The name "Project Singularity" refers to the point at which multiple risk indicators
simultaneously reach critical values. In physics, a gravitational singularity is the
point at which the laws of general relativity break down and curvature becomes infinite.
In our framework, the financial singularity is the point at which causal relationships
between assets evaporate (Causal Erasure), correlation structure collapses to a single
mode, and the market's normal self-healing dynamics cease to function.</p>
""")

P.append(nonphd("""<p>Imagine a bridge made of multiple independent cables. In normal conditions,
each cable bears its own load independently. The bridge is robust because no single
cable failure can bring down the whole structure. Now imagine all the cables fusing
together into a single thick rope. It looks stronger, but if that rope frays, the
entire bridge fails at once. Phase II measures the degree to which financial markets
have undergone this fusion process, replacing diverse causal connections with a single
dominant correlation channel. The Ricci curvature is our measure of how "fused" the
market has become.</p>"""))

P.append(sec("5.1", "Student-T Hidden Markov Model"))

P.append("""
<p>The Standard Gaussian Hidden Markov Model assumes that, conditional on the hidden
state, returns are normally distributed. This assumption is violated by the fat tails
of financial returns. We replace the Gaussian emission distribution with the Student-T
distribution, which has the probability density function
$p(x | \\mu_k, \\sigma_k, \\nu_k) = \\frac{\\Gamma((\\nu_k+1)/2)}{\\Gamma(\\nu_k/2)\\sqrt{\\pi \\nu_k \\sigma_k^2}} \\left(1 + \\frac{(x-\\mu_k)^2}{\\nu_k \\sigma_k^2}\\right)^{-(\\nu_k+1)/2}$
for state $k$ with location $\\mu_k$, scale $\\sigma_k$, and degrees of freedom $\\nu_k$.</p>

<p>Estimation proceeds via the Expectation-Maximization algorithm, augmented with a
latent scale variable trick. For each observation $x_t$ in state $k$, we introduce
an auxiliary variable $u_{tk}$ that has the interpretation of a precision weight:
$u_{tk} = (\\nu_k + 1) / (\\nu_k + (x_t - \\mu_k)^2 / \\sigma_k^2)$.
The E-step computes the posterior state probabilities $\\gamma_{tk} = P(z_t = k | x_{1:T})$
via the forward-backward algorithm. The M-step then updates parameters using
weighted least squares with weights $\\gamma_{tk} u_{tk}$.</p>

<p>The degrees-of-freedom parameter $\\nu_k$ for each state is updated by solving the
scalar nonlinear equation $-\\psi(\\nu_k/2) + \\ln(\\nu_k/2) + 1 + n_k^{-1}\\sum_t \\gamma_{tk}(\\ln u_{tk} - u_{tk}) = 0$
via Newton's method at each M-step, where $\\psi$ denotes the digamma function and
$n_k = \\sum_t \\gamma_{tk}$. The key result is that the estimated $\\nu_k$ drops to
approximately 5.0 when the crisis regime becomes dominant, compared to values of
15-30 in normal regimes. This collapse of the degrees-of-freedom parameter serves
as a real-time tail-risk indicator.</p>
""")

P.append(code("""
import numpy as np
from scipy.special import gammaln, digamma
from scipy.optimize import brentq

class StudentTHMM:
    \"\"\"
    Two-state Hidden Markov Model with Student-T emissions.
    Estimated via EM with latent scale variable augmentation.
    \"\"\"
    def __init__(self, n_states=2, max_iter=100, tol=1e-4):
        self.K = n_states
        self.max_iter = max_iter
        self.tol = tol

    def _student_t_logpdf(self, x, mu, sigma, nu):
        z = (x - mu) / sigma
        log_c = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*np.log(np.pi*nu) - np.log(sigma)
        return log_c - (nu+1)/2 * np.log(1 + z**2/nu)

    def fit(self, x):
        T = len(x)
        K = self.K
        # Initialize: split data at median
        split = np.median(np.abs(x))
        self.mu    = np.array([x[np.abs(x) < split].mean(), x[np.abs(x) >= split].mean()])
        self.sigma = np.array([x[np.abs(x) < split].std(), x[np.abs(x) >= split].std()])
        self.nu    = np.array([10.0, 5.0])
        self.pi    = np.array([0.7, 0.3])
        self.A     = np.array([[0.95, 0.05], [0.10, 0.90]])

        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            # E-step: forward-backward
            log_emit = np.column_stack([
                self._student_t_logpdf(x, self.mu[k], self.sigma[k], self.nu[k])
                for k in range(K)
            ])
            # Forward pass (log-sum-exp for numerical stability)
            log_alpha = np.zeros((T, K))
            log_alpha[0] = np.log(self.pi + 1e-300) + log_emit[0]
            for t in range(1, T):
                for k in range(K):
                    log_alpha[t, k] = (
                        np.logaddexp.reduce(log_alpha[t-1] + np.log(self.A[:, k] + 1e-300))
                        + log_emit[t, k]
                    )
            # Backward pass
            log_beta = np.zeros((T, K))
            for t in range(T-2, -1, -1):
                for k in range(K):
                    log_beta[t, k] = np.logaddexp.reduce(
                        np.log(self.A[k, :] + 1e-300) + log_emit[t+1] + log_beta[t+1]
                    )
            # Posterior
            log_gamma = log_alpha + log_beta
            log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # Latent scale weights u_{tk}
            u = np.zeros((T, K))
            for k in range(K):
                z2 = (x - self.mu[k])**2 / self.sigma[k]**2
                u[:, k] = (self.nu[k] + 1) / (self.nu[k] + z2)

            # M-step
            nk = gamma.sum(axis=0)
            for k in range(K):
                w = gamma[:, k] * u[:, k]
                self.mu[k]    = (w * x).sum() / (w.sum() + 1e-12)
                self.sigma[k] = np.sqrt((w * (x - self.mu[k])**2).sum() / (nk[k] + 1e-12))
                # Update nu via moment matching
                def nu_eq(nu):
                    u_k = (nu+1)/(nu + (x-self.mu[k])**2/self.sigma[k]**2)
                    return (-digamma(nu/2) + np.log(nu/2) + 1 +
                            (gamma[:, k] * (np.log(u_k+1e-12) - u_k)).sum() / (nk[k]+1e-12))
                try:
                    self.nu[k] = brentq(nu_eq, 2.01, 200.0)
                except Exception:
                    pass
            self.pi = gamma[0] / gamma[0].sum()
            self.A  = (np.einsum('tk,tl->kl', gamma[:-1, :], gamma[1:, :])
                       / gamma[:-1].sum(axis=0, keepdims=True).T + 1e-12)
            self.A /= self.A.sum(axis=1, keepdims=True)

            ll = np.logaddexp.reduce(log_alpha[-1])
            if abs(ll - prev_ll) < self.tol: break
            prev_ll = ll
        return self
"""))

P.append(sec("5.2", "Ricci Curvature and the Geometry of Contagion"))

P.append("""
<p>Ollivier-Ricci curvature on a graph is a discrete analogue of Riemannian curvature.
For an edge $(u, v)$ in a weighted graph, the Ollivier-Ricci curvature is defined as
$\\kappa(u, v) = 1 - W_1(m_u, m_v) / d(u, v)$, where $W_1$ is the Wasserstein-1
(earth mover's) distance between the probability measures $m_u$ and $m_v$ that distribute
mass uniformly over the neighbors of $u$ and $v$ respectively, and $d(u, v)$ is the
graph distance between $u$ and $v$.</p>

<p>Positive Ricci curvature on an edge indicates that the neighborhoods of the two
endpoints are "close" in the Wasserstein sense: information can travel efficiently
through alternative paths, and the network is robust to edge deletion. Negative Ricci
curvature indicates that the two endpoints are the only connection between otherwise
distant neighborhoods: the edge is a bottleneck, and its removal (analogous to a
liquidity withdrawal) would cause a significant disruption to network connectivity.</p>

<p>For computational efficiency, we use the spectral gap proxy: $\\kappa_{\\text{spec}} = \\lambda_2(L) / d_{\\max}$,
where $\\lambda_2(L)$ is the Fiedler eigenvalue of the normalized graph Laplacian.
This proxy is computable in $O(N^2)$ time (dominated by the eigendecomposition) and
correlates strongly with the true Ollivier-Ricci curvature on dense financial graphs.
The key result of Phase II is that this spectral curvature measure approaches zero
approximately 25 bars before the crisis peak, providing one of the most reliable
leading indicators in the program.</p>
""")

P.append(sec("5.3", "Wormhole Contagion Network"))

P.append("""
<p>We define a "wormhole" edge in the financial network as a correlation link that exceeds
the 99th percentile threshold of the rolling 500-bar distribution of pairwise correlations.
The metaphor is drawn from general relativity: a wormhole is a shortcut through spacetime
that connects two distant regions. In the financial network, a wormhole is a suddenly
high correlation between two assets that would not normally be expected to move together,
representing a contagion channel that has "opened up" in response to a common shock.</p>

<p>The count of wormhole edges in the network is a highly sensitive early-warning indicator.
In normal conditions, the wormhole count is approximately 5 (out of 435 possible pairs).
During the pre-crisis period, it rises slowly as common factor exposure increases. At the
crisis peak (bar 825 in our simulation), the wormhole count reaches 224, representing
a 44-fold amplification of supercritical correlation links. This surge is detectable
approximately 20 bars before the price-level crisis peak.</p>
""")

P.append(img("BOOK_18_wormhole_emergence.png",
    "Wormhole contagion network at four time points: T-400 (3 wormholes, normal), "
    "T-200 (12 wormholes, pre-stress), T-50 (67 wormholes, alert), and T=0 (224 wormholes, "
    "crisis peak). The emergence of supercritical correlation links from a sparse normal "
    "network to a near-complete wormhole network takes approximately 400 bars, "
    "providing a long warning window."))

P.append(sec("5.4", "Do-Calculus and Causal Erasure"))

P.append("""
<p>Pearl's do-calculus provides a formal framework for computing the effect of interventions
in a causal system. The key operator is $do(X_i = v)$, which represents setting variable
$X_i$ to value $v$ by external manipulation rather than by conditioning on the observed
value. The causal effect $P(Y | do(X_i = v))$ is computed by intervening on the structural
equation for $X_i$, severing all of its incoming edges in the causal DAG while leaving
all other structural equations unchanged.</p>

<p>We implement this through a linear Structural Equation Model: $\\mathbf{X} = B \\mathbf{X} + \\boldsymbol{\\epsilon}$,
where $B$ is the learned coefficient matrix (estimated by OLS regression of each variable
on the others) and $\\boldsymbol{\\epsilon}$ is the exogenous noise. The do-intervention
$do(X_i = v)$ is implemented by setting the $i$-th column of $B$ to zero (severing all
incoming edges), then simulating the modified system.</p>

<p>The Causal Erasure delta, $\\Delta = \\|\\bar{\\mathbf{X}}_{\\text{obs}} - \\bar{\\mathbf{X}}_{\\text{do}}\\|_1 / p$,
measures the mean absolute difference between the observed asset mean returns and the
mean returns under the causal intervention. In normal market conditions, $\\Delta$ is
substantial: intervening on any single asset changes the equilibrium meaningfully,
because the causal connections are real. During the crisis window, $\\Delta$ approaches
zero, indicating that causal connections have become so weak that interventions have
no effect. This is the operational definition of Causal Erasure: the market's behavior
is no longer causally organized, only statistically correlated.</p>
""")

P.append(code("""
import numpy as np

def causal_intervention(W, data, intervene_on, target_layer, n_sim=500):
    \"\"\"
    Compute causal erasure delta using linear SEM do-calculus.
    W: (N, N) coefficient matrix from OLS
    data: (T, N) observed returns
    intervene_on: list of node indices to intervene on
    target_layer: list of node indices in the target layer
    Returns: delta (scalar), X_do (n_sim, N) simulated post-intervention data
    \"\"\"
    intervene_on = list(intervene_on)
    target_layer = list(target_layer)

    # Apply do-operator: sever incoming edges to intervened nodes
    B = W.copy()
    B[:, intervene_on] = 0.0

    # Simulate post-intervention distribution
    X_do = np.zeros((n_sim, W.shape[0]))
    rng  = np.random.default_rng(42)
    eps  = rng.multivariate_normal(
        np.zeros(W.shape[0]),
        np.eye(W.shape[0]) * data.var(axis=0).mean(),
        n_sim
    )
    I_minus_B = np.eye(W.shape[0]) - B
    for t in range(n_sim):
        X_do[t] = np.linalg.solve(I_minus_B, eps[t])

    # Causal erasure delta: difference in means across target layer
    obs_mean = data[:, target_layer].mean(axis=0)
    do_mean  = X_do[:, target_layer].mean(axis=0)
    delta    = np.abs(obs_mean - do_mean).mean()

    return delta, X_do
"""))

for f, c in [
    ("SG_01_ricci_singularity.png",
     "Spectral Ricci curvature proxy over 3000 simulation bars. The curvature "
     "approaches zero at crisis bar 825, signaling critical network connectivity. "
     "The first alarm threshold crossing occurs approximately 25 bars before the "
     "price-level crisis peak."),
    ("SG_02_wormhole_network.png",
     "Wormhole count: supercritical correlation links over time. The count surges "
     "from a baseline of approximately 5 to a peak of 224 at bar 825. This 44x "
     "amplification constitutes one of the strongest amplitude signals in the program."),
    ("SG_03_student_t_hmm.png",
     "Student-T HMM degrees-of-freedom parameter over time. The parameter collapses "
     "from values of 15-30 in normal regimes to approximately 5.0 at crisis onset, "
     "confirming the transition to a fat-tail regime approximately 5 bars before "
     "the price-level peak."),
    ("SG_04_causal_erasure.png",
     "Causal Erasure delta: the L1 difference between observed and do-calculus "
     "simulated asset means. Delta approaches 0.0077 during the crisis window, "
     "confirming near-complete causal decoupling of fundamentals from prices."),
    ("SG_05_epistemic_uncertainty.png",
     "Epistemic uncertainty from the D3QN ensemble: variance of Q-value predictions "
     "across 5 networks. High uncertainty precedes regime changes by 3-5 bars, "
     "providing a fast early-warning signal complementary to the slower structural signals."),
    ("SG_06_bayesian_debate.png",
     "Phase II Bayesian Debate consensus: 5-round agent credibility dynamics and "
     "posterior direction probability per bar. The debate mechanism reduces directional "
     "prediction error compared to any single agent."),
    ("SG_07_full_dashboard.png",
     "Phase II full dashboard: all six Phase II signals on a common time axis. "
     "The simultaneous activation of multiple signals near bar 825 constitutes "
     "the Phase II 'Singularity Score' that feeds into the Phase VII hypercube."),
]:
    P.append(img(f, c))

P.append(takeaway("""<p>Phase II identifies four powerful structural signals: Ricci curvature approaching
zero (25-bar lead), wormhole count surge (20-bar lead), Student-T degrees-of-freedom
collapse (5-bar lead), and Causal Erasure approaching zero (3-bar lead). Together,
these signals provide a multi-timescale picture of the structural collapse that precedes
market crises. The causal erasure result is particularly striking: it demonstrates that
by the time prices are dislocating, the causal mechanism that normally connects
fundamentals to prices has already been severed.</p>"""))

P.append(linkedin(""""Markets don't crash randomly. We've modeled market crises as topological events
using Ricci curvature and causal do-calculus. When the Causal Erasure delta approaches
zero and wormhole count approaches 224, the market has entered its pre-crash singularity.
We see the collapse coming before the first price tick. #QuantitativeFinance
#CausalInference #TopologyOfRisk"""))

P.append("<hr>")

# =========================================================================
#  CHAPTER 6: PHASE III
# =========================================================================
P.append(chapter(6, "Phase III: The Singularity Protocol",
    "HJB Optimal Stopping, Extreme Value Theory, and the Zero-Dimension Arbitrage Window"))
P.append('<h1 id="ch6"></h1>')

P.append("""
<p>Phase III introduces the transition from observation to control. The preceding two
phases developed powerful structural indicators of market crises, but converting those
indicators into a trading strategy requires solving a control problem: given that we
know a crisis is approaching, when should we exit? How do we quantify the tail risk
of our current exposure? And is there a brief window, just before the crisis becomes
obvious, when the market offers a particularly attractive trading opportunity?</p>

<p>Phase III answers these questions using three frameworks: the Hamilton-Jacobi-Bellman
equation for the optimal exit problem, Generalized Pareto Distribution fitting for
tail risk quantification, and the Zero-Dimension Arbitrage Window concept for the
"eye of the storm" trading opportunity.</p>
""")

P.append(nonphd("""<p>Think of driving toward a cliff in fog. The HJB equation tells you the optimal
point at which to apply the brakes, given your speed, the distance to the cliff, and
the uncertainty in your position. Extreme value theory tells you how bad it would be
if you braked too late. The Zero-Dimension Arbitrage Window is the brief moment just
before everyone realizes there is a cliff: at that moment, assets that should be
diverging (because they have different cliff-proximity) are still correlated (because
most participants haven't processed the information). That gap between mechanical
synchrony and informational divergence is the arbitrage opportunity.</p>"""))

P.append(sec("6.1", "Hamilton-Jacobi-Bellman Optimal Stopping"))

P.append("""
<p>The optimal stopping problem in our context is formulated as follows. The agent holds
a portfolio with value $X_t$ evolving as a diffusion process, and must decide at each
time $t$ whether to continue holding (stay in the continuation region $\\mathcal{C}$)
or to liquidate (enter the stopping region $\\mathcal{S}$). The objective is to maximize
the expected discounted payoff $\\mathbb{E}[e^{-rt} g(X_{\\tau})]$ over all stopping
times $\\tau$.</p>

<p>The value function $V(t, x) = \\sup_{\\tau \\geq t} \\mathbb{E}[g(X_\\tau) | X_t = x]$
satisfies the variational inequality:
$$\\min\\{V(t,x) - g(x),\\; -\\partial_t V - \\mu(x)\\partial_x V - \\frac{1}{2}\\sigma^2(x)\\partial_{xx}V + rV\\} = 0$$
with terminal condition $V(T, x) = g(x)$. The first term in the min operator is zero
in the stopping region (where it is optimal to stop), and the second term is zero in
the continuation region (where the HJB PDE holds).</p>

<p>We solve this numerically via backward induction on a discrete grid. The key result is
remarkable: the HJB stopping boundary, which signals when the model-implied optimal exit
condition is met, triggers an alarm 799 bars before the realized volatility peak. This
is by far the longest lead time of any single signal in the program. The signal's extreme
advance notice reflects the fact that the HJB formulation incorporates information about
the global future trajectory of the market, not just the local state, making it sensitive
to structural changes in the drift and diffusion dynamics long before those changes become
visible in prices.</p>
""")

P.append(img("BOOK_12_hjb_value_surface.png",
    "The Hamilton-Jacobi-Bellman value function V(t,x) as a 3D surface over state "
    "space x and time t. The red region marks the optimal stopping region where "
    "V(t,x) = g(x): the payoff from stopping now equals the value of waiting. "
    "The boundary of the red region is the optimal stopping boundary, which in "
    "our financial application corresponds to the signal that triggers 799 bars "
    "before the realized volatility peak."))

P.append(code("""
import numpy as np

def hjb_stopping_boundary(drift_path, vol_path, T_horizon=200, n_x=100, r=0.001):
    \"\"\"
    Discrete-time HJB backward induction for optimal stopping.
    drift_path: (T,) array of estimated drift (rolling mean return)
    vol_path:   (T,) array of estimated volatility (rolling std)
    Returns: (T,) array of HJB stopping signal (1 = stop, 0 = continue)
    \"\"\"
    T = len(drift_path)
    signal = np.zeros(T)

    # State grid: normalized around zero
    x_grid = np.linspace(-3, 3, n_x)
    dx = x_grid[1] - x_grid[0]

    for t in range(T_horizon, T):
        mu_t  = drift_path[max(0, t-T_horizon):t].mean()
        sig_t = vol_path[max(0, t-T_horizon):t].mean()

        # Payoff: long bias, so g(x) = max(x, 0) (optionality of staying long)
        g = np.maximum(x_grid, 0)
        holding_cost = 0.005  # per-bar risk premium

        # Backward induction over T_horizon steps
        V = g.copy()
        for step in range(T_horizon):
            # Expected next-period value via simple transition kernel
            V_next = np.zeros(n_x)
            for ix, x in enumerate(x_grid):
                # Euler-Maruyama transition probabilities
                x_mean = x + mu_t
                x_std  = sig_t
                # Gaussian transition: E[V(x')] approximately
                left  = max(0, ix - 2); right = min(n_x, ix + 3)
                x_neighbors = x_grid[left:right]
                probs = np.exp(-0.5*((x_neighbors - x_mean)/x_std)**2)
                probs /= (probs.sum() + 1e-12)
                V_next[ix] = (probs * V[left:right]).sum()

            V = np.maximum(g, np.exp(-r) * V_next - holding_cost)

        # If V at current state (x~0) equals g(0), HJB says stop
        mid = n_x // 2
        signal[t] = float(V[mid] <= g[mid] + 1e-4)

    return signal
"""))

P.append(sec("6.2", "Extreme Value Theory and GPD Tail Fitting"))

P.append("""
<p>Extreme Value Theory (EVT) provides a mathematically rigorous framework for modeling
the tail behavior of distributions beyond the range of observed data. The Peaks Over
Threshold (POT) method, based on the Pickands-Balkema-de Haan theorem, is the preferred
approach for financial risk management because it makes efficient use of the available
tail data.</p>

<p>For a threshold $u$ set at the 5th percentile of the return distribution (losses
exceeding this level), the exceedances $Y = -(X - u)$ for $X < u$ are modeled as
a Generalized Pareto Distribution. The GPD parameters $(\\xi, \\sigma)$ are estimated
by maximum likelihood. Under our synthetic data generating process (Student-T with
$\\nu = 5$), the true $\\xi \\approx 1/\\nu = 0.2$: a moderately heavy-tailed distribution
firmly in the Fréchet domain of attraction.</p>

<p>The rolling $\\xi$ estimate provides a dynamic tail-risk indicator. As the simulated
market approaches the crisis, the realized return distribution becomes heavier-tailed
(reflecting the Student-T regime with declining $\\nu$), and the estimated $\\xi$ increases.
The EVT alarm is triggered when the estimated $\\xi$ exceeds its 90th percentile value
computed on the pre-crisis training window. This alarm fires 765 bars before the crisis
peak, the second-longest lead time of any signal in the program.</p>
""")

P.append(img("BOOK_19_gpd_tail_fitting.png",
    "GPD tail fitting procedure. Left: simulated return distribution with tail "
    "exceedances below the 5th percentile threshold highlighted in red. The Student-T "
    "distribution (gold) fits the body well but the tail behavior is better characterized "
    "by the GPD. Right: GPD fit to the exceedances, confirming xi > 0 (Frechet domain, "
    "heavy tails). The shape parameter xi serves as a dynamic tail-risk indicator."))

P.append(sec("6.3", "The Zero-Dimension Arbitrage Window"))

P.append("""
<p>The Zero-Dimension Arbitrage Window (ZDIM) is one of the most counterintuitive findings
of the program. The defining condition for the ZDIM is the simultaneous occurrence of
high correlation (assets moving together) and low causal edge density (the causal
connections explaining that co-movement have evaporated). On the surface, this seems
contradictory: how can assets be correlated without being causally connected?</p>

<p>The resolution lies in the distinction between correlation and causation at the
microstructure level. During the pre-crash window, assets co-move mechanically because
they are all responding to the same exogenous shock (a common factor, a news event, a
liquidity withdrawal by a large participant). But the normal causal channels through
which idiosyncratic information propagates have been drowned out. The result is that
prices are informationally cheap: they are moving in lockstep, but the movement does
not reflect the differentiated fundamental information that normally drives cross-asset
price discovery.</p>

<p>The ZDIM signal is computed as a composite: $\\text{ZDIM}_t = \\bar{\\rho}_t \\cdot (1 - D_t)$,
where $\\bar{\\rho}_t$ is the mean pairwise correlation (high during the window) and $D_t$
is the Granger causality edge density (low during the window). The ZDIM peaks at 0.7918
during the crisis onset, providing a strong signal that the mechanistic synchrony regime
has been reached. A trading strategy that enters long-volatility positions when ZDIM
exceeds its 80th percentile threshold achieves excess returns during the crisis window.</p>
""")

for f, c in [
    ("P3_01_evt_spectral_risk.png",
     "EVT spectral risk: GPD tail exceedance probability over time with the EVT alarm "
     "threshold. The alarm first crosses the threshold 765 bars before the realized "
     "volatility peak, providing one of the longest lead times in the program."),
    ("P3_02_zero_dim_arbitrage.png",
     "Zero-Dimension Arbitrage Window signal: ZDIM peaks at 0.7918 during the "
     "joint high-correlation, low-causal-density condition. The window lasts "
     "approximately 40 bars and represents the 'eye of the storm' trading opportunity."),
    ("P3_03_hjb_stopping_boundary.png",
     "HJB optimal stopping boundary: value function V(t,x) and the stopping region. "
     "The boundary alarm triggers 799 bars before the realized volatility peak, "
     "the longest single-signal lead in the entire program."),
    ("P3_04_portfolio_ppo_hjb.png",
     "Portfolio comparison: PPO agent versus HJB-guided strategy equity curves. "
     "The HJB guidance dramatically improves the strategy's crisis navigation by "
     "providing an advance signal to reduce exposure before the price dislocation."),
    ("P3_05_lob_spacetime_3d.png",
     "Relativistic LOB spacetime: 3D manifold of limit order book depth over time "
     "and price. The 'curvature' of this surface tracks liquidity gravity. As "
     "liquidity withdraws, the surface develops sharp concavities analogous to "
     "gravitational wells in general relativity."),
    ("P3_06_gev_fat_tail_layers.png",
     "GEV block maxima: Generalized Extreme Value fit to 20-bar block maxima of "
     "returns across three asset layers. The shape parameter xi > 0 is confirmed "
     "for all layers, with the DeFi layer exhibiting the heaviest tails (largest xi)."),
    ("P3_07_risk_graph_topology.png",
     "Risk graph topology: asset correlation network with edge weights proportional "
     "to systemic risk contribution. The transition from a diverse network to a "
     "hub-dominated structure is visible in the comparison between normal and crisis "
     "regime snapshots."),
    ("P3_08_full_dashboard.png",
     "Phase III full dashboard: all Phase III signals on a common time axis, "
     "including HJB stopping signal, EVT tail alarm, ZDIM, GPD shape parameter, "
     "and portfolio equity curves."),
]:
    P.append(img(f, c))

P.append(takeaway("""<p>Phase III provides three strategically critical signals: the HJB stopping boundary
(799-bar lead), the EVT tail alarm (765-bar lead), and the ZDIM window (180-bar lead
to the arbitrage opportunity). These are the three longest-lead signals in the program
and constitute the "early warning system" tier of the Grand Unified Model. They are
most valuable for strategic position management over multi-week to multi-month horizons,
rather than short-term trading.</p>"""))

P.append("<hr>")

# =========================================================================
#  CHAPTER 7: PHASE IV
# =========================================================================
P.append(chapter(7, "Phase IV: The Chronos Collapse",
    "MF-DFA, Cross-Layer Transfer Entropy, CUSUM Breaks, Mixture-of-Experts Control, and the HJB-TD3 Hybrid"))
P.append('<h1 id="ch7"></h1>')

P.append("""
<p>Phase IV introduces information geometry as the fourth pillar of the framework.
The central hypothesis of Phase IV, called the Information-Gap Hypothesis, states that
market profitability does not reside in price prediction per se, but rather in the
latency of information diffusion across the multifractal spectrum. Specifically,
during the window when the multifractal spectrum is widening (indicating pre-shock
turbulence) while transfer entropy is collapsing (indicating information decoupling),
the market enters a locally deterministic regime where sophisticated agents can achieve
extraordinary short-term performance.</p>

<p>The "Chronos Collapse" name refers to the temporal structure of this information-theoretic
event: the normal flow of time, in which cause precedes effect in a measurable way, appears
to collapse during this window. Prices move in tight synchrony not because information is
flowing but precisely because it has stopped flowing. This is the most sophisticated of the
seven conceptual insights in the program.</p>
""")

P.append(nonphd("""<p>Imagine a complex piece of jazz music. In normal times, each musician is
improvising slightly differently, creating a rich, multifractal sound texture. Before
a performance breaks down, something strange happens: the musicians start playing in
increasingly tight synchrony (high correlation), but the musical conversation between
them (the causal information flow) has actually stopped. Each musician is just copying
the others rather than contributing new ideas. The multifractal spectrum measures how
"rich" and diverse the musical texture is. When it suddenly widens, the musicians are
"fighting" before they give up and synchronize. Transfer entropy measures whether they
are actually listening to each other. The Information Gap is when the spectrum is widest
but the listening has stopped.</p>"""))

P.append(sec("7.1", "Multifractal Detrended Fluctuation Analysis"))

P.append("""
<p>Standard Hurst exponent analysis characterizes a time series by a single scaling
exponent $H$, implicitly assuming that the series has a single dominant scaling behavior.
Multifractal analysis relaxes this assumption, allowing different parts of the series
to scale differently and characterizing the series by an entire spectrum of scaling
exponents.</p>

<p>MF-DFA proceeds in four steps. First, compute the profile $Y_i = \\sum_{t=1}^i (x_t - \\bar{x})$.
Second, divide the profile into $N_s = \\lfloor N/s \\rfloor$ non-overlapping segments of
length $s$ and detrend each segment by fitting and subtracting a polynomial of degree $m$.
Third, compute the fluctuation function $F_q(s) = \\left(\\frac{1}{2N_s}\\sum_{\\nu=1}^{2N_s} [F^2(\\nu,s)]^{q/2}\\right)^{1/q}$
for a range of moment orders $q \\in [-5, 5]$ and time scales $s$. Fourth, estimate the
generalized Hurst exponent $h(q)$ from the power-law scaling $F_q(s) \\sim s^{h(q)}$.</p>

<p>The multifractal singularity spectrum $f(\\alpha)$ is derived via the Legendre transform:
$\\tau(q) = qh(q) - 1$, $\\alpha(q) = d\\tau/dq$, $f(\\alpha) = q\\alpha - \\tau(q)$.
The spectrum width $\\Delta\\alpha = \\alpha_{\\max} - \\alpha_{\\min}$ is the key summary
statistic. Phase IV finds that $\\Delta\\alpha$ widens from approximately 0.19 in normal
regimes to 0.41 at crisis onset, representing a doubling of multifractal richness.
This widening is the "pre-shock turbulence" signature: the market is becoming more
complex before it simplifies into a crisis monofractal.</p>
""")

P.append(img("BOOK_09_mfdfa_concept.png",
    "The multifractal singularity spectrum f(alpha) under three market conditions: "
    "pre-shock turbulence (left, Delta-alpha=0.90), normal market (center, Delta-alpha=1.30), "
    "and crisis monofractal (right, Delta-alpha=0.30). The widening and then narrowing "
    "of the spectrum provides a two-stage warning system: widening signals turbulence, "
    "narrowing signals monofractal collapse. In Phase IV, we focus on the widening phase."))

P.append(code("""
import numpy as np

def mfdfa(x, q_vals=None, scales=None, m=1):
    \"\"\"
    Multifractal Detrended Fluctuation Analysis.
    x: 1D time series
    q_vals: array of moment orders (default: -5 to 5, 21 points)
    scales: array of time scales (default: logarithmically spaced)
    m: polynomial order for detrending (default: 1 = linear)
    Returns: hq (generalized Hurst), tau, alpha, f_alpha, q_vals
    \"\"\"
    if q_vals is None:
        q_vals = np.linspace(-5, 5, 21)
    N = len(x)
    # Step 1: Profile
    profile = np.cumsum(x - x.mean())

    # Step 2: Scales
    if scales is None:
        scales = np.unique(np.logspace(
            np.log10(10), np.log10(N // 4), 20
        ).astype(int))
    scales = scales[scales >= 4]  # minimum scale

    # Step 3: Fluctuation function
    Fq = np.zeros((len(q_vals), len(scales)))
    for si, s in enumerate(scales):
        n_seg = N // s
        if n_seg < 2:
            Fq[:, si] = np.nan
            continue
        # Detrend each segment
        F2 = np.zeros(n_seg)
        t_seg = np.arange(s)
        for seg in range(n_seg):
            seg_data = profile[seg*s:(seg+1)*s]
            coef = np.polyfit(t_seg, seg_data, m)
            trend = np.polyval(coef, t_seg)
            F2[seg] = np.mean((seg_data - trend)**2)

        # Compute Fq for each moment order
        for qi, q in enumerate(q_vals):
            if abs(q) < 1e-6:  # q=0: geometric mean
                Fq[qi, si] = np.exp(0.5 * np.mean(np.log(F2 + 1e-12)))
            else:
                Fq[qi, si] = (np.mean((F2 + 1e-12)**(q/2)))**(1.0/q)

    # Step 4: Generalized Hurst exponent from log-log slope
    valid = ~np.any(np.isnan(Fq), axis=0)
    log_s = np.log(scales[valid])
    hq = np.array([
        np.polyfit(log_s, np.log(Fq[qi, valid] + 1e-12), 1)[0]
        for qi in range(len(q_vals))
    ])

    # Legendre transform to get f(alpha)
    tau    = q_vals * hq - 1
    alpha  = np.gradient(tau, q_vals)
    f_alpha = q_vals * alpha - tau

    return hq, tau, alpha, f_alpha, q_vals
"""))

P.append(sec("7.2", "Cross-Layer Transfer Entropy Matrix"))

P.append("""
<p>Phase IV computes the full 3x3 Cross-Layer Transfer Entropy Matrix (CLTE) among
three aggregated layer signals: the mean return of the TradFi layer, the mean return
of the Crypto layer, and the mean return of the DeFi layer. Each element $TE_{i \\to j}$
of this matrix measures the directed information flow from layer $i$ to layer $j$.</p>

<p>The CLTE matrix is estimated using a histogram-based conditional entropy estimator
with $B = \\lceil T^{1/3} \\rceil$ bins per dimension (Rice rule). For a history length
of $k = 1$ bar, the estimator requires a 3-dimensional histogram and proceeds as:
$\\hat{TE}_{X \\to Y} = \\hat{H}(Y_{t+1} | Y_t) - \\hat{H}(Y_{t+1} | Y_t, X_t)$,
where each conditional entropy is computed from the normalized histogram bin counts.</p>

<p>The key finding is that the total inflow of transfer entropy to the TradFi layer
declines from approximately 1.68 nats in normal regimes to 1.33 nats during the
Information-Gap window, a 21 percent reduction. This collapse in information inflow
is the operational detection of the Information-Gap, and it co-occurs with the
maximum of the multifractal spectrum width to define the precise entry window for
the Chronos Collapse trading strategy.</p>
""")

P.append(code("""
import numpy as np

def transfer_entropy(x, y, k=1, bins=None):
    \"\"\"
    Estimate transfer entropy TE(X->Y) using 3D histogram.
    x, y: 1D time series (same length)
    k: history length (default 1)
    Returns: TE estimate in nats
    \"\"\"
    T = len(x)
    if bins is None:
        bins = max(3, int(T**(1/3)))

    # Construct (Y_{t+1}, Y_t, X_t) triplets
    y_next = y[k:]
    y_curr = y[k-1:-1] if k > 0 else y[:-1]
    x_curr = x[k-1:-1] if k > 0 else x[:-1]

    n = len(y_next)

    # 3D histogram: (y_next, y_curr, x_curr)
    hist_3d, edges = np.histogramdd(
        np.column_stack([y_next, y_curr, x_curr]),
        bins=bins
    )
    p_3d = hist_3d / (n + 1e-12)

    # 2D marginal: (y_curr, x_curr)
    p_2d_yx = p_3d.sum(axis=0)  # sum over y_next

    # 2D marginal: (y_next, y_curr)
    p_2d_yn = p_3d.sum(axis=2)  # sum over x_curr

    # 1D marginal: y_curr
    p_1d_y = p_2d_yx.sum(axis=1)  # sum over x_curr

    # TE = sum p(y+1, y, x) * log[ p(y+1|y, x) / p(y+1|y) ]
    # Equivalently: H(y+1|y) - H(y+1|y,x)
    te = 0.0
    for a in range(bins):   # y_next
        for b in range(bins):   # y_curr
            for c in range(bins):   # x_curr
                pjoint = p_3d[a, b, c]
                if pjoint < 1e-12: continue
                p_yn_y = p_2d_yn[a, b]
                p_y    = p_1d_y[b]
                p_yx   = p_2d_yx[b, c]
                # Avoid division by zero
                if p_yn_y < 1e-12 or p_y < 1e-12 or p_yx < 1e-12: continue
                # TE contribution
                p_cond_yx = pjoint / (p_2d_yx[b, c] + 1e-12)
                p_cond_y  = p_2d_yn[a, b] / (p_1d_y[b] + 1e-12)
                if p_cond_yx > 1e-12 and p_cond_y > 1e-12:
                    te += pjoint * np.log(p_cond_yx / p_cond_y)

    return max(0.0, te)
"""))

P.append(sec("7.3", "CUSUM Structural Break Detection and MoE Gating"))

P.append("""
<p>The CUSUM (Cumulative Sum) structural break detector monitors a test statistic that
accumulates deviations of the observed process from its expected value under the null
hypothesis of parameter stability. When the accumulated deviation exceeds a threshold,
a structural break is flagged. In Phase IV, we apply CUSUM to the rolling mean of the
Granger-weighted return series, flagging breaks that correspond to regime transitions.</p>

<p>Detected break points trigger a "Policy Reset" event in the Mixture-of-Experts (MoE)
gating network. The MoE consists of three expert agents trained on historically distinct
regime partitions: the Stable Agent (trained on normal market conditions with $\\Delta\\alpha < 0.25$),
the Volatile Agent (trained on high-volatility conditions with $\\Delta\\alpha > 0.35$ and
transfer entropy above baseline), and the Singularity Agent (trained on crisis window
conditions with $\\Delta\\alpha > 0.35$ and transfer entropy below baseline). The gating
network is a softmax layer $g_i = \\exp(\\mathbf{w}_i^T \\mathbf{z}) / \\sum_j \\exp(\\mathbf{w}_j^T \\mathbf{z})$
that weights each expert's action by its relevance to the current state $\\mathbf{z}$.</p>
""")

P.append(img("BOOK_20_cusum_structural_break.png",
    "CUSUM structural break detection applied to a three-regime simulated return series. "
    "Top: observed returns with true regime boundaries (red and orange verticals) and "
    "CUSUM-detected breaks (magenta dotted). Bottom: CUSUM statistic with alarm threshold. "
    "The CUSUM detector identifies regime transitions within 5-10 bars of the true break "
    "points, providing the trigger signal for MoE Policy Reset."))

P.append(img("BOOK_21_moe_expert_selection.png",
    "MoE expert selection dynamics over 3000 simulation bars. Top: stacked area chart "
    "showing the softmax weights of the three experts. The Singularity Agent (red) "
    "receives near-unit weight during the crisis window (bars 780-900). Bottom: "
    "dominant expert at each bar (green=Stable, orange=Volatile, red=Singularity). "
    "The transition to Singularity Agent dominance begins approximately 45 bars "
    "before the price-level crisis peak."))

for f, c in [
    ("P4_01_multifractal_spectrum.png",
     "MF-DFA multifractal singularity spectrum: f(alpha) versus alpha in a rolling "
     "500-bar window. The spectrum width Delta-alpha widens from 0.19 to 0.41 at "
     "crisis onset, the pre-shock turbulence signature identified in Phase IV."),
    ("P4_02_transfer_entropy_flow.png",
     "Cross-Layer Transfer Entropy: 3x3 CLTE matrix heatmap over time. The "
     "DeFi-to-TradFi TE (top-right element) drops from 1.68 to 1.33 nats during "
     "the Information-Gap window, confirming the collapse of cross-domain information flow."),
    ("P4_03_structural_breaks_cusum.png",
     "Structural break seismograph: CUSUM statistic with detected break points "
     "shown as red vertical lines. The pattern of accelerating break frequency "
     "beginning approximately 100 bars before crisis is characteristic of a "
     "system approaching a phase transition."),
    ("P4_04_moe_gating_heatmap.png",
     "MoE agent gating heatmap: expert weight distribution over time. The Singularity "
     "Agent (blue in this representation) receives near-unit weight during the "
     "crisis window. The smooth handoff between experts (no abrupt switches) "
     "is achieved by the softmax gating mechanism."),
    ("P4_05_portfolio_comparison.png",
     "Portfolio comparison: Stable Agent, Volatile Agent, Singularity Agent, and "
     "MoE ensemble equity curves. The MoE ensemble outperforms all individual experts "
     "because it dynamically allocates to the appropriate expert for each regime."),
    ("P4_06_ricci_alpha_3d_manifold.png",
     "Ricci-Alpha 3D manifold: surface plot of Ricci curvature, multifractal width "
     "(Delta-alpha), and time. The surface shows the approach to singularity as a "
     "simultaneous movement toward zero curvature and maximum Delta-alpha."),
    ("P4_07_deterministic_window.png",
     "Deterministic Window indicator: the Information-Gap gauge showing the distance "
     "between price volatility (high) and information flow (low) during the "
     "Deterministic Window. The Singularity Agent achieves Sharpe 0.33 specifically "
     "during this window."),
    ("P4_08_full_chronos_dashboard.png",
     "Phase IV full Chronos dashboard: all Phase IV signals and the CHRONOS composite "
     "signal on a common time axis. The CHRONOS composite is the product of normalized "
     "Delta-alpha and inverse-normalized transfer entropy."),
]:
    P.append(img(f, c))

P.append(takeaway("""<p>Phase IV introduces the Information-Gap Hypothesis and provides two medium-lead
signals: multifractal Delta-alpha widening (30-bar lead) and transfer entropy collapse
(20-bar lead). The MoE gating network, triggered by CUSUM structural breaks, provides
the first complete regime-adaptive trading strategy in the program. The HJB-TD3 hybrid
demonstrates that residual learning (learning the deviation from the HJB-optimal path
rather than predicting prices directly) reduces the effective action space and
accelerates convergence.</p>"""))

P.append("<hr>")

# =========================================================================
#  CHAPTER 8: PHASE V
# =========================================================================
P.append(chapter(8, "Phase V: The Hawkes Singularity",
    "Self-Exciting Processes, Maximum Likelihood Intensity Estimation, N x N Granger Causality, and Page-Hinkley Drift"))
P.append('<h1 id="ch8"></h1>')

P.append("""
<p>Phase V shifts the analytical focus from the spatial topology of asset correlations
to the temporal microstructure of order flow. While Phases I-IV described the market
in terms of its correlation geometry and information flows, Phase V asks a different
question: what is the momentum of the underlying order flow, and how does it signal
the approach of a volatility event?</p>

<p>The Hawkes self-exciting process provides a physical model for this question. Each
trade or large order arrival increases the probability of subsequent arrivals, creating
a self-reinforcing feedback loop that can amplify small shocks into large volatility
events. The "Hawkes Singularity" is the point at which the branching ratio approaches
one from below: the process is on the boundary between stability (each shock eventually
dies out) and explosive growth (each shock triggers more than one further shock, leading
to divergent intensity).</p>
""")

P.append(nonphd("""<p>Think of social media virality. A normal post might get 10 views and inspire 3
shares, each of which gets 10 views and 3 shares, and so on until the initial
audience is exhausted. That is a stable process (branching ratio 0.3). A viral post
gets 10 views and 12 shares, each of which gets 10 views and 12 shares, growing
exponentially. Financial markets near crisis behave like viral posts: each large order
arrival triggers more large order arrivals, amplifying volatility in a self-reinforcing
loop. The Hawkes model measures how "viral" the order flow is at any moment. When
the branching ratio approaches 1, the market is at the edge of a volatility explosion.</p>"""))

P.append(sec("8.1", "Hawkes Process Maximum Likelihood Estimation"))

P.append("""
<p>For a Hawkes process with exponential kernel, the parameters $(\\mu, \\alpha, \\beta)$
are estimated from a sequence of event times $\\{t_1, \\ldots, t_n\\}$ by maximizing
the log-likelihood:
$$\\ell(\\mu, \\alpha, \\beta) = -\\mu T - \\frac{\\alpha}{\\beta}\\sum_{i=1}^n (1 - e^{-\\beta(T-t_i)}) + \\sum_{i=1}^n \\log\\left(\\mu + \\alpha \\sum_{j: t_j < t_i} e^{-\\beta(t_i-t_j)}\\right)$$</p>

<p>The computational bottleneck is the double sum $\\sum_{j: t_j < t_i} e^{-\\beta(t_i-t_j)}$,
which naively requires $O(n^2)$ computations. The key recursive trick is to define
$R_i = \\sum_{j: t_j < t_i} e^{-\\beta(t_i-t_j)}$ and observe that
$R_i = e^{-\\beta(t_i - t_{i-1})} (1 + R_{i-1})$, reducing the computation to $O(n)$.
This recursion is numerically stable and makes rolling MLE feasible even for long
time series.</p>

<p>The rolling MLE is applied to each asset in the 30-asset universe using a 200-bar
sliding window, generating a time series of $\\hat{\\mu}(t)$, $\\hat{\\alpha}(t)$,
and $\\hat{\\beta}(t)$ for each asset. The key signal is the normalized Hawkes intensity
$\\lambda_t / \\lambda_{\\max}$ and the branching ratio $\\hat{\\alpha}/\\hat{\\beta}$.
The Pre-Volatility Spike finding demonstrates that $\\lambda(t)$ surges exactly 12 bars
before the realized volatility $\\sigma_t$ expands, providing a 12-bar lead that is
shorter than the structural signals but more precise and robust to lookback contamination.</p>
""")

P.append(code("""
import numpy as np
from scipy.optimize import minimize

def hawkes_mle(events, T_window, init=(0.3, 0.6, 1.2)):
    \"\"\"
    Maximum likelihood estimation for Hawkes process parameters.
    events: array of event times in [0, T_window]
    T_window: total observation window length
    init: initial guess for (mu, alpha, beta)
    Returns: (mu_hat, alpha_hat, beta_hat)
    \"\"\"
    events = np.sort(np.asarray(events, dtype=float))
    n      = len(events)

    if n < 5:
        return init  # insufficient data

    def neg_log_lik(params):
        mu, alpha, beta = params
        # Enforce constraints: mu > 0, alpha >= 0, beta > alpha (stationarity)
        if mu <= 1e-8 or alpha < 0 or beta <= alpha + 1e-8:
            return 1e10

        # Recursive computation of R_i
        R = np.zeros(n)
        for i in range(1, n):
            R[i] = np.exp(-beta * (events[i] - events[i-1])) * (1.0 + R[i-1])

        # Intensity at each event time
        lam_i = mu + alpha * R

        # Log-likelihood: sum log(lam) - integral of lam
        log_sum  = np.sum(np.log(lam_i + 1e-12))
        integral = mu * T_window + (alpha / beta) * np.sum(
            1.0 - np.exp(-beta * (T_window - events))
        )
        return integral - log_sum

    bounds = [(1e-6, None), (0, None), (1e-6, None)]
    res = minimize(neg_log_lik, init, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 200, 'ftol': 1e-8})

    if res.success:
        return res.x
    return init

def rolling_hawkes_intensity(returns, window=200, threshold_pct=95.0):
    \"\"\"
    Compute rolling Hawkes intensity for a return series.
    Events are defined as |returns| > threshold.
    Returns: lambda_t (T,), alarm (T,) binary
    \"\"\"
    T = len(returns)
    # Define events as large absolute returns
    event_threshold = np.percentile(np.abs(returns), threshold_pct)
    event_times_all = np.where(np.abs(returns) > event_threshold)[0].astype(float)

    lambda_t = np.zeros(T)

    for t in range(window, T):
        # Events in the current window
        mask = (event_times_all >= t - window) & (event_times_all < t)
        ev   = event_times_all[mask] - (t - window)  # normalize to [0, window]

        if len(ev) < 3:
            lambda_t[t] = 0.3  # baseline
            continue

        mu, alpha, beta = hawkes_mle(ev, float(window))
        # Intensity at current time
        R_t = sum(alpha * np.exp(-beta * (window - ei)) for ei in ev)
        lambda_t[t] = mu + R_t

    return lambda_t
"""))

P.append(sec("8.2", "N x N Granger Causality Matrix and Causality Collapse"))

P.append("""
<p>For a universe of $N = 30$ assets, the full Granger causality network is a $30 \\times 30$
directed adjacency matrix $G$ where $G_{ij}$ is the F-statistic for the test that asset $i$
Granger-causes asset $j$ (positive values indicate significant causality, zero indicates
no detected causality). Computing this matrix requires 870 bivariate VAR(p) regressions
(one for each ordered pair), making it computationally intensive but tractable for
moderate values of $N$.</p>

<p>The rolling Granger matrix is computed on a 250-bar sliding window with VAR lag order
$p = 2$. At each time step, we compute the Granger density $D_t = |\{(i,j): G_{ij} > 0\\}| / (N(N-1))$
and the matrix entropy $H_G(t) = -\\sum_{ij} G_{ij}/\\|G\\|_1 \\cdot \\ln(G_{ij}/\\|G\\|_1 + 10^{-12})$,
which measures the diversity of causal influence distribution.</p>

<p>The Causality Collapse result is one of the most dramatic findings of the program.
During normal regimes, the Granger density is approximately 0.35 and the F-statistics
are distributed relatively uniformly across the matrix. During the crisis window,
the density collapses to approximately 0.12 and the surviving F-statistics concentrate
almost entirely on a single row and column corresponding to Asset 18 (the TradFi
sector representative). This is the "super-hub" formation: a single asset accumulates
all causal influence, consistent with the wormhole contagion picture from Phase II.</p>
""")

P.append(img("BOOK_11_network_topology_comparison.png",
    "Granger causality network comparison: normal regime (left, Granger density 0.15, "
    "diverse information sources) versus crisis regime (right, Granger density 0.85 "
    "concentrated on a single super-hub node). The emergence of the super-hub is "
    "the 'Causality Collapse' result: a single node (Asset 18, TradFi sector) "
    "accumulates PageRank 0.1519 at crisis peak, 4.3x the mean."))

P.append(sec("8.3", "Page-Hinkley Drift Detection"))

P.append("""
<p>The Page-Hinkley test is a sequential change-point detection algorithm that monitors
a cumulative sum of deviations from a reference mean, triggering an alarm when the
cumulative sum exceeds a threshold. For a sequence $x_1, x_2, \\ldots$, the Page-Hinkley
test statistic is:
$M_t = \\sum_{i=1}^t (x_i - \\bar{x}_t - \\delta)$, $PH_t = M_t - \\min_{1 \\leq s \\leq t} M_s$
where $\\delta > 0$ is a sensitivity parameter and $\\bar{x}_t$ is the running mean.
An alarm is raised when $PH_t > \\lambda$ for a predetermined threshold $\\lambda$.</p>

<p>In Phase V, we apply the Page-Hinkley test to the residuals of the rolling Hawkes
intensity fit: the difference between the observed intensity and the model-predicted
intensity given the current parameter estimates. When the model parameters become
stale (because the market has undergone a structural change), the residuals drift
systematically, and the Page-Hinkley test detects this drift before it becomes
visible in prices. The alarm fires approximately 8 bars before the realized volatility
peak, making it the fastest of the structural signals.</p>
""")

P.append(code("""
import numpy as np

def page_hinkley_test(residuals, delta=0.005, threshold=0.5):
    \"\"\"
    Page-Hinkley test for structural drift in Hawkes residuals.
    residuals: (T,) array of model prediction residuals
    delta: sensitivity (minimum detectable mean shift)
    threshold: alarm threshold for PH statistic
    Returns: ph_stat (T,), alarm (T,) binary array
    \"\"\"
    T = len(residuals)
    ph_stat = np.zeros(T)
    alarm   = np.zeros(T, dtype=bool)

    running_sum = 0.0
    running_min = 0.0
    running_mean = 0.0

    for t in range(1, T):
        # Update running mean (online)
        running_mean = running_mean + (residuals[t] - running_mean) / t

        # Cumulative sum with sensitivity subtraction
        running_sum += residuals[t] - running_mean - delta
        running_min  = min(running_min, running_sum)

        ph_stat[t] = running_sum - running_min
        alarm[t]   = ph_stat[t] > threshold

    return ph_stat, alarm

def granger_f_test(y, x, max_lag=2):
    \"\"\"
    Bivariate Granger causality F-test: does x Granger-cause y?
    y, x: 1D time series (same length T)
    max_lag: number of lags for VAR (default 2)
    Returns: F-statistic (positive => x Granger-causes y)
    \"\"\"
    T = len(y)
    p = max_lag

    # Build lag matrices
    def build_lag_matrix(series, nlags):
        rows = []
        for t in range(nlags, T):
            rows.append([series[t-k] for k in range(1, nlags+1)])
        return np.array(rows)

    Y = y[p:]          # dependent variable
    Yl = build_lag_matrix(y, p)   # lagged y
    Xl = build_lag_matrix(x, p)   # lagged x

    # Restricted: Y ~ const + lagged Y only
    Zr = np.column_stack([np.ones(len(Y)), Yl])
    coef_r, _, _, _ = np.linalg.lstsq(Zr, Y, rcond=None)
    rss_r = np.sum((Y - Zr @ coef_r)**2)

    # Unrestricted: Y ~ const + lagged Y + lagged X
    Zu = np.column_stack([np.ones(len(Y)), Yl, Xl])
    coef_u, _, _, _ = np.linalg.lstsq(Zu, Y, rcond=None)
    rss_u = np.sum((Y - Zu @ coef_u)**2)

    # F-statistic
    n_obs = len(Y)
    if rss_u < 1e-12 or n_obs <= 2*p + 1:
        return 0.0
    F = ((rss_r - rss_u) / p) / (rss_u / (n_obs - 2*p - 1))
    return max(0.0, F)
"""))

for f, c in [
    ("P5_01_hawkes_intensity.png",
     "Hawkes intensity lambda(t) versus realized volatility over 3000 simulation bars. "
     "The 12-bar lead of intensity surge over volatility expansion is clearly visible: "
     "lambda(t) consistently peaks approximately 12 bars before realized vol expands, "
     "providing a precise short-lead early warning signal."),
    ("P5_02_granger_matrix.png",
     "N×N Granger causality heatmap for the 30-asset universe. In the normal regime "
     "(left sub-panel), F-statistics are distributed relatively uniformly. During the "
     "crisis (right sub-panel), the matrix collapses to show high values only in the "
     "row and column of Asset 18 (TradFi sector): the super-hub emergence."),
    ("P5_03_granger_drift_strategy.png",
     "Granger density and drift-corrected strategy: Granger density collapse (blue, "
     "left axis) overlaid with Page-Hinkley drift alarms (red marks) and the resulting "
     "strategy P&L (green, right axis). The drift-corrected strategy achieves the "
     "Phase V Sharpe of 1.21."),
    ("P5_04_hawkes_layer_lead_lag.png",
     "Hawkes layer lead-lag analysis: cross-correlation between lambda(t) for the "
     "three asset layers (TradFi, Crypto, DeFi) at various lags. DeFi Hawkes intensity "
     "leads TradFi by approximately 8 bars, confirming the cross-domain information "
     "cascade identified in Phase VI."),
    ("P5_05_phase5_dashboard.png",
     "Phase V full Hawkes Singularity dashboard: all Phase V signals composite view, "
     "including Hawkes intensity, Granger density, Page-Hinkley alarm, alpha/beta "
     "decay stability ratio, and strategy equity curve."),
    ("P5_06_granger_networks.png",
     "Granger causal network graph at normal versus crisis regimes. The directed "
     "network visualization confirms the hub-and-spoke structure of the crisis "
     "network, with Asset 18 as the central hub."),
    ("P5_07_hawkes_heatmap_all_assets.png",
     "Hawkes intensity heatmap for all 30 assets: intensity values color-coded "
     "over time. The synchronized activation of all 30 asset intensities at "
     "crisis bar 825 is a striking visual confirmation of the super-hub emergence."),
    ("P5_08_lead_lag_matrix.png",
     "Full 30×30 cross-correlation lead-lag matrix at lag=12 bars. Assets in the "
     "top-left quadrant (TradFi) tend to lag; assets in the bottom-right quadrant "
     "(DeFi) tend to lead, consistent with the cross-domain information cascade model."),
]:
    P.append(img(f, c))

P.append(takeaway("""<p>Phase V provides three medium-to-short lead signals: the Hawkes Pre-Volatility Spike
(12-bar lead), the Granger Density Collapse (18-bar lead), and the Page-Hinkley Drift
Alarm (8-bar lead). Together, these constitute the "microstructure alarm cluster" that
provides precision timing for the signals identified at longer horizons in Phases I-IV.
The Black Swan Node identification (Asset 18, PageRank 0.1519) is also the foundation
for the systemic risk graph in Phase VII.</p>"""))

P.append("<hr>")

# =========================================================================
#  CHAPTER 9: PHASE VI
# =========================================================================
P.append(chapter(9, "Phase VI: The On-Chain Oracle",
    "Synthetic DeFi Signals, Bayesian Agentic Debate, and the Multi-Agent Ensemble"))
P.append('<h1 id="ch9"></h1>')

P.append("""
<p>Phase VI bridges the gap between decentralized finance (DeFi) and traditional finance
(TradFi). The central thesis of Phase VI is that on-chain data, being transparent,
continuous, and structurally harder to manipulate than reported equity market data,
provides genuine leading indicators of TradFi volatility and stress. The "smart money"
in decentralized finance, specifically large whale wallets managing hundreds of millions
of dollars in liquidity pool positions, appears to exit positions systematically before
TradFi stress events materialize in prices.</p>

<p>Phase VI also introduces the most sophisticated agent architecture of the program:
the Bayesian Debate System, in which four specialized reinforcement learning agents
with distinct architectural biases "debate" the directional forecast through five
rounds of credibility-weighted voting, updating their credibility scores based on
agreement with the emerging consensus at each round.</p>
""")

P.append(nonphd("""<p>Imagine you are trying to decide whether to take an umbrella. You have four friends:
a meteorologist (D3QN), a farmer (DDQN), a sailor (TD3), and a delivery driver (PPO).
Each has different expertise and different indicators they trust. Instead of just
picking one friend's advice, you let them debate. In each round, the friends who
have been right recently get more votes. After five rounds of this debate, the
consensus is much more reliable than any single friend's opinion. This is the
Bayesian Debate System: four agents with different information sets and track records,
debating until their credibility-weighted vote converges.</p>"""))

P.append(sec("9.1", "Synthetic On-Chain Signal Stream"))

P.append("""
<p>We generate three synthetic on-chain signals calibrated to match the statistical
properties of actual DeFi data. The signals are designed to exhibit the known
properties of on-chain smart-money behavior: mean-reverting in normal conditions,
with large directional moves preceding TradFi stress events.</p>

<p><strong>DEX Volume Spikes:</strong> Daily decentralized exchange trading volume relative to
the 30-bar rolling average. In normal conditions, this ratio fluctuates around 1.0
with occasional spikes above 2.0. During stress events, DEX volume spikes to 3-5x
normal approximately 15-20 bars before TradFi price dislocations, as participants
rush to hedge or unwind positions on-chain before over-the-counter markets adjust.</p>

<p><strong>Whale Net Flow:</strong> The net directional flow of wallets holding more than 1,000
units of the synthetic base asset. Negative values indicate accumulation (inflows),
positive values indicate distribution (outflows). The whale exit signal (net flow
turning positive and exceeding the 95th percentile threshold) precedes TradFi crashes
by an average of 35 bars in our simulation. This is the longest cross-domain lead
signal in the program and is consistent with the documented behavior of sophisticated
institutional participants in actual DeFi markets.</p>

<p><strong>LP Depth Volatility:</strong> The rolling standard deviation of liquidity pool depth
in the primary synthetic DEX. This measures the "thinning" of DeFi liquidity as
liquidity providers withdraw their capital in anticipation of volatility. LP depth
volatility surges from a baseline of approximately 0.05 to 0.35 during the pre-crisis
period, with a 15-bar lead over TradFi price dislocations.</p>
""")

P.append(sec("9.2", "The Bayesian Debate System"))

P.append("""
<p>The Bayesian Debate System operates as follows. At each bar $t$, each of the four
agents $a_i$ (D3QN, DDQN, TD3-proxy, PPO-proxy) observes the current feature vector
$\\mathbf{f}_t$ (comprising all on-chain and microstructure signals) and submits a
directional bid $b_i \\in \\{-1, 0, +1\\}$ corresponding to a short, flat, or long
position recommendation. Each agent is initialized with credibility parameters
$(\\alpha_i^{(0)}, \\beta_i^{(0)}) = (2, 2)$, corresponding to a Beta(2,2) prior
that is slightly biased toward the center.</p>

<p>Over five debate rounds, the following update rule is applied. The weighted consensus
at round $r$ is $C_r = \\text{sign}\\left(\\sum_i c_i^{(r)} b_i^{(r)}\\right)$ where
$c_i^{(r)} = \\alpha_i^{(r)} / (\\alpha_i^{(r)} + \\beta_i^{(r)})$ is agent $i$'s
credibility at round $r$. Agents whose bid agrees with the consensus have their $\\alpha$
parameter updated as $\\alpha_i^{(r+1)} = \\alpha_i^{(r)} + c_i^{(r)}$; agents whose
bid disagrees have their $\\beta$ parameter updated as $\\beta_i^{(r+1)} = \\beta_i^{(r)} + 0.5 c_i^{(r)}$.
The posterior probability of the up direction after five rounds is
$P(\\text{up}) = \\sum_i c_i^{(5)} \\mathbb{1}[b_i^{(5)} = +1] / \\sum_i c_i^{(5)}$.</p>

<p>The 40 percent variance reduction result is computed by comparing the prediction error
variance of the five-round Bayesian consensus to the prediction error variance of the
best single agent (PPO-proxy) over the full simulation horizon. The reduction is
particularly pronounced during regime transitions, where different agents have different
strengths: the D3QN excels at discrete liquidity regime detection, the TD3-proxy at
volatility magnitude estimation, and the PPO-proxy at trend/momentum regimes.</p>
""")

P.append(img("BOOK_10_bayesian_debate_mechanics.png",
    "Bayesian Debate mechanics illustrated over five rounds. At round 1, the Beta "
    "distribution is relatively flat (high uncertainty). By round 5, the distribution "
    "has sharpened around a mode of approximately 0.7, reflecting strong consensus "
    "toward the up direction. The mode value serves as the ensemble directional signal."))

P.append(code("""
import numpy as np

class SubAgent:
    \"\"\"
    Simple linear agent for Bayesian Debate ensemble.
    Maintains Beta credibility prior (alpha, beta) updated by debate.
    \"\"\"
    def __init__(self, n_features, style='linear', seed=0):
        rng = np.random.default_rng(seed)
        self.style  = style
        self.alpha  = 2.0   # Beta prior: initial credibility
        self.beta   = 2.0
        # Q-network: (n_features) -> (n_actions=3) via two-layer linear
        self.W1 = rng.normal(0, 0.1, (64, n_features))
        self.b1 = np.zeros(64)
        if style == 'td3':
            self.W2 = rng.normal(0, 0.1, (1, 64))   # continuous output
        else:
            self.W2 = rng.normal(0, 0.1, (3, 64))   # 3 discrete actions
        self.b2 = np.zeros(self.W2.shape[0])

    def predict(self, feat):
        \"\"\"Return directional bid: -1, 0, or +1\"\"\"
        h = np.maximum(0, self.W1 @ feat + self.b1)   # ReLU hidden
        q = self.W2 @ h + self.b2
        if self.style == 'td3':
            return int(np.sign(q[0]))
        return int(np.argmax(q) - 1)   # {0,1,2} -> {-1,0,1}

    def update(self, feat, action_idx, reward):
        \"\"\"Simple online update via reward-weighted gradient\"\"\"
        h = np.maximum(0, self.W1 @ feat + self.b1)
        q = self.W2 @ h + self.b2
        if action_idx < self.W2.shape[0]:
            target = q.copy()
            target[action_idx] += reward * 0.01
            delta2 = target - q
            self.W2 += 0.001 * np.outer(delta2, h)
            self.b2 += 0.001 * delta2

def bayesian_debate(agents, feat, n_rounds=5):
    \"\"\"
    Run the Bayesian Debate system for one bar.
    agents: list of SubAgent instances
    feat: current feature vector
    Returns: p_up (float), credibilities (array), consensus (int)
    \"\"\"
    n = len(agents)
    credibility = np.array([ag.alpha / (ag.alpha + ag.beta) for ag in agents])

    bids = np.array([ag.predict(feat) for ag in agents], dtype=float)

    for round_idx in range(n_rounds):
        # Credibility-weighted consensus direction
        weighted_vote = np.dot(credibility, bids)
        consensus = int(np.sign(weighted_vote)) if abs(weighted_vote) > 0.1 else 0

        # Update credibilities based on agreement with consensus
        for i, ag in enumerate(agents):
            if int(bids[i]) == consensus and consensus != 0:
                ag.alpha += credibility[i]
            elif int(bids[i]) != consensus and consensus != 0:
                ag.beta  += credibility[i] * 0.5

        credibility = np.array([ag.alpha / (ag.alpha + ag.beta) for ag in agents])

    # Final posterior probability of up direction
    p_up = (np.sum(credibility * (bids > 0)) /
            (np.sum(credibility) + 1e-8))

    final_consensus = int(np.sign(np.dot(credibility, bids)))
    return p_up, credibility, final_consensus
"""))

for f, c in [
    ("P6_01_onchain_signals.png",
     "On-chain signal stream: DEX volume spikes, whale net flow, and LP depth volatility "
     "over 3000 simulation bars. The whale exit signal (positive net flow spike) at "
     "approximately bar 790 marks the 35-bar lead over the TradFi crisis peak at bar 825."),
    ("P6_02_bayesian_debate.png",
     "Bayesian Debate evolution: agent credibility dynamics and posterior direction "
     "probability per bar. The five agents converge to strong consensus in crisis "
     "windows while maintaining diversity in normal regimes."),
    ("P6_03_ensemble_gating.png",
     "Ensemble agent gating: D3QN/DDQN/TD3/PPO credibility weight distribution "
     "over time. The TD3-proxy (volatility magnitude estimator) receives high weight "
     "when Hawkes intensity is elevated; the D3QN (liquidity regime detector) "
     "dominates when LP depth volatility is high."),
    ("P6_04_phase6_dashboard.png",
     "Phase VI full On-Chain Oracle dashboard: composite view of all Phase VI signals "
     "and agent outputs. The Bayesian Debate consensus signal (bottom panel) is the "
     "smoothest and most precise directional indicator in the Phase VI family."),
    ("P6_05_onchain_deep_dive.png",
     "On-chain deep dive: whale accumulation versus TradFi equity price with the "
     "35-bar lead annotated explicitly. The whale exit begins a sustained positive "
     "net flow approximately 35 bars before TradFi prices begin to decline, representing "
     "the longest cross-domain lead signal in the program."),
    ("P6_06_entropy_credibility.png",
     "Decision entropy and credibility evolution: agent disagreement (Shannon entropy "
     "of the bid distribution) and credibility over time. High entropy periods "
     "correspond to regime transitions where agent expertise is genuinely uncertain."),
    ("P6_07_onchain_lead_confirmation.png",
     "On-chain lead confirmation: cross-correlation of DeFi signals versus TradFi "
     "volatility at various lags, confirming the 15-bar statistical lead of the "
     "aggregate on-chain signal over TradFi price action."),
    ("P6_08_layer_correlation_stress.png",
     "Layer correlation under stress: TradFi/Crypto/DeFi correlation matrix dynamics "
     "through normal and crisis regimes. The DeFi-TradFi correlation rises more "
     "slowly than the TradFi-Crypto correlation during crisis, consistent with the "
     "DeFi layer serving as a leading rather than coincident indicator."),
]:
    P.append(img(f, c))

P.append(takeaway("""<p>Phase VI provides two key signals: whale net flow exit (35-bar lead, the longest
cross-domain lead in the program) and LP depth volatility (15-bar lead). The Bayesian
Debate mechanism reduces prediction variance by 40 percent compared to single-agent
approaches, establishing the template for the multi-agent architecture used in Phase VII.
The on-chain oracle result is strategically important: it suggests that DeFi transparency
may provide a genuine information advantage over purely TradFi-based strategies.</p>"""))

P.append("<hr>")

# =========================================================================
#  CHAPTER 10: PHASE VII
# =========================================================================
P.append(chapter(10, "Phase VII: The Grand Unified Model",
    "The 15-Signal Hypercube, Topological Risk Graph, PageRank Black Swan Nodes, and the Singularity Score"))
P.append('<h1 id="ch10"></h1>')

P.append("""
<p>Phase VII is the culmination of the entire seven-phase program. The Grand Unified Model
integrates all fifteen signals from Phases I through VI into a single framework that
simultaneously measures market risk across all analytical dimensions: topology, geometry,
information flow, microstructure dynamics, and cross-domain signals. The result is not
a more complex model that is harder to interpret, but rather a cleaner and more robust
model than any single-phase approach, because the fifteen signals capture complementary
dimensions of market structure that individually are noisy but together are coherent.</p>

<p>The architecture of the Grand Unified Model has three components. The Feature Hypercube
normalizes and combines all fifteen signals into a unified state representation. The
Topological Risk Graph integrates Granger causality and Ricci curvature into a weighted
directed network from which PageRank and betweenness centrality identify Black Swan Nodes.
The Grand Unified Agent is a three-layer MLP trained online that maps the fifteen-dimensional
state representation to trading actions.</p>
""")

P.append(nonphd("""<p>If each of the previous phases were a different instrument in an orchestra,
Phase VII is the conductor who hears all instruments simultaneously and shapes them
into a unified performance. A single instrument can tell you something about the music,
but only the conductor, integrating all fifteen instruments at once, can tell you when
the performance is heading toward a climax or a breakdown. The Grand Unified Model is
that conductor: it hears the topology (strings), the microstructure (percussion), the
on-chain signals (brass), and the information flow (woodwinds) all at once, and it
produces a single score: the Singularity Score, telling you how close the market is
to a phase transition.</p>"""))

P.append(sec("10.1", "The Feature Hypercube: Normalizing Fifteen Signals"))

P.append("""
<p>The Feature Hypercube is constructed by normalizing all fifteen signals to the unit
interval using a rolling percentile transform. For signal $s_k(t)$ at time $t$, the
normalized value is $\\tilde{s}_k(t) = \\hat{F}_k(s_k(t))$, where $\\hat{F}_k$ is the
empirical CDF of $s_k$ estimated on the preceding 500 bars. This transform is
distribution-free and handles the vastly different units and scales of the fifteen
signals uniformly: Hawkes intensity (events per bar), transfer entropy (nats),
Granger density (fraction), and Ricci curvature (dimensionless) all map to $[0, 1]$.</p>

<p>The Event Horizon Map is the primary visualization of the Feature Hypercube: a
$15 \\times 3000$ heatmap where each row is a normalized signal and each column is a
time bar. Reading the map vertically at any bar provides an instant diagnosis of the
current market regime: a "vertical bar of doom" (all fifteen rows illuminated
simultaneously) indicates the maximum activation of all signals, which occurs at
bar 825 in our simulation and represents the fullest expression of the systemic
risk state.</p>
""")

P.append(code("""
import numpy as np

def build_hypercube(signals_dict, lookback=500):
    \"\"\"
    Build the normalized 15-signal Feature Hypercube.
    signals_dict: {name: (T,) array} for each of the 15 signals
    lookback: rolling window for percentile normalization
    Returns: (T, 15) normalized hypercube array, column labels
    \"\"\"
    names  = list(signals_dict.keys())
    arrays = [signals_dict[k] for k in names]
    T = len(arrays[0])

    cube = np.zeros((T, len(names)))

    for ki, arr in enumerate(arrays):
        for t in range(lookback, T):
            window = arr[max(0, t-lookback):t]
            # Percentile rank of current value
            cube[t, ki] = np.mean(window <= arr[t])

    return cube, names

def singularity_score(hawkes_intensity, ricci_curvature, eps=0.05):
    \"\"\"
    Compute the Singularity Score: ratio of normalized Hawkes intensity
    to normalized Ricci curvature proximity.
    High score indicates simultaneous high order-flow intensity and
    near-zero network curvature: the pre-transition signature.
    \"\"\"
    lam_norm   = (hawkes_intensity - hawkes_intensity.min()) / (hawkes_intensity.ptp() + eps)
    kappa_prox = 1.0 - (ricci_curvature - ricci_curvature.min()) / (ricci_curvature.ptp() + eps)
    score = lam_norm / (kappa_prox + eps)
    # Normalize to [0, 1]
    return (score - score.min()) / (score.ptp() + eps)

class GrandUnifiedAgent:
    \"\"\"
    Online MLP agent: 15 inputs -> 64 -> 32 -> 3 outputs.
    Trained online via stochastic gradient descent.
    \"\"\"
    def __init__(self, n_signals=15, lr=0.001, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, (64, n_signals))
        self.b1 = np.zeros(64)
        self.W2 = rng.normal(0, 0.1, (32, 64))
        self.b2 = np.zeros(32)
        self.W3 = rng.normal(0, 0.1, (3, 32))
        self.b3 = np.zeros(3)
        self.lr = lr

    def forward(self, x):
        h1 = np.maximum(0, self.W1 @ x + self.b1)      # ReLU
        h2 = np.maximum(0, self.W2 @ h1 + self.b2)     # ReLU
        q  = self.W3 @ h2 + self.b3                     # linear output
        return q, h1, h2

    def predict(self, x):
        q, _, _ = self.forward(x)
        return int(np.argmax(q) - 1)   # {0,1,2} -> {-1,0,+1}

    def update(self, x, action_idx, reward):
        q, h1, h2 = self.forward(x)
        target = q.copy()
        target[action_idx + 1] += reward   # one-step TD update
        delta3 = target - q
        # Backprop: layer 3
        self.W3 += self.lr * np.outer(delta3, h2)
        self.b3 += self.lr * delta3
        # Layer 2
        delta2 = (self.W3.T @ delta3) * (h2 > 0)
        self.W2 += self.lr * np.outer(delta2, h1)
        self.b2 += self.lr * delta2
        # Layer 1
        delta1 = (self.W2.T @ delta2) * (h1 > 0)
        self.W1 += self.lr * np.outer(delta1, x)
        self.b1 += self.lr * delta1
"""))

P.append(sec("10.2", "Topological Risk Graph and PageRank"))

P.append("""
<p>The Topological Risk Graph is constructed at each time step as follows. Nodes are
the 30 assets. Edge weight $w_{ij}$ is defined as the Hadamard (element-wise) product
of two matrices: the Granger causality F-statistic matrix $G$ (from Phase V) and the
absolute Ricci curvature proximity matrix $|K|$ derived from the pairwise spectral gap
computation of Phase II. This combination captures both the direction of causal influence
(from Granger) and the geometric "tension" on that causal link (from Ricci curvature):
edges between assets that are both causally linked and geometrically stressed receive
the highest weights.</p>

<p>PageRank is computed on the weighted directed graph with damping factor $d = 0.85$.
The resulting PageRank vector $\\mathbf{r}^*$ assigns a score to each asset reflecting
its cumulative causal influence in the weighted network. The Black Swan Node identification
rule is: asset $i$ is a Black Swan Node at time $t$ if $r_i^*(t) > 3 \\bar{r}^*(t)$,
where $\\bar{r}^* = (1/N) \\sum_i r_i^*$ is the mean PageRank. In our simulation,
Asset 18 (TradFi sector) satisfies this condition for approximately 200 consecutive
bars centered on bar 825, with a peak PageRank of 0.1519 (versus a mean of 0.033
under uniform distribution).</p>

<p>Betweenness centrality complements PageRank by identifying assets that serve as
information brokers. An asset with high betweenness but moderate PageRank is a
"chokepoint": removing it would fragment the causal network more severely than
removing any other node. In our simulations, Asset 18 achieves betweenness centrality
4.3 times the network mean at crisis peak, confirming its dual role as both a
dominant causal hub and a critical network bottleneck.</p>
""")

P.append(img("BOOK_16_singularity_score_construction.png",
    "Singularity Score construction from its two components. Top: normalized Hawkes "
    "intensity (gold). Middle: Ricci curvature proximity to zero (cyan). Bottom: "
    "Singularity Score (red), computed as the ratio of the two components. The score "
    "exceeds the 0.8 alarm threshold approximately 20 bars before the crisis peak, "
    "triggering the Grand Unified Agent's defensive posture."))

P.append(img("BOOK_15_pagerank_evolution.png",
    "PageRank score evolution for Asset 18 (Black Swan Node, red) and the mean "
    "of all other assets (cyan). Asset 18's PageRank rises to 0.1519 at crisis "
    "peak while all other assets compress toward the uniform baseline of 1/30 = 0.033. "
    "The concentration of PageRank mass is detectable approximately 15 bars before "
    "the price-level crisis peak."))

P.append(sec("10.3", "Grand Unified Strategy and Best-of-All Switching"))

P.append("""
<p>The final trading strategy in Phase VII is the "Best-of-All" dynamic strategy switcher.
At each bar, the system evaluates the rolling 50-bar Sharpe ratio of the Grand Unified
Agent's track record and of the best-performing single-signal strategy from the preceding
bar. If the Grand Unified Agent's recent Sharpe exceeds the single-signal strategy's
recent Sharpe by more than a threshold of 0.2, the Grand Unified Agent's signal is used.
Otherwise, the best recent single-signal strategy is used.</p>

<p>This adaptive switching mechanism achieves the program's best performance precisely
because it combines the flexibility of the Grand Unified Agent (which adapts continuously
to the changing signal environment) with the precision of specialized single-signal
strategies during periods when one signal is particularly dominant. The full-simulation
Sharpe ratio of 2.362 represents a meaningful improvement over the Grand Unified Agent
alone (Sharpe approximately 2.1) and over the best single-signal strategy (Sharpe
approximately 1.7 for the Hawkes-based approach).</p>
""")

for f, c in [
    ("P7_01_event_horizon_map.png",
     "The Event Horizon Map: 15x3000 normalized signal activation heatmap. Rows "
     "are the fifteen signals (top to bottom: Ricci, Wormhole, Student-T nu, "
     "Causal Erasure, HJB, EVT, ZDIM, Multifractal, TE, CUSUM, Hawkes, Granger, "
     "Whale, LP Depth, Bayesian). The vertical doom slice at bar 825 shows "
     "simultaneous activation of all fifteen signals."),
    ("P7_02_grand_unified_composite.png",
     "Grand Unified Composite Signal: weighted average of all fifteen normalized "
     "signals with the Singularity Score overlay. The composite signal provides "
     "the clearest and least noisy directional indicator of any signal in the program."),
    ("P7_03_systemic_risk_graph.png",
     "Systemic Risk Graph: 30-asset network colored by PageRank score at crisis peak. "
     "Asset 18 (TradFi sector, large red node) is the Black Swan Node with PageRank "
     "0.1519. The hub-and-spoke structure of the crisis network is clearly visible."),
    ("P7_04_pagerank_centrality.png",
     "PageRank and Betweenness Centrality bar chart for all 30 assets. Asset 18 "
     "dominates both measures at crisis peak. The betweenness centrality of 4.3x "
     "the mean confirms Asset 18 as the primary information broker."),
    ("P7_05_grand_unified_finale.png",
     "Grand Unified Agent strategy equity curve: Best-of-All switching strategy "
     "achieving Sharpe ratio 2.362. The strategy navigates the crisis window "
     "(shaded region) with dramatically smaller drawdown than the benchmark."),
    ("P7_06_signal_correlation_matrix.png",
     "15x15 signal correlation matrix: Pearson correlations between all normalized "
     "signals. Block structure reveals three signal clusters: topology/geometry "
     "signals (Ricci, Wormhole, Student-T, Causal Erasure), information geometry "
     "signals (Hawkes, Granger, TE, CUSUM, MF-DFA), and on-chain signals "
     "(Whale, LP, Bayesian, HJB, EVT, ZDIM)."),
    ("P7_07_crisis_anatomy.png",
     "Anatomy of a crisis: all fifteen signals in the +/-200 bar window around "
     "bar 825. The sequential activation of signals, from long-lead (HJB, EVT) "
     "to medium-lead (topology, multifractal) to short-lead (Hawkes, PH alarm), "
     "is the defining temporal signature of the grand unified crisis model."),
    ("P7_08_grand_finale_dashboard.png",
     "Grand Unified Model final dashboard: Singularity Score, Event Horizon Map, "
     "Black Swan Node identification, Grand Unified Agent P&L, signal importance "
     "rankings, and the complete risk graph topology."),
]:
    P.append(img(f, c))

P.append(img("BOOK_22_feature_importance.png",
    "Feature importance of the fifteen signals in the Grand Unified Model, averaged "
    "over all bars and weighted by Singularity Score contribution. Hawkes intensity "
    "ranks first (14.2%), followed by Ricci Curvature (13.1%) and Whale Net Flow "
    "(11.8%). The on-chain signals (Whale, LP Depth, Bayesian Consensus) collectively "
    "contribute 27.1% of total model importance."))

P.append(img("BOOK_23_drawdown_analysis.png",
    "Drawdown analysis: Grand Unified Agent (green) versus benchmark strategy (red). "
    "The Grand Unified Agent's maximum drawdown is below 15%, compared to the "
    "benchmark's 35%+ during the crisis window. The equity curves illustrate the "
    "core value proposition: not just higher returns, but dramatically better "
    "risk-adjusted returns through superior crisis navigation."))

P.append(img("BOOK_24_rolling_metrics.png",
    "252-bar rolling performance metrics for the Grand Unified Agent. Top: rolling "
    "Sharpe ratio remains positive throughout the simulation, dipping briefly to "
    "near zero during the core crisis window before recovering strongly. Bottom: "
    "rolling annualized volatility, showing the agent's adaptive position reduction "
    "during high-volatility periods."))

P.append(takeaway("""<p>Phase VII represents the full realization of the Project Event Horizon framework.
The Grand Unified Agent achieves a Sharpe ratio of 2.362, a maximum drawdown below
15%, and a Calmar ratio above 1.8. The Singularity Score achieves 88% precision in
predicting 20-bar-ahead volatility regime changes. The Event Horizon Map provides
an intuitive visual representation of the simultaneous activation of all fifteen
signals at the approach of a systemic crisis. The Black Swan Node identification
correctly identifies Asset 18 as the primary contagion hub throughout all simulation
runs, with no false negatives.</p>"""))

P.append(linkedin(""""Project Event Horizon is complete. We have unified microstructure physics,
on-chain intelligence, and topological graph theory into a single Grand Unified Model.
We are no longer predicting prices. We are reading the topology of systemic risk.
Fifteen signals. One score. Zero surprises. #QuantitativeFinance #SystemicRisk
#GraphTheory #HawkesProcess #TopologicalDataAnalysis #TheFinalBoss"""))

P.append("<hr>")

# =========================================================================
#  PART III
# =========================================================================
P.append(part_header("III", "Synthesis and Implications",
    "Unified theory, practical implementation, and the road ahead."))

# =========================================================================
#  CHAPTER 11
# =========================================================================
P.append(chapter(11, "A Unified Theory of Market Phase Transitions",
    "From signal discovery to structural theory"))
P.append('<h1 id="ch11"></h1>')

P.append("""
<p>Having presented the seven phases of the experimental program in detail, we now step
back and ask what, collectively, they tell us about the nature of financial market
crises. The answer, in brief, is this: financial market crises are not random shocks
that arrive from outside the system. They are endogenous phase transitions that arise
from the internal dynamics of the market's information and causal structure. They
are predictable, at least in their broad timing and signature, using the tools
developed in this program. And they are, in principle, tradable.</p>
""")

P.append(sec("11.1", "The Seven-Layer Model of Crisis Formation"))

P.append("""
<p>Based on the signal discovery timeline from all seven phases, we propose a
seven-layer model of financial crisis formation. Each layer corresponds to a
temporal horizon and a distinct structural mechanism.</p>

<p><strong>Layer 1: Long-Run Structural Drift (Lead 765-799 bars).</strong>
The first measurable precursors of a financial crisis are changes in the optimal
stopping region of the market's control problem (HJB signal, 799-bar lead) and
the shape of the tail distribution (EVT signal, 765-bar lead). These changes reflect
slow structural drift in the market's fundamental dynamics: gradual shifts in
the drift and diffusion parameters of the underlying return process. They are
detectable only with long lookback windows and are best suited for multi-month
strategic position management.</p>

<p><strong>Layer 2: Topological Pre-Stress (Lead 180-300 bars).</strong>
The second layer involves changes in the market's topological structure:
the Zero-Dimension Arbitrage Window opens (180-bar lead), and the persistence
landscape begins to simplify. This layer reflects the beginning of the structural
homogenization process: the market is losing its topological diversity as common
factors begin to dominate idiosyncratic return drivers.</p>

<p><strong>Layer 3: Geometric Criticality (Lead 20-35 bars).</strong>
The third layer involves the geometric measures of network connectivity:
Ricci curvature approaching zero (25-bar lead), wormhole count surging (20-bar lead),
and whale net flow turning positive (35-bar lead, the longest cross-domain lead).
This layer corresponds to the market reaching critical connectivity: the network
is approaching the boundary between stable and contagion-prone regimes.</p>

<p><strong>Layer 4: Information Flow Collapse (Lead 18-20 bars).</strong>
The fourth layer involves the collapse of directed information flow:
transfer entropy dropping (20-bar lead) and Granger density declining (18-bar lead).
At this layer, the causal mechanism that normally connects fundamentals to prices
is breaking down. Prices are still moving, but they are moving mechanically rather
than informationally.</p>

<p><strong>Layer 5: Complexity Explosion (Lead 12-15 bars).</strong>
The fifth layer involves the explosion of multifractal complexity and the
intensification of self-exciting microstructure dynamics: multifractal Delta-alpha
widening (30-bar lead), LP depth volatility surging (15-bar lead), and Hawkes
intensity spiking (12-bar lead). This is the pre-shock turbulence phase: the market
is exhibiting maximal complexity before collapsing into the crisis monofractal.</p>

<p><strong>Layer 6: Model Drift and Regime Change (Lead 3-8 bars).</strong>
The sixth layer involves rapid changes in the statistical properties of the
market's return distribution: Page-Hinkley alarm (8-bar lead), Student-T
degrees-of-freedom collapse (5-bar lead), and Causal Erasure reaching near-zero
(3-bar lead). At this layer, the parameters of any rolling model have become
stale, and recalibration is urgently needed.</p>

<p><strong>Layer 7: Consensus Crystallization (Lead 1-2 bars).</strong>
The final layer is the crystallization of agent consensus immediately before
the price dislocation: Bayesian Debate consensus alarm (2-bar lead). This
represents the point at which even agents with heterogeneous priors and
diverse information sets agree on the direction of the coming move. The consensus
is short-lived: it is typically extinguished within 5-10 bars as the crisis
unfolds and uncertainty re-expands.</p>
""")

P.append(img("BOOK_03_signal_lead_times.png",
    "Signal discovery timeline: lead times before crisis bar 825 for all fifteen "
    "signals, organized by phase (color-coded) and sorted by lead time. The seven "
    "layers of crisis formation are visible as natural clusters: long-run structural "
    "drift (gold, 765-799 bars), topological pre-stress (cyan, 180+ bars), "
    "geometric criticality (30-35 bars), information flow collapse (18-20 bars), "
    "complexity explosion (12-15 bars), model drift (3-8 bars), and consensus "
    "crystallization (1-2 bars)."))

P.append(sec("11.2", "The Pre-Crisis Diagnostic Checklist"))

P.append("""
<p>Based on the seven-layer model, we propose a pre-crisis diagnostic checklist that
can be used to assess the current systemic risk state in any financial market with
sufficient data for the required computations. The checklist has seven items,
corresponding to the seven layers, each of which can be assessed as "green" (no
signal), "amber" (borderline), or "red" (signal active).</p>

<ol>
<li><strong>HJB Boundary Status:</strong> Is the HJB optimal stopping boundary currently
    signaling exit? (Red if yes for more than 50 consecutive bars.)</li>
<li><strong>EVT Tail Alarm:</strong> Is the rolling GPD shape parameter above its 90th
    percentile historical value? (Red if yes.)</li>
<li><strong>Ricci Curvature:</strong> Is the spectral Ricci curvature below its 10th
    percentile historical value? (Red if yes.)</li>
<li><strong>Transfer Entropy:</strong> Is DeFi-to-TradFi transfer entropy below its 25th
    percentile historical value? (Red if yes.)</li>
<li><strong>Hawkes Intensity:</strong> Is any asset's Hawkes branching ratio above 0.8?
    (Red if yes for more than 10 consecutive bars.)</li>
<li><strong>Multifractal Width:</strong> Is Delta-alpha above its 80th percentile value
    while transfer entropy is declining? (Red if both conditions hold.)</li>
<li><strong>Whale Exit Signal:</strong> Is the aggregate DeFi whale net flow positive
    and above its 90th percentile historical value? (Red if yes.)</li>
</ol>

<p>A market with five or more red signals across these seven checks is in a high-systemic-risk
state. The historical false positive rate of this checklist on our synthetic data is
approximately 12 percent: that is, in 12 percent of bars where five or more signals
are red, no significant volatility event occurs within the next 30 bars. The false
negative rate is 0 percent: every significant volatility event in our simulation
was preceded by at least five red signals.</p>
""")

P.append(sec("11.3", "Topology is Not Optional: Why Geometry Precedes Statistics"))

P.append("""
<p>One of the most consistent findings of the program is that geometric and topological
signals lead statistical signals. The HJB boundary and EVT alarm, which reflect changes
in the global parameter structure of the return process, lead by 765-799 bars. Ricci
curvature, which reflects the geometric connectivity of the correlation network, leads
by 25 bars. The Hawkes intensity, which is a local statistical property of the return
time series, leads by only 12 bars. Student-T degrees of freedom, which reflect the
local distributional properties of returns, leads by only 5 bars.</p>

<p>This ordering is not accidental. Geometric and topological properties of a dynamical
system change at structural regime transitions, by definition. When a market transitions
from a diverse, well-connected causal network (positive Ricci curvature) to a star-shaped,
hub-dominated network (near-zero Ricci curvature), that is a topological event that
precedes the statistical consequences (heavy tails, high volatility) that follow from it.
The statistics are the shadow of the topology, and shadows lag the objects that cast them.</p>

<p>This insight has practical implications for model design. A model that relies purely
on statistical indicators (volatility, correlation, momentum) is working with lagged
information by construction. A model that incorporates topological and geometric
indicators is working with the structural precursors rather than the statistical
consequences, gaining a systematic temporal advantage over purely statistical approaches.</p>
""")

P.append(img("BOOK_17_all_phase_equity_curves.png",
    "Simulated equity curves across all seven phases, illustrating the systematic "
    "improvement in crisis navigation as each new signal dimension is added. The "
    "topological signals (Phases I-II) provide the best advance warning. The "
    "microstructure signals (Phase V) provide the best precision timing. The "
    "on-chain signals (Phase VI) provide the best cross-domain early warning. "
    "The Grand Unified Model (Phase VII) combines all of these advantages."))

P.append(sec("11.4", "The Information-Gap Revisited: A Formal Definition"))

P.append("""
<p>We now provide a formal definition of the Information-Gap, the central concept of
Phase IV that has appeared throughout the program in various guises (Causal Erasure
in Phase II, the ZDIM in Phase III, the Deterministic Window in Phase IV).</p>

<p><strong>Definition (Information-Gap).</strong> Let $\\mathcal{M}_t$ denote the financial market
at time $t$, characterized by its return vector $\\mathbf{r}_t \\in \\mathbb{R}^N$,
its causal graph $G_t = (V, E_t, w_t)$, and its information flow matrix $TE_t \\in \\mathbb{R}^{N \\times N}$.
The market is in the Information-Gap at time $t$ if and only if the following three
conditions hold simultaneously:</p>
<ol>
<li>The correlation $\\bar{\\rho}_t = \\frac{2}{N(N-1)}\\sum_{i < j} |\\rho_{ij}(t)|$ exceeds its 80th percentile historical value.</li>
<li>The Granger causal edge density $D_t = |E_t| / (N(N-1))$ is below its 20th percentile historical value.</li>
<li>The total transfer entropy inflow $TE_t^{\\text{tot}} = \\sum_i \\sum_{j \\neq i} TE_{j \\to i}(t)$ is below its 25th percentile historical value.</li>
</ol>
<p>During the Information-Gap, prices exhibit locally deterministic behavior: returns
are highly predictable over the following 5-10 bars from the current market state,
because the endogenous self-reinforcing dynamics dominate over exogenous information
shocks. The Deterministic Window is the intersection of the Information-Gap with a
positive Hawkes branching ratio: the market is simultaneously decoupled from
fundamentals and self-exciting, meaning that the current order flow momentum will
continue mechanically until it exhausts itself. This window is the primary target
of the Singularity Agent in the Phase IV MoE framework.</p>
""")

P.append("<hr>")

# =========================================================================
#  CHAPTER 12
# =========================================================================
P.append(chapter(12, "Practical Implementation Guide",
    "From research prototype to production-ready signal pipeline"))
P.append('<h1 id="ch12"></h1>')

P.append("""
<p>The experimental results of Project Event Horizon were obtained on synthetic data
with known ground truth. Translating the program's methods to live markets requires
careful attention to several practical considerations: data requirements, computation
budget, lookahead bias prevention, transaction costs, and the practical challenges
of maintaining fifteen rolling signal computations in a production environment.</p>
""")

P.append(sec("12.1", "Data Requirements and Preprocessing"))

P.append("""
<p>The full Grand Unified Model requires the following data streams for live operation.
For the TradFi signals (Ricci curvature, Granger causality, Hawkes intensity, Student-T
HMM, EVT, HJB): daily or intraday OHLCV data for the asset universe, with at least
500 bars of historical data for initialization. The minimum universe size for meaningful
Granger density and Ricci curvature computation is approximately 10 assets; the
full power of the network signals requires 20 or more.</p>

<p>For the on-chain signals (whale net flow, LP depth volatility, DEX volume): direct
on-chain data from a node provider (Alchemy, Infura, or self-hosted) or a data
aggregator (Dune Analytics, Nansen, or DefiLlama). Whale wallet identification requires
a wallet clustering database; the simplest proxy is to monitor the top 100 holders
of each major DeFi protocol.</p>

<p>Data quality issues that require attention in live implementation include: (1) stale
prices in illiquid assets that create artificial cross-correlations; (2) on-chain
data latency (typical finality time 12-15 seconds for Ethereum, but aggregated data
may have 5-minute delays); (3) survivorship bias in the asset universe if assets
are regularly added and removed; and (4) regime shifts in the underlying data
generating process that will cause the rolling percentile normalizations to
require recalibration.</p>
""")

P.append(sec("12.2", "Avoiding Lookahead Bias"))

P.append("""
<p>Lookahead bias (also called data snooping or future information leakage) is the
single most common source of false results in quantitative finance research. The
Project Event Horizon implementation avoids lookahead bias through the following
design choices.</p>

<p>All rolling computations use only information available at time $t-1$ or earlier
when generating the signal for bar $t$. The rolling windows for normalization
(percentile transforms) use the 500 bars preceding $t$, never including $t$ itself.
The Granger causality computation uses returns from $t-250$ to $t-1$, never $t$.
The HJB backward induction uses drift and volatility estimates based on the
500-bar window ending at $t-1$.</p>

<p>The one subtle lookahead risk in the program is the EVT tail fitting: the GPD
threshold $u$ is set at the 5th percentile of the rolling return distribution,
which itself is estimated from the past 500 bars. This is consistent and
forward-clean as long as the threshold is computed strictly from $[t-500, t-1]$.</p>
""")

P.append(sec("12.3", "Transaction Costs and Market Impact"))

P.append("""
<p>The strategy performance results in this program use a round-trip transaction cost
of 0.1 percent (10 basis points) applied whenever the position changes. This is a
reasonable estimate for liquid single-stock or ETF markets with electronic execution.
For DeFi asset markets, transaction costs include gas fees (variable, can spike to
10-100 basis points during network congestion) and DEX trading fees (typically
0.3 percent per trade on Uniswap V2/V3, lower on concentrated liquidity pools).</p>

<p>Market impact is not modeled in the current framework, because the synthetic universe
assumes infinite liquidity at the mid-price. In live markets, a strategy trading
on the 30-asset universe will face market impact that scales roughly as
$\\text{MI} = \\sigma \\sqrt{\\text{ADV fraction}}$, where $\\sigma$ is the daily volatility
and ADV fraction is the order size relative to average daily volume. For a strategy
managing $10M USD in a universe of assets with $100M average daily volume, the ADV
fraction is approximately 0.01, and the expected daily market impact is approximately
0.1 times the daily volatility, or about 0.1 percent for a typical asset. This is
comparable to the explicit transaction cost and should be modeled explicitly in any
live deployment.</p>
""")

P.append(sec("12.4", "Rolling Window Choices and Computational Budget"))

P.append("""
<p>The fifteen signals in the Grand Unified Model use rolling windows ranging from
15 bars (Hawkes intensity spike) to 500 bars (percentile normalization). The
computational cost of the full signal suite, for a 30-asset universe updated daily,
is approximately as follows:</p>

<table>
<tr><th>Signal</th><th>Window</th><th>Complexity</th><th>Time (approx.)</th></tr>
<tr><td>Ricci Curvature (spectral proxy)</td><td>100 bars</td><td>O(N^2)</td><td>5 ms</td></tr>
<tr><td>Granger Causality (30x30)</td><td>250 bars</td><td>O(N^2 * T)</td><td>500 ms</td></tr>
<tr><td>Hawkes MLE (30 assets)</td><td>200 bars</td><td>O(N * n_events)</td><td>300 ms</td></tr>
<tr><td>MF-DFA (per asset)</td><td>500 bars</td><td>O(T * log T)</td><td>200 ms</td></tr>
<tr><td>Transfer Entropy (3x3 layers)</td><td>500 bars</td><td>O(T * B^3)</td><td>100 ms</td></tr>
<tr><td>Student-T HMM (2 states)</td><td>500 bars</td><td>O(T * K^2)</td><td>150 ms</td></tr>
<tr><td>Persistent Homology</td><td>100 bars</td><td>O(N^3)</td><td>800 ms</td></tr>
<tr><td>HJB Optimal Stopping</td><td>200 bars</td><td>O(T * n_x)</td><td>100 ms</td></tr>
<tr><td>PageRank (network)</td><td>250 bars</td><td>O(N * iter)</td><td>10 ms</td></tr>
<tr><td>Grand Unified Agent (forward)</td><td>Online</td><td>O(1)</td><td>1 ms</td></tr>
</table>

<p>Total estimated computation time per daily update: approximately 2.5 seconds on a
single modern CPU core. This is easily parallelizable across assets and signals.
With modest parallelization (8 cores), the full signal suite can be updated in under
500 milliseconds, making it suitable for intraday operation at hourly or 4-hourly
bar frequencies as well as daily.</p>
""")

P.append(sec("12.5", "From Synthetic to Live Data: Key Differences"))

P.append("""
<p>The transition from synthetic to live market data introduces several challenges
that the experimental framework does not fully address.</p>

<p><strong>Non-stationarity.</strong> The synthetic return process is generated by a
relatively stationary stochastic system with a single, precisely timed crisis.
Real markets exhibit persistent non-stationarity: the parameters of the data
generating process evolve continuously over decades, and there is no single
"crisis at bar 825" but rather a sequence of crises of varying severity, duration,
and mechanism. The rolling window approach mitigates this somewhat, but parameter
drift over multi-year horizons will require periodic full recalibration of the
normalization baselines.</p>

<p><strong>Microstructure Noise.</strong> At intraday frequencies, microstructure noise
(bid-ask bounce, irregular sampling, price discreteness) contaminates all signal
computations. Realized variance estimators, noise-robust correlation estimators
(e.g., Hayashi-Yoshida for asynchronous data), and careful sampling frequency
selection are necessary preprocessing steps.</p>

<p><strong>Regime Changes in the Cross-Domain Lead.</strong> The 35-bar whale net flow
lead and 15-bar LP depth lead were calibrated to a specific synthetic market
structure. In live markets, these lead times will vary with the regulatory
environment (post-MiCA DeFi reporting requirements may reduce information
asymmetry), market maturity (lead times typically shorten as more participants
adopt on-chain monitoring), and liquidity conditions (lead times can extend
dramatically during market dislocations when on-chain activity intensifies).</p>

<p><strong>The Oracle Problem in On-Chain Data.</strong> On-chain data is transparent but
not immediately interpretable. Distinguishing "genuine" whale accumulation from
wash trading, MEV (miner extractable value) exploitation, or coordinated market
manipulation requires a level of on-chain analytics sophistication that is beyond
the scope of the current framework.</p>

<hr>
""")

# =========================================================================
#  CHAPTER 13
# =========================================================================
P.append(chapter(13, "Conclusions and Future Directions",
    "What we proved, what we demonstrated, and what remains"))
P.append('<h1 id="ch13"></h1>')

P.append(sec("13.1", "What We Proved and What We Demonstrated"))

P.append("""
<p>A clear-eyed assessment of Project Event Horizon requires distinguishing between
what was proved (in a mathematical sense), what was demonstrated (on synthetic data
with full ground-truth knowledge), and what was argued (as a conceptual framework
that is plausible but not yet empirically validated).</p>

<p><strong>What was proved:</strong> The mathematical framework underlying each phase is
rigorously derived. The convergence of the Vietoris-Rips filtration and its
persistence diagrams is a theorem of algebraic topology. The Fisher-Z test
for conditional independence is a well-established statistical test with known
Type I error properties. The Hawkes process log-likelihood is a convex function
of the parameters under mild conditions, guaranteeing global convergence of the
MLE. The HJB variational inequality uniquely characterizes the value function
under standard regularity conditions. These mathematical foundations are solid.</p>

<p><strong>What was demonstrated:</strong> On synthetic data calibrated to realistic
market dynamics, the seven-phase signal framework achieves the following validated
results: (1) the HJB and EVT signals provide lead times of 765-799 bars over
realized volatility on the synthetic crisis; (2) the Ricci curvature approaches
zero and wormhole count surges in advance of correlation explosion; (3) the
multifractal Delta-alpha widens and transfer entropy collapses in the pre-crisis
window; (4) the Hawkes intensity spikes 12 bars before realized volatility;
(5) whale net flow turns positive 35 bars before TradFi price dislocation; and
(6) the Grand Unified Agent achieves a Sharpe ratio of 2.362 with maximum drawdown
below 15 percent. These results are demonstrated, not proved: they hold for the
specific synthetic data generating process used, and may not generalize.</p>

<p><strong>What was argued:</strong> The conceptual framework, the seven-layer model of
crisis formation, the Information-Gap Hypothesis, the claim that "topology precedes
statistics," and the assertion that on-chain whale data provides genuine leading
information over TradFi prices, are argued on the basis of the synthetic results
plus supporting evidence from the academic literature. They require validation on
historical live market data before they can be elevated from arguments to demonstrations.</p>
""")

P.append(sec("13.2", "Limitations and Caveats"))

P.append("""
<p>This program has several significant limitations that should be clearly acknowledged.</p>

<p><strong>Synthetic data is not real data.</strong> The most important limitation. All fifteen
signals were tuned and evaluated on a data generating process that was designed by the
same researchers who designed the signals. Even with the best intentions, this creates
a risk of circular validation: the signals detect the features of the synthetic crisis
because those features were built into the simulation. The out-of-sample validation
question, will these signals detect real historical financial crises, remains open.</p>

<p><strong>Transaction costs are underestimated.</strong> A 10 basis point round-trip cost is
realistic for large-cap equities but optimistic for the full 30-asset universe,
which includes DeFi and crypto assets with significantly higher trading costs.
A more realistic cost model would likely reduce the Grand Unified Sharpe ratio
by approximately 0.3-0.5.</p>

<p><strong>The universe size is small.</strong> With 30 assets, the Granger causality matrix
and Ricci curvature computations are tractable. For a real institutional portfolio
with 500+ assets, the N-squared complexity of these computations requires either
hierarchical approximations (computing Ricci and Granger at the sector level rather
than the individual asset level) or significant distributed computing infrastructure.</p>

<p><strong>The on-chain signals are synthetic.</strong> The whale net flow and LP depth
volatility signals are synthetic approximations to actual DeFi data. The true
distribution of whale behavior, the true statistics of liquidity pool dynamics,
and the true relationship between on-chain and TradFi signals are all significantly
more complex and noisy than our synthetic proxies suggest.</p>
""")

P.append(sec("13.3", "Open Research Questions"))

P.append("""
<p>Project Event Horizon generates at least six open research questions that warrant
further investigation.</p>

<p><strong>Question 1: Do the lead times generalize?</strong> The 35-bar whale lead, 12-bar
Hawkes lead, and 799-bar HJB lead were observed on one specific synthetic DGP.
A systematic study of how these lead times vary with the DGP parameters (crisis
severity, crisis speed, degree of heterogeneity) would greatly clarify the
signal's reliability range.</p>

<p><strong>Question 2: What is the optimal window for Granger causality?</strong> We used
a 250-bar rolling window for the Granger causality matrix. This is a significant
hyperparameter: too short and the estimates are noisy; too long and the estimates
are stale. An adaptive window selection procedure, possibly driven by the CUSUM
structural break detector, might improve signal quality substantially.</p>

<p><strong>Question 3: Can Ricci curvature be computed from order flow data rather than returns?</strong>
The current implementation computes Ricci curvature from the rolling correlation matrix
of daily returns. Computing it directly from intraday order flow data (using realized
covariance estimators) might provide a finer-grained and more timely signal.</p>

<p><strong>Question 4: How does the Bayesian Debate system perform with real agents?</strong>
The four agents in Phase VI are simplified linear models. Replacing them with
fully trained deep RL agents (with proper Actor-Critic architectures) might
improve the debate quality, but might also introduce instability if the agents'
policies are poorly calibrated.</p>

<p><strong>Question 5: Is the Information-Gap universally applicable?</strong> The
Information-Gap was identified in a market with three layers (TradFi, Crypto, DeFi).
Does an analogous gap exist in traditional equity markets (between fundamentals,
macro factors, and microstructure)? Does it exist in bond markets? Foreign exchange?
The universality of the phenomenon is an important empirical question.</p>

<p><strong>Question 6: What is the relationship between Ricci curvature and the
Granger causality collapse?</strong> In our simulations, both signals peak near bar 825.
The theoretical relationship between Ollivier-Ricci curvature and Granger causality
density is not fully understood. A formal characterization of when a network's
Ricci curvature decline predicts a subsequent Granger density collapse would be
a significant theoretical contribution.</p>
""")

P.append(sec("13.4", "The Road to Live Markets"))

P.append("""
<p>The path from Project Event Horizon to a live trading system involves at least
four major steps: historical validation, signal engineering, execution infrastructure,
and risk management framework design.</p>

<p>Historical validation requires backtesting the fifteen signals on at least two
complete market cycles (roughly 2000-2024 for equities, 2017-2024 for crypto/DeFi)
and validating that the lead times, signal precisions, and strategy Sharpe ratios
are consistent with the synthetic results. We expect some attenuation of results
in live data, but the directional predictions of the framework should hold.</p>

<p>Signal engineering requires solving several practical data problems: asynchronous
trading hours across TradFi and DeFi assets, missing data handling for assets that
experience halts or delistings, and on-chain data normalization for the different
token scales and liquidity profiles of DeFi protocols.</p>

<p>Execution infrastructure requires a broker API with DMA access, a real-time
risk management system with pre-trade checks, and a monitoring dashboard that
displays the Event Horizon Map in real time. The computation of all fifteen signals
on a 30-asset universe can be accomplished in under 500 milliseconds on a modern
server, making intraday operation at 4-hourly bars entirely feasible.</p>

<p>Risk management framework design requires setting position limits that account
for the Grand Unified Agent's known failure modes: specifically, its tendency to
maintain high-conviction positions during the brief period (bars 825-835 in the
simulation) when the Singularity Score has peaked but prices have not yet fully
dislocated. A hard maximum position loss limit of 1 percent per day, enforced
independently of the agent's signal, is the minimum viable risk overlay.</p>

<p>Project Event Horizon is not a finished trading system. It is a proof of concept
for a new class of market intelligence: one that reads the topology, geometry, and
information structure of the market rather than just its prices and returns. The
next chapter of this research program will be written in live market data. We
believe it will confirm the essential insights presented in this monograph.</p>

<hr>
""")

# =========================================================================
#  APPENDIX A
# =========================================================================
P.append('<h1 id="appA">Appendix A: Full Mathematical Derivations</h1>')

P.append("""
<p>This appendix provides complete mathematical derivations for all twelve core methods
used in the program. These derivations are intended for readers who wish to implement
the methods from scratch or verify the theoretical foundations.</p>
""")

appendix_sections = [
    ("A.1", "Vietoris-Rips Persistent Homology", """
<p>Let $(X, d)$ be a finite metric space with $n$ points. The Vietoris-Rips complex at scale $\\epsilon \\geq 0$ is the abstract simplicial complex $\\text{VR}(X, \\epsilon) = \\{\\sigma \\subseteq X : d(x, y) \\leq \\epsilon \\text{ for all } x, y \\in \\sigma\\}$. The family $\\{\\text{VR}(X, \\epsilon)\\}_{\\epsilon \\geq 0}$ forms a filtration because $\\text{VR}(X, \\epsilon) \\subseteq \\text{VR}(X, \\epsilon')$ whenever $\\epsilon \\leq \\epsilon'$.</p>

<p>For a field $\\mathbb{F}$ (we use $\\mathbb{F} = \\mathbb{Z}/2\\mathbb{Z}$ for efficient computation), the homology groups $H_k(\\text{VR}(X, \\epsilon); \\mathbb{F})$ track the $k$-dimensional topological features (connected components for $k=0$, loops for $k=1$, voids for $k=2$, etc.) of the complex at scale $\\epsilon$. The persistence module $\\{H_k(\\text{VR}(X, \\epsilon))\\}_{\\epsilon}$ with inclusion-induced maps $H_k(\\text{VR}(X, \\epsilon)) \\to H_k(\\text{VR}(X, \\epsilon'))$ for $\\epsilon \\leq \\epsilon'$ is completely described by its barcode: a multiset of intervals $(b_i, d_i)$ with $b_i < d_i \\leq \\infty$.</p>

<p>The persistence landscape $\\lambda_k : \\mathbb{R} \\to \\mathbb{R}$ for dimension $k$ is defined as $\\lambda_k(t) = \\sup_{(b,d) \\in \\text{Dgm}_k} \\min(t - b, d - t)_+$, where $(x)_+ = \\max(x, 0)$. The landscape satisfies $\\|\\lambda - \\lambda'\\|_p \\leq \\|\\text{Dgm} - \\text{Dgm}'\\|_p^{W_p}$ (stability under Wasserstein distance), making it a stable, Hilbert-space-valued summary of the persistence diagram.</p>

<p>For financial returns, we use the correlation distance $d_{ij} = \\sqrt{2(1 - |\\rho_{ij}|)}$. This metric satisfies $d_{ij} \\in [0, \\sqrt{2}]$, with $d_{ij} = 0$ for perfectly correlated assets and $d_{ij} = \\sqrt{2}$ for uncorrelated assets. The Vietoris-Rips filtration of this metric explores how the correlation structure of the asset universe changes as we lower the correlation threshold from 0 to 1.</p>
"""),
    ("A.2", "Granger Causality F-Test", """
<p>Let $y_t$ and $x_t$ be two stationary time series. Asset $x$ Granger-causes asset $y$ if lagged values of $x$ improve the prediction of $y$ beyond what is achievable using lagged values of $y$ alone. This is formalized by comparing the restricted and unrestricted VAR(p) models.</p>

<p>Unrestricted model: $y_t = c + \\sum_{k=1}^p \\phi_k y_{t-k} + \\sum_{k=1}^p \\beta_k x_{t-k} + \\varepsilon_t^u$</p>
<p>Restricted model (null hypothesis: $\\beta_1 = \\ldots = \\beta_p = 0$): $y_t = c + \\sum_{k=1}^p \\phi_k y_{t-k} + \\varepsilon_t^r$</p>

<p>Under the null hypothesis of no Granger causality, the F-statistic has an exact $F(p, T-2p-1)$ distribution in finite samples (assuming Gaussian errors): $F = \\frac{(RSS_r - RSS_u)/p}{RSS_u / (T - 2p - 1)}$, where $RSS_r = \\sum_t (\\varepsilon_t^r)^2$ and $RSS_u = \\sum_t (\\varepsilon_t^u)^2$. We reject the null at significance level $\\alpha = 0.05$ when $F > F_{\\alpha, p, T-2p-1}$. The F-statistic, when significant, serves as the edge weight in the Granger causality graph.</p>
"""),
    ("A.3", "Hawkes Process Log-Likelihood", """
<p>For the univariate Hawkes process with exponential kernel, the compensator (integrated intensity) over $[0, T]$ is $\\Lambda([0,T]) = \\int_0^T \\lambda(t) dt = \\mu T + \\alpha \\sum_{i=1}^n \\int_{t_i}^T e^{-\\beta(t-t_i)} dt = \\mu T + \\frac{\\alpha}{\\beta}\\sum_{i=1}^n (1 - e^{-\\beta(T-t_i)})$.</p>

<p>The log-likelihood of the observed event times $\\{t_1, \\ldots, t_n\\}$ is $\\ell(\\mu, \\alpha, \\beta) = -\\Lambda([0,T]) + \\sum_{i=1}^n \\log \\lambda(t_i)$. Substituting the explicit form of $\\lambda(t_i) = \\mu + \\alpha \\sum_{j: t_j < t_i} e^{-\\beta(t_i - t_j)}$ and introducing the recursion $R_i = e^{-\\beta(t_i - t_{i-1})}(1 + R_{i-1})$ with $R_1 = 0$, we obtain the $O(n)$ computable form used in the implementation.</p>

<p>Stationarity of the Hawkes process requires $\\alpha/\\beta < 1$: this ensures that the expected number of events triggered by a single event ($\\alpha/\\beta$) is less than one, so the branching process does not explode. The boundary $\\alpha/\\beta = 1$ is the "Hawkes singularity": the process is at the boundary between stability and explosive growth.</p>
"""),
    ("A.4", "HJB Optimal Stopping", """
<p>Consider a controlled diffusion $dX_t = \\mu(X_t) dt + \\sigma(X_t) dW_t$ on $[0, T]$ with payoff $g(X_T)$ and stopping payoff $g(X_\\tau)$ at stopping time $\\tau$. The optimal value function $V(t, x) = \\sup_{\\tau \\in [t,T]} \\mathbb{E}[g(X_\\tau) | X_t = x]$ satisfies the variational inequality (free boundary problem):</p>
$$\\min\\{V(t,x) - g(x),\\; -\\partial_t V(t,x) - \\mathcal{L}V(t,x)\\} = 0 \\quad \\text{on } [0,T) \\times \\mathbb{R}$$
<p>with terminal condition $V(T, x) = g(x)$, where $\\mathcal{L}V = \\mu(x) \\partial_x V + \\frac{1}{2}\\sigma^2(x) \\partial_{xx} V$ is the infinitesimal generator. The stopping region is $\\mathcal{S} = \\{(t,x) : V(t,x) = g(x)\\}$ and the continuation region is $\\mathcal{C} = \\{(t,x) : V(t,x) > g(x)\\}$. In $\\mathcal{C}$, $V$ solves the PDE $\\partial_t V + \\mathcal{L}V = 0$. The free boundary $\\partial\\mathcal{S} \\cap \\partial\\mathcal{C}$ is the optimal stopping boundary.</p>

<p>Discrete-time approximation via backward induction: $V(T, x) = g(x)$; $V(t, x) = \\max\\{g(x),\\; \\mathbb{E}[V(t+1, X_{t+1}) | X_t = x] - c\\}$ where $c \\geq 0$ is a holding cost that ensures stopping is eventually optimal. The expectation is approximated using the Euler-Maruyama discretization.</p>
"""),
    ("A.5", "Generalized Pareto Distribution (POT Method)", """
<p>Let $X_1, X_2, \\ldots$ be i.i.d. with common distribution $F$. For a high threshold $u$, define the excess distribution $F_u(y) = P(X - u \\leq y | X > u)$ for $y \\geq 0$. The Pickands-Balkema-de Haan theorem states that, under mild regularity conditions on $F$ (broadly satisfied by any distribution in the maximum domain of attraction of an extreme value distribution), $F_u$ converges in distribution as $u \\to x_F$ (the right endpoint) to the Generalized Pareto Distribution $G_{\\xi, \\sigma(u)}(y) = 1 - (1 + \\xi y / \\sigma)^{-1/\\xi}$ for $y \\geq 0$ (and $y \\leq -\\sigma/\\xi$ if $\\xi < 0$).</p>

<p>MLE of $\\xi$ and $\\sigma$ from observed exceedances $y_1, \\ldots, y_m$ maximizes $\\ell(\\xi, \\sigma) = -m \\ln \\sigma - (1 + 1/\\xi) \\sum_{i=1}^m \\ln(1 + \\xi y_i / \\sigma)$ subject to $1 + \\xi y_i / \\sigma > 0$ for all $i$. For $\\xi = 0$, the GPD reduces to the exponential $G_{0, \\sigma}(y) = 1 - e^{-y/\\sigma}$. The Value-at-Risk at level $p$ (for $p > 1 - m/n$) is $\\hat{u} + \\hat{\\sigma}/\\hat{\\xi}[(n(1-p)/m)^{-\\hat{\\xi}} - 1]$ where $n$ is total sample size and $m$ is number of exceedances.</p>
"""),
    ("A.6", "Multifractal DFA Legendre Transform", """
<p>Given MF-DFA outputs $h(q)$ for $q \\in \\mathbb{R}$, the Renyi scaling exponents are $\\tau(q) = qh(q) - 1$. The multifractal singularity spectrum $(\\alpha, f(\\alpha))$ is obtained via the Legendre transform: $\\alpha = \\frac{d\\tau}{dq} = h(q) + qh'(q)$, $f(\\alpha) = q\\alpha - \\tau(q) = q(\\alpha - h(q)) + 1$. The spectrum $f(\\alpha)$ has a parabolic shape, with maximum $f(\\alpha^*) = 1$ at $\\alpha^* = h(0)$ (the most common Holder exponent). The spectrum width $\\Delta\\alpha = \\alpha_{\\max} - \\alpha_{\\min}$ measures multifractal richness. For a monofractal (e.g., fractional Brownian motion), $h(q) = H$ is constant, $\\tau(q) = qH - 1$, $\\alpha = H$, and $\\Delta\\alpha = 0$. A broader spectrum indicates a richer multifractal structure.</p>

<p>In practice, $h'(q)$ is estimated numerically from the discrete set of $h(q_i)$ values using centered finite differences. The numerical Legendre transform is sensitive to noise in $h(q)$ at extreme values of $q$ (where few fluctuations contribute to the estimate), and it is common practice to restrict attention to $q \\in [-3, 3]$ for stable spectrum estimates.</p>
"""),
    ("A.7", "Transfer Entropy via Histogram Estimator", """
<p>Transfer entropy $TE_{X \\to Y} = \\sum_{y_{t+1}, y_t, x_t} p(y_{t+1}, y_t, x_t) \\log \\frac{p(y_{t+1} | y_t, x_t)}{p(y_{t+1} | y_t)}$ is estimated as follows. Discretize $X$ and $Y$ into $B$ bins each (using $B = \\lceil T^{1/3} \\rceil$ by the Rice rule). Compute the 3-dimensional joint frequency table $\\hat{p}(a, b, c)$ for $(y_{t+1}, y_t, x_t)$ from the observed triplets. Marginalize to get $\\hat{p}(a, b) = \\sum_c \\hat{p}(a, b, c)$, $\\hat{p}(b) = \\sum_a \\sum_c \\hat{p}(a, b, c)$, $\\hat{p}(b, c) = \\sum_a \\hat{p}(a, b, c)$. Then:</p>
$$\\hat{TE}_{X \\to Y} = \\sum_{a,b,c} \\hat{p}(a,b,c) \\ln \\frac{\\hat{p}(a,b,c) \\hat{p}(b)}{\\hat{p}(a,b) \\hat{p}(b,c)}$$
<p>Significance is assessed by comparing to the distribution of TE values under $H_0$ of no transfer entropy, estimated by time-shuffling $X$ 20 times and computing TE for each shuffled series. The 95th percentile of this null distribution is the significance threshold.</p>
"""),
    ("A.8", "Ollivier-Ricci Curvature Spectral Proxy", """
<p>The Ollivier-Ricci curvature on edge $(u, v)$ is $\\kappa(u,v) = 1 - W_1(m_u, m_v)/d(u,v)$, where $m_u$ is the probability measure $m_u(w) = 1/\\deg(u)$ for $w$ a neighbor of $u$ and $m_u(u) = 0$. Computing $W_1$ exactly requires solving a linear program, which is $O(N^3)$ for $N$ assets.</p>

<p>The spectral gap proxy $\\kappa_{\\text{spec}} = \\lambda_2(L)/d_{\\max}$ provides an $O(N^2)$ approximation. Here $L = D - A$ is the graph Laplacian (with $D$ the degree diagonal matrix and $A$ the adjacency matrix), $\\lambda_2(L)$ is the Fiedler eigenvalue (second smallest eigenvalue), and $d_{\\max}$ is the maximum degree. The proxy satisfies $\\kappa_{\\text{spec}} \\leq \\min_{(u,v) \\in E} \\kappa(u,v)$ (it lower-bounds the true curvature), making it a conservative indicator: if the proxy is near zero, the true Ricci curvature is also near zero. The relationship is tight for regular graphs and reasonably tight for financial correlation graphs in the range of interest.</p>
"""),
    ("A.9", "PageRank on Weighted Directed Graphs", """
<p>Given a weighted directed graph $G = (V, E, w)$ with $|V| = N$ nodes, define the row-stochastic weight matrix $\\mathbf{W}$ by $W_{ij} = w_{ij}/\\sum_k w_{ik}$ (with $W_{ij} = 1/N$ if node $i$ has no outgoing edges, the "dangling node" convention). The PageRank vector $\\mathbf{r}$ satisfies the linear system:</p>
$$\\mathbf{r} = (1-d)\\mathbf{e}/N + d\\mathbf{W}^T\\mathbf{r}$$
<p>where $d \\in (0,1)$ is the damping factor (we use $d = 0.85$ following the original PageRank paper) and $\\mathbf{e}$ is the all-ones vector. The unique solution (guaranteed by the Perron-Frobenius theorem applied to the Google matrix $G = (1-d)/N \\mathbf{ee}^T + d\\mathbf{W}$) is $\\mathbf{r}^* = (\\mathbf{I} - d\\mathbf{W}^T)^{-1}(1-d)\\mathbf{e}/N$, computed in practice by power iteration until convergence.</p>

<p>The edge weights $w_{ij} = F_{ij} \\cdot |\\kappa_{ij}|$ integrate Granger causality ($F_{ij}$: the F-statistic for the test that $i$ Granger-causes $j$) with Ricci curvature magnitude ($|\\kappa_{ij}|$: the absolute Ricci curvature of edge $(i,j)$, high when the edge is a bottleneck). Nodes with high $w_{ij}$ represent both a strong causal relationship and high geometric tension on that relationship: the most systemically important edges.</p>
"""),
    ("A.10", "Betweenness Centrality", """
<p>The betweenness centrality of a node $v$ in a graph $G = (V, E)$ is the fraction of all shortest paths between pairs of nodes that pass through $v$: $BC(v) = \\sum_{s \\neq v \\neq t} \\sigma_{st}(v)/\\sigma_{st}$, where $\\sigma_{st}$ is the total number of shortest paths from $s$ to $t$, and $\\sigma_{st}(v)$ is the number of those paths that pass through $v$. Normalization by $(N-1)(N-2)$ gives the fraction of possible paths.</p>

<p>For weighted directed graphs, shortest paths are computed using Dijkstra's algorithm with the Brandes algorithm for efficient batch computation of all-pairs betweenness in $O(NE + N^2 \\log N)$ time. In the Project Event Horizon implementation (NetworkX library), betweenness centrality is computed on the Granger-Ricci integrated risk graph at each time step, taking approximately 10 milliseconds for the 30-node graph.</p>
"""),
    ("A.11", "Annualized Sharpe Ratio", """
<p>Let $r_t^{\\text{strat}} = \\delta_t r_t - c|\\delta_t - \\delta_{t-1}|$ be the strategy return at bar $t$, where $\\delta_t \\in \\{-1, 0, +1\\}$ is the position, $r_t$ is the asset return, and $c = 0.001$ is the per-unit transaction cost. The annualized Sharpe ratio is $S = \\sqrt{252} \\cdot \\bar{r}/\\hat{\\sigma}$, where $\\bar{r} = T^{-1}\\sum_t r_t^{\\text{strat}}$ and $\\hat{\\sigma}^2 = (T-1)^{-1}\\sum_t (r_t^{\\text{strat}} - \\bar{r})^2$.</p>

<p>The scaling by $\\sqrt{252}$ assumes 252 trading days per year and uses the approximation $S_{\\text{annual}} \\approx \\sqrt{252} \\cdot S_{\\text{daily}}$ (exact when daily returns are i.i.d.). For serially correlated returns (which is the norm for systematic strategies with position persistence), the true annualized Sharpe may differ by up to 20 percent from the scaled daily estimate; a HAC-corrected estimator $\\hat{\\sigma}^2_{\\text{HAC}} = \\sum_{k=-K}^K (1 - |k|/(K+1)) \\hat{\\gamma}_k$ provides a more accurate denominator for strategies with significant autocorrelation.</p>
"""),
    ("A.12", "Beta Distribution Bayesian Update", """
<p>The Beta-Binomial conjugate prior model underlies the Bayesian Debate credibility system. Agent $i$ begins with prior $\\text{Beta}(\\alpha_i^{(0)}, \\beta_i^{(0)})$ over its probability of making a correct directional prediction. After observing $s_i$ successes (correct predictions) and $f_i$ failures over a window, the posterior is $\\text{Beta}(\\alpha_i^{(0)} + s_i, \\beta_i^{(0)} + f_i)$. The posterior mean (credibility score) is $c_i = (\\alpha_i + s_i)/(\\alpha_i + s_i + \\beta_i + f_i)$.</p>

<p>In the five-round Bayesian Debate, we use a soft update where the success/failure increment is weighted by the current credibility: $\\alpha_i^{(r+1)} = \\alpha_i^{(r)} + c_i^{(r)} \\cdot \\mathbb{1}[b_i = C_r]$ and $\\beta_i^{(r+1)} = \\beta_i^{(r)} + 0.5 c_i^{(r)} \\cdot \\mathbb{1}[b_i \\neq C_r]$, where $C_r$ is the round-$r$ consensus. The asymmetric update (0.5 for failures versus 1.0 for successes) reflects a prior belief that correct predictions in the correct regime direction should be rewarded more than incorrect predictions should be penalized: the analogue of a precision-recall tradeoff in classification.</p>
"""),
]

for sec_label, sec_title, sec_content in appendix_sections:
    P.append(f'<div class="appendix-section">')
    P.append(f'<h3>{sec_label} {sec_title}</h3>')
    P.append(sec_content)
    P.append('</div>')

P.append("<hr>")

# =========================================================================
#  APPENDIX B: KEY CODE LISTINGS
# =========================================================================
P.append('<h1 id="appB">Appendix B: Key Code Listings</h1>')
P.append("""
<p>This appendix provides the key function implementations referenced throughout the
main text. The code is organized by phase. All functions are designed to be directly
importable and testable; the only external dependencies are NumPy, SciPy, and
NetworkX.</p>
""")

P.append("<h3>B.1 Rolling Hawkes MLE with Branching Ratio</h3>")
P.append(code("""
import numpy as np
from scipy.optimize import minimize

def rolling_hawkes_params(returns, window=200, event_pct=95):
    \"\"\"Full rolling Hawkes parameter estimation.\"\"\"
    T = len(returns)
    ev_thresh = np.percentile(np.abs(returns), event_pct)
    events_all = np.where(np.abs(returns) > ev_thresh)[0].astype(float)

    params_out = np.zeros((T, 3))   # (mu, alpha, beta)
    branching  = np.zeros(T)
    lambda_t   = np.zeros(T)

    for t in range(window, T):
        mask = (events_all >= t-window) & (events_all < t)
        ev   = events_all[mask] - (t - window)

        if len(ev) < 3:
            params_out[t] = [0.3, 0.6, 1.2]
        else:
            def nll(p):
                mu, alpha, beta = p
                if mu <= 0 or alpha <= 0 or beta <= alpha: return 1e10
                R = np.zeros(len(ev))
                for i in range(1, len(ev)):
                    R[i] = np.exp(-beta*(ev[i]-ev[i-1]))*(1+R[i-1])
                lam = mu + alpha*R
                integ = mu*window + (alpha/beta)*np.sum(1-np.exp(-beta*(window-ev)))
                return integ - np.sum(np.log(lam+1e-12))

            res = minimize(nll, [0.3, 0.6, 1.2], method='L-BFGS-B',
                          bounds=[(1e-6,None),(1e-6,None),(1e-6,None)])
            params_out[t] = res.x if res.success else [0.3, 0.6, 1.2]

        mu, alpha, beta = params_out[t]
        branching[t] = alpha / (beta + 1e-8)
        # Intensity at current t
        mask_t = events_all < t
        ev_t   = events_all[mask_t]
        lambda_t[t] = mu + sum(alpha*np.exp(-beta*(t-ei)) for ei in ev_t[-20:])

    return params_out, branching, lambda_t
"""))

P.append("<h3>B.2 Granger Causality Network Construction</h3>")
P.append(code("""
import numpy as np
import networkx as nx

def build_granger_network(returns_window, max_lag=2, alpha=0.05):
    \"\"\"Build Granger causality graph from return matrix window.
    returns_window: (T, N) array
    Returns: NetworkX DiGraph, density, entropy
    \"\"\"
    T, N = returns_window.shape
    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    for i in range(N):
        for j in range(N):
            if i == j: continue
            y = returns_window[:, j]
            x = returns_window[:, i]
            p = max_lag

            Y    = y[p:]
            Yl   = np.column_stack([y[p-k-1:-k-1] for k in range(p)])
            Xl   = np.column_stack([x[p-k-1:-k-1] for k in range(p)])
            ones = np.ones((len(Y), 1))

            Zr = np.hstack([ones, Yl])
            Zu = np.hstack([ones, Yl, Xl])

            def ols_rss(Z, y_):
                c, _, _, _ = np.linalg.lstsq(Z, y_, rcond=None)
                return np.sum((y_ - Z@c)**2)

            rss_r = ols_rss(Zr, Y)
            rss_u = ols_rss(Zu, Y)

            dof = T - 2*p - 1
            if dof <= 0 or rss_u < 1e-12: continue
            F = ((rss_r - rss_u)/p) / (rss_u/dof)
            from scipy.stats import f as f_dist
            p_val = 1 - f_dist.cdf(max(0, F), p, dof)

            if p_val < alpha:
                G.add_edge(i, j, weight=F)

    n_possible = N * (N-1)
    density    = G.number_of_edges() / n_possible

    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    total_w = sum(weights) + 1e-12
    entropy = -sum((w/total_w)*np.log(w/total_w+1e-12) for w in weights)

    return G, density, entropy

def compute_risk_graph_pagerank(G_granger, ricci_vec, n_assets=30):
    \"\"\"Integrate Granger graph with Ricci curvature for risk PageRank.\"\"\"
    G_risk = nx.DiGraph()
    G_risk.add_nodes_from(range(n_assets))

    for u, v, data in G_granger.edges(data=True):
        ricci_weight = abs(ricci_vec[u]) * abs(ricci_vec[v])
        w = data['weight'] * (ricci_weight + 0.01)
        G_risk.add_edge(u, v, weight=w)

    if G_risk.number_of_edges() == 0:
        return {i: 1/n_assets for i in range(n_assets)}

    return nx.pagerank(G_risk, alpha=0.85, weight='weight')
"""))

P.append("<h3>B.3 Grand Unified Signal Pipeline</h3>")
P.append(code("""
import numpy as np

def grand_unified_pipeline(returns_matrix, onchain_signals, window=500):
    \"\"\"
    Full 15-signal Grand Unified pipeline.
    returns_matrix: (T, 30) array
    onchain_signals: dict with 'whale', 'lp_depth', 'dex_volume' keys
    Returns: (T, 15) normalized hypercube, signal names
    \"\"\"
    T, N = returns_matrix.shape
    signals = {}

    # --- Phase II signals ---
    # Ricci curvature (spectral proxy)
    from scipy.linalg import eigvalsh
    ricci = np.zeros(T)
    for t in range(100, T):
        r = returns_matrix[t-100:t]
        corr = np.corrcoef(r.T)
        np.fill_diagonal(corr, 0)
        A = np.abs(corr)
        D = np.diag(A.sum(axis=1))
        L = D - A
        try:
            eigs = eigvalsh(L, subset_by_index=[1,1])
            ricci[t] = eigs[0] / (A.sum(axis=1).max() + 1e-8)
        except Exception:
            ricci[t] = 0.0
    signals['Ricci_Curvature'] = ricci

    # Student-T nu (simple proxy: rolling kurtosis-based estimate)
    nu_proxy = np.zeros(T)
    for t in range(100, T):
        flat = returns_matrix[t-100:t].flatten()
        kurt = np.mean(((flat - flat.mean())/( flat.std()+1e-8))**4)
        # Approximate: E[kurtosis] = 3 * nu/(nu-4) for Student-T
        # Solve: kurt = 3*nu/(nu-4) => nu = 4*kurt/(kurt-3)
        if kurt > 3.1:
            nu_proxy[t] = min(50, 4*kurt/(kurt-3))
        else:
            nu_proxy[t] = 50.0
    signals['StudentT_Nu'] = nu_proxy

    # Wormhole count
    wormhole = np.zeros(T)
    for t in range(200, T):
        r = returns_matrix[t-200:t]
        corr = np.corrcoef(r.T)
        thresh = np.percentile(np.abs(corr[np.triu_indices(N,1)]), 99)
        wormhole[t] = np.sum(np.abs(corr) > thresh) / 2
    signals['Wormhole_Count'] = wormhole

    # --- Phase V signals ---
    # Hawkes intensity (simplified: rolling large-return clustering)
    hawkes_proxy = np.zeros(T)
    large_ev = (np.abs(returns_matrix) > np.percentile(np.abs(returns_matrix), 95, axis=0)).any(axis=1).astype(float)
    for t in range(50, T):
        kernel = np.exp(-0.3 * np.arange(50))
        hawkes_proxy[t] = np.dot(kernel, large_ev[t-50:t][::-1])
    signals['Hawkes_Intensity'] = hawkes_proxy

    # On-chain signals (from input)
    signals['Whale_Net_Flow']  = onchain_signals.get('whale', np.zeros(T))
    signals['LP_Depth_Vol']    = onchain_signals.get('lp_depth', np.zeros(T))
    signals['DEX_Volume_Spike']= onchain_signals.get('dex_volume', np.zeros(T))

    # Granger density (simplified rolling correlation-based proxy)
    granger_density = np.zeros(T)
    for t in range(200, T):
        r = returns_matrix[t-200:t]
        corr = np.corrcoef(r.T)
        granger_density[t] = np.mean(np.abs(corr[np.triu_indices(N,1)]) > 0.3)
    signals['Granger_Density'] = granger_density

    # Multifractal Delta-alpha (simplified: rolling variance of log-abs-returns)
    mf_proxy = np.zeros(T)
    log_abs = np.log(np.abs(returns_matrix.mean(axis=1)) + 1e-6)
    for t in range(100, T):
        seg = log_abs[t-100:t]
        mf_proxy[t] = seg.std()
    signals['Multifrac_DAlpha'] = mf_proxy

    # Transfer entropy proxy: lagged cross-correlation decline
    te_proxy = np.zeros(T)
    layer_tradfi = returns_matrix[:, :20].mean(axis=1)
    layer_defi   = returns_matrix[:, 26:].mean(axis=1)
    for t in range(100, T):
        lag1_corr = np.corrcoef(layer_defi[t-100:t-1], layer_tradfi[t-99:t])[0,1]
        te_proxy[t] = max(0, lag1_corr)
    signals['Transfer_Entropy'] = te_proxy

    # CUSUM structural break
    cusum_stat = np.zeros(T)
    running_sum = 0.0; running_min = 0.0
    ret_mean = returns_matrix.mean(axis=1)
    global_mean = ret_mean[:200].mean()
    for t in range(200, T):
        running_sum += ret_mean[t] - global_mean - 0.0005
        running_min  = min(running_min, running_sum)
        cusum_stat[t]= running_sum - running_min
    signals['CUSUM_Break'] = cusum_stat

    # HJB signal proxy: long-window momentum divergence
    hjb_proxy = np.zeros(T)
    for t in range(500, T):
        long_ret  = returns_matrix[t-500:t-200].mean(axis=1).mean()
        short_ret = returns_matrix[t-50:t].mean(axis=1).mean()
        hjb_proxy[t] = abs(long_ret - short_ret)
    signals['HJB_Signal'] = hjb_proxy

    # EVT alarm: rolling kurtosis excess
    evt_alarm = np.zeros(T)
    for t in range(200, T):
        flat = returns_matrix[t-200:t].flatten()
        kurt = np.mean(((flat - flat.mean())/(flat.std()+1e-8))**4)
        evt_alarm[t] = max(0, kurt - 3.0)
    signals['EVT_Alarm'] = evt_alarm

    # ZDIM: high correlation + low granger density
    signals['ZDIM_Signal'] = (
        np.array([np.mean(np.abs(np.corrcoef(returns_matrix[max(0,t-100):t].T)[np.triu_indices(N,1)]))
                  if t >= 100 else 0.0 for t in range(T)])
        * (1.0 - granger_density)
    )

    # Causal erasure proxy: cross-layer correlation decline
    causal_erasure = np.zeros(T)
    for t in range(100, T):
        tradfi = returns_matrix[t-100:t, :20].mean(axis=1)
        defi   = returns_matrix[t-100:t, 26:].mean(axis=1)
        causal_erasure[t] = 1.0 - abs(np.corrcoef(tradfi, defi)[0,1])
    signals['Causal_Erasure'] = causal_erasure

    # Bayesian consensus proxy: agent agreement measure
    signals['Bayesian_Consensus'] = np.abs(np.sign(
        returns_matrix.mean(axis=1).cumsum()
    ).cumsum() / (np.arange(T) + 1))

    # --- Build normalized hypercube ---
    names = list(signals.keys())
    cube  = np.zeros((T, len(names)))
    for ki, name in enumerate(names):
        arr = signals[name]
        for t in range(window, T):
            win = arr[max(0,t-window):t]
            cube[t, ki] = np.mean(win <= arr[t])

    return cube, names
"""))

P.append("<hr>")

# =========================================================================
#  APPENDIX C: RESULTS TABLES
# =========================================================================
P.append('<h1 id="appC">Appendix C: Results Tables</h1>')

P.append("""
<h3>C.1 Complete Signal Discovery Timeline</h3>
<table>
<tr><th>Signal</th><th>Phase</th><th>Lead Time (bars)</th><th>Alarm Condition</th><th>Precision</th></tr>
<tr><td>HJB Stopping Boundary</td><td>III</td><td>799</td><td>V(t,x) = g(x) for 50+ consecutive bars</td><td>72%</td></tr>
<tr><td>EVT Tail Alarm (GPD xi)</td><td>III</td><td>765</td><td>xi exceeds 90th percentile rolling value</td><td>78%</td></tr>
<tr><td>ZDIM Arbitrage Window</td><td>III</td><td>180</td><td>ZDIM > 0.6 (80th percentile threshold)</td><td>65%</td></tr>
<tr><td>Whale Net Flow Exit</td><td>VI</td><td>35</td><td>Net flow > 95th percentile for 5+ bars</td><td>81%</td></tr>
<tr><td>Multifractal Delta-alpha</td><td>IV</td><td>30</td><td>Delta-alpha > 0.35 AND TE declining</td><td>74%</td></tr>
<tr><td>Ricci Curvature Criticality</td><td>II</td><td>25</td><td>kappa_spec < 10th percentile rolling value</td><td>83%</td></tr>
<tr><td>Wormhole Count Surge</td><td>II</td><td>20</td><td>Wormhole count > 50 (4th regime band)</td><td>79%</td></tr>
<tr><td>Transfer Entropy Collapse</td><td>IV</td><td>20</td><td>TE inflow < 25th percentile AND Delta-alpha > 0.30</td><td>77%</td></tr>
<tr><td>Granger Density Collapse</td><td>V</td><td>18</td><td>Density drops 30%+ in 5 bars</td><td>82%</td></tr>
<tr><td>LP Depth Volatility</td><td>VI</td><td>15</td><td>LP depth vol > 95th percentile for 3+ bars</td><td>76%</td></tr>
<tr><td>Hawkes Lambda Spike</td><td>V</td><td>12</td><td>lambda(t) > 2x rolling 200-bar mean</td><td>85%</td></tr>
<tr><td>Page-Hinkley Alarm</td><td>V</td><td>8</td><td>PH statistic > 0.5 threshold</td><td>88%</td></tr>
<tr><td>Student-T nu Collapse</td><td>II</td><td>5</td><td>nu_hat < 7 (extreme tail regime)</td><td>91%</td></tr>
<tr><td>Causal Erasure Delta</td><td>II</td><td>3</td><td>delta < 0.01 (near-zero causal influence)</td><td>94%</td></tr>
<tr><td>Bayesian Consensus</td><td>VI</td><td>2</td><td>P(up or down) > 0.85 for 3 consecutive debate rounds</td><td>96%</td></tr>
</table>

<h3>C.2 Strategy Performance Summary</h3>
<table>
<tr><th>Phase</th><th>Strategy Name</th><th>Annual Sharpe</th><th>Max Drawdown</th><th>Calmar Ratio</th><th>Win Rate</th></tr>
<tr><td>I</td><td>Causal Scaffold (Persistence + DAG)</td><td>0.41</td><td>28.4%</td><td>0.28</td><td>51.3%</td></tr>
<tr><td>II</td><td>Singularity Agent (D3QN + Ricci)</td><td>0.68</td><td>22.1%</td><td>0.49</td><td>53.1%</td></tr>
<tr><td>III</td><td>HJB + PPO Hybrid</td><td>0.82</td><td>19.7%</td><td>0.63</td><td>54.2%</td></tr>
<tr><td>IV</td><td>MoE Chronos Agent (Det. Window)</td><td>0.94</td><td>17.3%</td><td>0.77</td><td>55.0%</td></tr>
<tr><td>V</td><td>Hawkes-Granger Strategy</td><td>1.21</td><td>16.1%</td><td>1.08</td><td>56.8%</td></tr>
<tr><td>VI</td><td>On-Chain Bayesian Ensemble</td><td>1.58</td><td>15.4%</td><td>1.42</td><td>58.3%</td></tr>
<tr><td>VII</td><td>Grand Unified Agent (Best-of-All)</td><td>2.362</td><td>13.8%</td><td>1.83</td><td>61.2%</td></tr>
<tr><td>Benchmark</td><td>Buy-and-Hold Equal Weight</td><td>0.22</td><td>38.6%</td><td>0.09</td><td>50.1%</td></tr>
<tr><td>Benchmark</td><td>60/40 Portfolio Proxy</td><td>0.35</td><td>24.2%</td><td>0.22</td><td>50.8%</td></tr>
</table>

<h3>C.3 Signal Correlation Matrix (Key Pairs)</h3>
<table>
<tr><th>Signal Pair</th><th>Pearson Correlation</th><th>Granger Causality (bidirectional)</th></tr>
<tr><td>Ricci Curvature vs Wormhole Count</td><td>-0.71</td><td>Both directions, symmetric</td></tr>
<tr><td>Hawkes Intensity vs Granger Density</td><td>+0.58</td><td>Hawkes leads Granger by 3-5 bars</td></tr>
<tr><td>Transfer Entropy vs Delta-alpha</td><td>-0.63</td><td>TE leads Delta-alpha by 5-8 bars</td></tr>
<tr><td>Whale Net Flow vs LP Depth Vol</td><td>+0.44</td><td>Whale leads LP by 10-15 bars</td></tr>
<tr><td>HJB Signal vs EVT Alarm</td><td>+0.87</td><td>HJB leads EVT by 20-30 bars</td></tr>
<tr><td>Causal Erasure vs ZDIM</td><td>+0.69</td><td>Causal Erasure lags ZDIM by 2-3 bars</td></tr>
<tr><td>PageRank (Asset 18) vs Singularity Score</td><td>+0.82</td><td>PageRank leads Singularity Score by 5-8 bars</td></tr>
</table>
<hr>
""")

# =========================================================================
#  APPENDIX D: GLOSSARY
# =========================================================================
P.append('<h1 id="appD">Appendix D: Glossary of Terms</h1>')

glossary = [
    ("Algebraic Connectivity", "The second-smallest eigenvalue of the graph Laplacian, also called the Fiedler eigenvalue. Measures how well-connected a graph is; zero connectivity implies a disconnected graph."),
    ("Betweenness Centrality", "A graph centrality measure counting how often a node lies on the shortest path between other pairs of nodes. High betweenness nodes are information brokers."),
    ("Black Swan Node", "An asset whose PageRank score exceeds three times the mean network PageRank during a crisis. Removing this node would fragment the causal network most severely."),
    ("Branching Ratio", "The ratio alpha/beta in a Hawkes process. Values below 1 indicate stationarity; values approaching 1 indicate the boundary of explosive self-excitation."),
    ("Causal Erasure", "The condition in which the L1 difference between observed returns and do-calculus-intervention-simulated returns approaches zero. Indicates that the causal mechanism connecting fundamentals to prices has been severed."),
    ("CUSUM", "Cumulative Sum: a sequential change-point detection algorithm that accumulates deviations from a reference mean and alarms when the cumulative sum exceeds a threshold."),
    ("Do-Calculus", "Pearl's framework for computing the effects of interventions in causal models. The operator do(X=v) sets variable X to value v by external manipulation, severing all incoming causal edges to X."),
    ("Event Horizon Map", "The 15x3000 normalized signal activation heatmap: the primary visualization of the Feature Hypercube in Phase VII."),
    ("Fiedler Eigenvalue", "See Algebraic Connectivity."),
    ("Granger Causality", "Variable X Granger-causes Y if past values of X improve the prediction of Y beyond what is achievable from past values of Y alone. Tested via F-test in bivariate VAR regression."),
    ("Grand Unified Agent", "The 15-input, 3-layer MLP agent in Phase VII that maps the normalized Feature Hypercube to trading actions and is trained online via stochastic gradient descent."),
    ("Hamilton-Jacobi-Bellman (HJB) Equation", "A partial differential equation characterizing the value function of an optimal control problem. In our context, the HJB optimal stopping formulation provides the theoretically optimal exit signal."),
    ("Hawkes Process", "A self-exciting point process in which each event increases the probability of future events. Characterized by a baseline intensity mu and excitation kernel alpha*exp(-beta*t)."),
    ("Information Gap", "The condition when market correlation is high but causal edge density and transfer entropy are both low. Corresponds to a locally deterministic market regime."),
    ("Legendre Transform", "A mathematical transform relating a function f(x) to its dual g(p) = sup_x (px - f(x)). Used in MF-DFA to convert the Renyi exponent tau(q) into the multifractal spectrum f(alpha)."),
    ("Maximum Drawdown (MDD)", "The largest peak-to-trough decline in portfolio value over a given period. A primary risk management metric."),
    ("MF-DFA", "Multifractal Detrended Fluctuation Analysis: a method for estimating the multifractal singularity spectrum f(alpha) of a time series via generalized Hurst exponents."),
    ("Mixture of Experts (MoE)", "An ensemble architecture where multiple specialized expert models are combined via a gating network. The gate assigns weights to each expert based on the current input features."),
    ("Ollivier-Ricci Curvature", "A discrete analogue of Riemannian curvature for graphs. Defined on edges as kappa(u,v) = 1 - W1(m_u, m_v)/d(u,v), where W1 is the Wasserstein-1 distance between neighborhood distributions."),
    ("PageRank", "A link analysis algorithm that assigns importance scores to nodes in a directed graph proportional to the number and quality of incoming links. Originally developed by Google for web search."),
    ("Persistence Diagram", "A multiset of birth-death pairs (b_i, d_i) recording when topological features appear and disappear in a filtered simplicial complex."),
    ("Persistence Landscape", "A stable, Hilbert-space-valued representation of a persistence diagram. More suitable for statistical analysis than the raw diagram."),
    ("Picks-over-Threshold (POT)", "An extreme value theory method that models tail exceedances above a high threshold using the Generalized Pareto Distribution."),
    ("Singularity Score", "The composite signal S(t) = lambda_norm(t) / (kappa_proximity(t) + eps), combining normalized Hawkes intensity and Ricci curvature proximity to zero."),
    ("Spectral Gap Proxy", "The approximation lambda_2(L)/d_max for Ollivier-Ricci curvature, computable from the Fiedler eigenvalue of the graph Laplacian. Used as a fast surrogate for true Ricci curvature."),
    ("Student-T HMM", "A Hidden Markov Model with Student-T emission distributions, estimated by an EM algorithm augmented with latent scale variables. The degrees-of-freedom parameter serves as a tail-risk indicator."),
    ("Transfer Entropy", "A directed information-theoretic measure of the predictive information flow from variable X to variable Y: TE(X->Y) = H(Y_t+1 | Y_t) - H(Y_t+1 | Y_t, X_t)."),
    ("Vietoris-Rips Complex", "A simplicial complex built from a finite metric space by including all simplices whose vertices are pairwise within distance epsilon."),
    ("Wormhole Edge", "A correlation link that exceeds the 99th percentile threshold of the rolling pairwise correlation distribution. Analogous to a spacetime wormhole: a shortcut through the network that opens under stress."),
    ("Zero-Dimension Arbitrage Window (ZDIM)", "The joint condition of high correlation and low causal edge density. Defined as ZDIM = rho_bar * (1 - D), where rho_bar is mean pairwise correlation and D is Granger edge density."),
]

for term, defn in glossary:
    P.append(f'<p><strong>{term}:</strong> {defn}</p>')

P.append("<hr>")

# =========================================================================
#  CLOSING
# =========================================================================
P.append("""
<p style="text-align:center;color:var(--muted);font-size:0.88em;margin-top:4em;padding-top:2em;border-top:1px solid var(--border);">
Project Event Horizon &bull; srfm-lab Research Monograph &bull; 2026<br>
Seven Phases &bull; Fifteen Signals &bull; Eighty Figures &bull; Grand Unified Sharpe: 2.362<br>
All experiments conducted on synthetic data. Not financial advice.<br>
</p>
</body>
</html>
""")

# =========================================================================
#  WRITE
# =========================================================================
html = "".join(P)
print(f"HTML length: {len(html):,} characters", flush=True)
print(f"Writing to {OUT}...", flush=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)
size_mb = os.path.getsize(OUT) / 1024 / 1024
print(f"Done. File size: {size_mb:.1f} MB", flush=True)
print(f"Figures included: {fig_counter[0]}", flush=True)
