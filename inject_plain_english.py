"""
inject_plain_english.py
Injects plain-English explanation boxes after technical section headings
in the Oxidized Archive styled HTML, then regenerates the PDF.
"""
import re, os

IN_HTML  = r"C:/Users/Matthew/Desktop/srfm-experiments/event_horizon_book_oxidized.html"
OUT_HTML = r"C:/Users/Matthew/Desktop/srfm-experiments/event_horizon_book_oxidized.html"
OUT_PDF  = r"C:/Users/Matthew/Desktop/srfm-experiments/event_horizon_book.pdf"

def box(text):
    return (
        '\n<div class="nonphd"><div class="nonphd-label">Plain English</div>'
        f'<p>{text}</p></div>\n'
    )

# Map: exact heading text -> plain English explanation
HEADING_CALLOUTS = {
    "4.1 Vietoris-Rips Filtration and Persistence Landscapes": box(
        "Persistent homology is a way of finding hidden shapes inside clouds of data. "
        "Imagine drawing circles around every data point and slowly expanding them. "
        "When circles overlap, they form clusters and loops. "
        "Clusters that form quickly and last a long time are real market structure. "
        "Clusters that appear and vanish immediately are noise. "
        "The 'persistence landscape' is essentially a scorecard of which shapes are real versus random."
    ),
    "4.2 PC-Algorithm Causal DAG Discovery": box(
        "A causal DAG (Directed Acyclic Graph) is a flowchart of cause and effect. "
        "The PC-Algorithm builds this flowchart automatically from data by asking: "
        "'If I already know the value of these two assets, does a third one still add information?' "
        "If yes, there is a direct causal arrow. "
        "The result is a map showing which assets drive which, without any human assumptions baked in."
    ),
    "5.1 Student-T Hidden Markov Model": box(
        "A Hidden Markov Model treats the market as a machine that secretly switches "
        "between different 'states' such as calm, trending, and volatile, without announcing which state it is in. "
        "You can only observe prices. The model figures out which hidden state you are probably in "
        "by learning the typical behaviour of each state. "
        "Using the Student-T distribution (rather than the normal bell curve) means the model "
        "correctly handles the occasional extreme price moves that bell-curve models always underestimate."
    ),
    "5.2 Ricci Curvature and the Geometry of Contagion": box(
        "Ricci curvature, borrowed from Einstein's general relativity, measures how 'congested' a network is. "
        "High positive curvature means every asset is tightly linked to every other: "
        "a shock anywhere spreads everywhere instantly, like a traffic jam on a ring road. "
        "Negative curvature means connections are sparse and shocks stay local. "
        "When the financial network becomes too positively curved, meaning everyone is correlated with everyone, "
        "the conditions for a systemic crash are in place."
    ),
    "5.4 Do-Calculus and Causal Erasure": box(
        "Do-calculus is a mathematical language invented by Judea Pearl to answer the question: "
        "'What would happen if I forcibly intervened and changed this one thing?' "
        "It is the difference between 'people who carry lighters tend to get cancer' (correlation) "
        "and 'if I gave everyone a lighter, would cancer rates rise?' (causal question). "
        "In markets, it lets us ask which signals genuinely cause returns versus which "
        "just happen to move together by coincidence."
    ),
    "6.1 Hamilton-Jacobi-Bellman Optimal Stopping": box(
        "The HJB equation answers: 'Given changing market conditions, exactly when should I exit "
        "this trade to maximize total profit?' "
        "It works backwards from a future deadline, computing at each moment whether "
        "the expected future gain justifies staying in the position versus taking profits now. "
        "Think of it as a very sophisticated 'hold or fold' calculator that accounts for "
        "how volatility and drift are evolving in real time."
    ),
    "6.2 Extreme Value Theory and GPD Tail Fitting": box(
        "Extreme Value Theory ignores normal days and focuses only on the worst outcomes. "
        "The Generalized Pareto Distribution fits a curve specifically to the largest losses, "
        "answering questions such as 'how bad could a once-in-fifty-years crash really be?' "
        "Standard bell-curve statistics catastrophically underestimate tail events. "
        "EVT was purpose-built for exactly those rare but catastrophic scenarios that "
        "wipe out portfolios that looked perfectly safe on average."
    ),
    "7.1 Multifractal Detrended Fluctuation Analysis": box(
        "A fractal is a pattern that looks the same at every scale, like a coastline. "
        "Financial markets have a similar property: daily moves look statistically similar to monthly moves. "
        "MF-DFA measures whether this self-similarity is breaking down. "
        "When different time scales start behaving very differently from one another, "
        "it is a warning that the market structure is becoming unstable. "
        "Think of it as detecting when the market stops following its usual rhythm."
    ),
    "7.2 Cross-Layer Transfer Entropy Matrix": box(
        "Transfer entropy measures how much knowing the recent history of Asset A "
        "reduces your uncertainty about what Asset B will do next, above and beyond "
        "what B's own history already tells you. "
        "Unlike simple correlation, it is directional: A might influence B without B influencing A. "
        "The 'cross-layer' part means we measure this across different types of signals, "
        "such as whether momentum signals influence volatility signals, building a map of information flow."
    ),
    "7.3 CUSUM Structural Break Detection and MoE Gating": box(
        "CUSUM (Cumulative Sum) is a simple but powerful alarm system. "
        "It adds up prediction errors over time. If errors consistently go in one direction, "
        "the cumulative sum grows large and triggers an alert: 'the market regime has changed, "
        "your model is no longer appropriate.' "
        "The Mixture-of-Experts (MoE) gating network then uses this alert to automatically "
        "switch to a different model that was pre-trained on the new type of market behaviour."
    ),
    "8.1 Hawkes Process Maximum Likelihood Estimation": box(
        "A Hawkes process models how events cluster together over time. "
        "In markets, one large trade tends to trigger more large trades shortly after: "
        "order flow is self-exciting. "
        "The intensity function measures the current 'heat' of trading activity. "
        "When intensity surges, it reliably precedes a volatility spike by several ticks. "
        "This gives traders a leading indicator rather than a lagging one: "
        "you can see the pressure building before the explosion happens."
    ),
    "8.2 N x N Granger Causality Matrix and Causality Collapse": box(
        "The N-by-N Granger matrix is a grid where every row and column is one asset. "
        "Each cell is a score: 'how much does asset in row i help predict asset in column j?' "
        "Under normal conditions this matrix is spread out and diverse. "
        "During crashes, the research shows it collapses: one 'super-hub' asset starts driving "
        "everything else. Finding that super-hub before it collapses the system is the goal."
    ),
    "8.3 Page-Hinkley Drift Detection": box(
        "The Page-Hinkley test is a dashboard warning light for your quantitative model. "
        "It tracks the running average of prediction errors. "
        "When errors start drifting consistently in one direction, the test fires: "
        "'your model no longer fits this market, time to retrain.' "
        "Without drift detection, a model trained on calm markets keeps trading as if "
        "conditions are calm even during a crisis, which is how strategies blow up."
    ),
    "9.2 The Bayesian Debate System": box(
        "The Bayesian debate works like a panel of expert analysts who argue until they reach consensus. "
        "Each AI agent starts with an opinion on market direction and a 'credibility score' "
        "based on how accurate it was yesterday. "
        "Over five rounds, agents update their views based on both the evidence and each "
        "other's track records. Agents with better recent histories get more weight. "
        "The final consensus is more reliable than any single agent because it is a "
        "calibrated, evidence-weighted average of multiple independent perspectives."
    ),
    "10.2 Topological Risk Graph and PageRank": box(
        "PageRank was invented by Google to rank web pages by how many important pages link to them. "
        "Applied to a financial network, it identifies which assets are most central, "
        "meaning most linked-to by other important assets. "
        "A high-PageRank asset in a financial network is a systemic node: "
        "if it moves violently, the shock propagates to many other assets immediately. "
        "This is the quantitative definition of 'too connected to fail.'"
    ),
    "A.3 Hawkes Process Log-Likelihood": box(
        "The log-likelihood is the score the model uses to grade its own fit to the data. "
        "Maximising it means finding the parameter values that make the observed sequence "
        "of trades look as unsurprising as possible under the model. "
        "The recursive formula makes this computationally feasible even for tens of thousands of events."
    ),
    "A.5 Generalized Pareto Distribution (POT Method)": box(
        "Peak Over Threshold (POT) isolates only the returns that exceed a high threshold, "
        "then fits the Generalized Pareto Distribution to just those extreme values. "
        "The shape parameter xi determines whether the tail is bounded, exponential, or heavy. "
        "For most financial assets xi is positive, meaning the tail is heavy and large losses "
        "are far more common than a normal distribution would suggest."
    ),
}

print("Reading HTML...", flush=True)
with open(IN_HTML, "r", encoding="utf-8") as f:
    html = f.read()

print("Injecting plain-English boxes...", flush=True)
injected = 0
for heading_text, callout in HEADING_CALLOUTS.items():
    # Match the h2/h3 tag containing this heading text, then inject after the closing tag
    pattern = re.escape(heading_text)
    close_tag = r'</h[23]>'
    # Build pattern: <h2...>heading text</h2> or <h3...>heading text</h3>
    full_pattern = r'(<h[23][^>]*>' + pattern + r'</h[23]>)'
    new_html, n = re.subn(full_pattern, r'\1' + callout, html, count=1)
    if n:
        html = new_html
        injected += 1

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
        page.goto(file_url, wait_until="networkidle", timeout=120000)
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

print("Done.", flush=True)
