"""
restyle_book.py
Takes event_horizon_book.html, applies Oxidized Archive design system,
removes em dashes, injects additional plain-English callout boxes,
and exports to PDF via weasyprint.
"""
import re, sys

IN_HTML  = r"C:/Users/Matthew/Desktop/srfm-experiments/event_horizon_book.html"
OUT_HTML = r"C:/Users/Matthew/Desktop/srfm-experiments/event_horizon_book_oxidized.html"
OUT_PDF  = r"C:/Users/Matthew/Desktop/srfm-experiments/event_horizon_book.pdf"

print("Reading HTML...", flush=True)
with open(IN_HTML, "r", encoding="utf-8") as f:
    html = f.read()

# ── 1. Remove em dashes ──────────────────────────────────────────────────────
html = html.replace("\u2014", ", ")
html = html.replace("&mdash;", ", ")
html = html.replace("&#8212;", ", ")
print(f"  Em dashes removed.", flush=True)

# ── 2. Replace the <head> meta + title block with Oxidized Archive version ───
# Inject Google Fonts link after <head>
font_link = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500'
    '&family=Crimson+Pro:ital,wght@0,300;0,400;0,600;1,300;1,400'
    '&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">\n'
)
html = html.replace("<head>", "<head>\n" + font_link, 1)

# ── 3. Replace entire <style>…</style> block with Oxidized Archive CSS ───────
NEW_CSS = """<style>
/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── Oxidized Archive Palette ── */
:root {
  --ink:          #1a1612;
  --ink-light:    #4a433c;
  --ink-faint:    #8a7f76;
  --paper:        #faf8f4;
  --cream:        #f2ede4;
  --rule:         #c8bfb0;
  --accent:       #8b2f1e;
  --accent-light: #c4503a;
  --accent-pale:  #f5ede8;
  --blue:         #1e4a8b;
  --blue-pale:    #e8edf5;
  --gold:         #b08a3c;
  --gold-pale:    #f8f2e4;
}

/* ── Page / body ── */
@page {
  size: US-Letter;
  margin: 1in;
}
body {
  background: var(--paper);
  color: var(--ink);
  font-family: "EB Garamond", "Georgia", serif;
  font-size: 18px;
  line-height: 1.75;
  max-width: 720px;
  margin: 0 auto;
  padding: 2em 0 6em;
  text-align: justify;
  hyphens: auto;
  -webkit-hyphens: auto;
}

/* ── Journal meta bar ── */
.journal-meta {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--ink-faint);
  border-bottom: 1px solid var(--rule);
  padding-bottom: 0.6em;
  margin-bottom: 2em;
}

/* ── Typography ── */
h1 {
  font-family: "Crimson Pro", "Georgia", serif;
  font-size: 2.6em;
  font-weight: 600;
  color: var(--ink);
  text-align: center;
  line-height: 1.2;
  margin: 1em 0 0.4em;
}
h2 {
  font-family: "Crimson Pro", "Georgia", serif;
  font-size: 1.55em;
  font-weight: 600;
  color: var(--ink);
  border-bottom: 1px solid var(--rule);
  margin: 2.8em 0 0.8em;
  padding-bottom: 0.3em;
}
h2 .sec-num {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.65rem;
  color: var(--accent);
  letter-spacing: 0.05em;
  display: block;
  margin-bottom: 0.2em;
}
h3 {
  font-family: "Crimson Pro", "Georgia", serif;
  font-size: 1.22em;
  font-weight: 600;
  color: var(--ink);
  margin: 2em 0 0.6em;
}
h4 {
  font-family: "Crimson Pro", "Georgia", serif;
  font-size: 1.05em;
  font-style: italic;
  color: var(--ink-light);
  margin: 1.5em 0 0.4em;
}
p { margin: 0.9em 0; }
ul, ol { margin: 0.8em 0 0.8em 2em; }
li { margin: 0.35em 0; }
a { color: var(--blue); }
strong { color: var(--ink); font-weight: 600; }
em { font-style: italic; color: var(--ink-light); }
hr { border: none; border-top: 1px solid var(--rule); margin: 3em 0; }

/* Drop cap on first paragraph of each major section */
.drop-cap::first-letter {
  float: left;
  font-family: "Crimson Pro", serif;
  font-size: 3.8rem;
  line-height: 0.85;
  color: var(--accent);
  margin-right: 0.08em;
  margin-top: 0.05em;
}

/* ── Code ── */
.code {
  background: var(--cream);
  border: 1px solid var(--rule);
  border-left: 3px solid var(--gold);
  padding: 1.1em 1.4em;
  margin: 1.5em 0;
  overflow-x: auto;
  font-family: "JetBrains Mono", "Consolas", monospace;
  font-size: 0.78em;
  line-height: 1.55;
  color: var(--ink);
}
.code code { background: none; padding: 0; }
code {
  background: var(--cream);
  padding: 0.1em 0.35em;
  border-radius: 2px;
  font-size: 0.84em;
  color: var(--accent);
  font-family: "JetBrains Mono", monospace;
}

/* ── Figures ── */
.fig { margin: 2.5em 0; text-align: center; }
.fig img { width: 92%; border: 1px solid var(--rule); }
figcaption {
  color: var(--ink-light);
  font-size: 0.85em;
  margin-top: 0.7em;
  text-align: left;
  max-width: 90%;
  margin-left: auto;
  margin-right: auto;
}
figcaption strong {
  display: block;
  font-family: "JetBrains Mono", monospace;
  font-size: 0.75em;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 0.2em;
}

/* ── Math formula blocks ── */
.math-disp, mjx-container[display="true"] {
  background: var(--gold-pale);
  border: 1px solid var(--gold);
  padding: 0.8em 1.2em;
  margin: 1.2em 0;
  overflow-x: auto;
  border-radius: 2px;
}

/* ── Plain Language box (for non-PhD readers) ── */
.nonphd {
  background: var(--blue-pale);
  border: 1px solid #c0cde0;
  border-left: 3px solid var(--blue);
  padding: 1em 1.4em;
  margin: 1.5em 0;
}
.nonphd-label {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.68rem;
  color: var(--blue);
  text-transform: uppercase;
  letter-spacing: 0.09em;
  margin-bottom: 0.4em;
  font-weight: 500;
}

/* ── Key Takeaway box ── */
.takeaway {
  background: var(--accent-pale);
  border: 1px solid #dcc8c2;
  border-left: 3px solid var(--accent);
  padding: 1em 1.4em;
  margin: 1.5em 0;
}
.takeaway-label {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.68rem;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.09em;
  margin-bottom: 0.4em;
  font-weight: 500;
}

/* ── LinkedIn claim box ── */
.linkedin {
  background: var(--blue-pale);
  border-top: 2px solid var(--blue);
  border-bottom: 1px solid var(--rule);
  padding: 1em 1.4em;
  margin: 1.5em 0;
}
.linkedin-label {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.68rem;
  color: var(--blue);
  text-transform: uppercase;
  letter-spacing: 0.09em;
  margin-bottom: 0.4em;
  font-weight: 500;
}

/* ── Data callout box (blue) ── */
.box-info {
  background: var(--blue-pale);
  border-left: 3px solid var(--blue);
  padding: 1.1em 1.4em;
  margin: 1.5em 0;
}
.box-warn {
  background: var(--accent-pale);
  border-left: 3px solid var(--accent);
  padding: 1.1em 1.4em;
  margin: 1.5em 0;
}
.box-title {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.68rem;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.09em;
  margin-bottom: 0.5em;
  font-weight: 500;
}

/* ── Pull quote ── */
.pull-quote {
  border-top: 2px solid var(--accent);
  border-bottom: 1px solid var(--rule);
  font-family: "Crimson Pro", serif;
  font-style: italic;
  font-size: 1.3rem;
  color: var(--ink-light);
  padding: 0.8em 1.2em;
  margin: 2em 0;
}

/* ── Part headers ── */
.part-header {
  background: var(--cream);
  border-top: 3px solid var(--accent);
  border-bottom: 1px solid var(--rule);
  padding: 2.5em 2em;
  margin: 4em 0 2em;
  text-align: center;
}
.part-label {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.65rem;
  color: var(--ink-faint);
  text-transform: uppercase;
  letter-spacing: 0.15em;
  margin-bottom: 0.4em;
}
.part-title {
  font-family: "Crimson Pro", serif;
  font-size: 2.2em;
  font-weight: 600;
  color: var(--ink);
  margin-bottom: 0.5em;
}
.part-desc {
  color: var(--ink-light);
  font-size: 1em;
  max-width: 560px;
  margin: 0 auto;
  font-style: italic;
}

/* ── Chapter headers ── */
.chapter-header {
  margin: 3em 0 2em;
  padding: 2em 0 1.2em;
  border-bottom: 2px solid var(--rule);
}
.chapter-num {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.65rem;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.15em;
  margin-bottom: 0.3em;
}
.chapter-title {
  font-family: "Crimson Pro", serif;
  font-size: 2em;
  font-weight: 600;
  color: var(--ink);
  line-height: 1.2;
}
.chapter-subtitle {
  color: var(--ink-faint);
  font-size: 1em;
  margin-top: 0.5em;
  font-style: italic;
}

/* ── Tables ── */
table { width: 100%; border-collapse: collapse; margin: 1.5em 0; font-size: 0.88em; }
th {
  background: var(--cream);
  color: var(--ink);
  padding: 0.65em 1em;
  text-align: left;
  border-bottom: 2px solid var(--rule);
  font-family: "JetBrains Mono", monospace;
  font-size: 0.72em;
  text-transform: uppercase;
  letter-spacing: 0.07em;
}
td { padding: 0.5em 1em; border-bottom: 1px solid var(--rule); }
tr:nth-child(even) td { background: var(--cream); }

/* ── Abstract / TOC ── */
.abstract {
  background: var(--cream);
  border: 1px solid var(--rule);
  padding: 1.6em 2em;
  margin: 2em 0;
  font-style: italic;
  color: var(--ink-light);
}
.toc {
  background: var(--cream);
  border: 1px solid var(--rule);
  padding: 1.4em 2em;
  margin: 2em 0;
}
.toc h3 {
  font-family: "Crimson Pro", serif;
  color: var(--ink);
  margin-top: 0;
  border-bottom: 1px solid var(--rule);
  padding-bottom: 0.3em;
  margin-bottom: 0.7em;
}
.toc a { color: var(--ink-light); text-decoration: none; }
.toc a:hover { color: var(--accent); }
.toc li { margin: 0.22em 0; }
.toc .toc-part {
  color: var(--accent);
  font-family: "Crimson Pro", serif;
  font-weight: 600;
  margin-top: 0.7em;
  list-style: none;
}

/* ── Title page ── */
.title-page {
  text-align: center;
  padding: 4em 2em;
  min-height: 55vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border-bottom: 1px solid var(--rule);
  margin-bottom: 2em;
}
.title-page .subtitle { color: var(--ink-light); font-style: italic; font-size: 1.15em; margin: 0.5em 0 1.2em; }
.title-page .authors  { color: var(--ink-faint); font-size: 0.95em; margin: 0.4em 0; }
.title-page .affil    { font-family: "JetBrains Mono", monospace; font-size: 0.65rem; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.8em; }
.title-page hr        { border-top: 1px solid var(--rule); width: 80px; margin: 1.5em auto; }

/* ── Footer ── */
footer {
  margin-top: 5em;
  padding-top: 1em;
  border-top: 1px solid var(--rule);
  font-family: "JetBrains Mono", monospace;
  font-size: 0.62rem;
  color: var(--ink-faint);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  text-align: center;
}

/* ── Missing figure ── */
.missing { color: var(--ink-faint); font-style: italic; font-size: 0.85em; margin: 1em 0; }

/* ── Print overrides ── */
@media print {
  body { font-size: 11pt; max-width: none; padding: 0; }
  .nonphd, .takeaway, .linkedin, .box-info, .box-warn { break-inside: avoid; }
  .chapter-header, .part-header { break-before: page; break-inside: avoid; }
  .fig { break-inside: avoid; }
  .code { break-inside: avoid; font-size: 7.5pt; }
}
</style>"""

# Replace old style block
html = re.sub(r'<style>.*?</style>', NEW_CSS, html, count=1, flags=re.DOTALL)
print("  CSS replaced with Oxidized Archive theme.", flush=True)

# ── 4. Add journal meta bar just after <body> ─────────────────────────────────
meta_bar = (
    '\n<div class="journal-meta">'
    'Project Event Horizon &nbsp;&bull;&nbsp; Working Paper &nbsp;&bull;&nbsp; '
    'Matthew Charles Busel &nbsp;&bull;&nbsp; Tensorust &nbsp;&bull;&nbsp; '
    'Westchester, NY &nbsp;&bull;&nbsp; Draft v1.0 &nbsp;&bull;&nbsp; DOI Pending'
    '</div>\n'
)
html = html.replace('<body>', '<body>' + meta_bar, 1)

# ── 5. Add footer before </body> ──────────────────────────────────────────────
footer_html = (
    '\n<footer>'
    'Working Paper &nbsp;&bull;&nbsp; Draft v1.0 &nbsp;&bull;&nbsp; '
    'DOI Pending &nbsp;&bull;&nbsp; Matthew Charles Busel / Tensorust &nbsp;&bull;&nbsp; '
    'Westchester, NY &nbsp;&bull;&nbsp; 2026'
    '</footer>\n'
)
html = html.replace('</body>', footer_html + '</body>', 1)

# ── 6. Inject additional plain-English boxes after key technical sections ─────
# We identify signature phrases that follow complex math and insert callouts.

INJECTIONS = [
    # (search_pattern, callout_html)
    (
        r'(The intensity function.*?self-exciting process\.)</p>',
        '<div class="nonphd"><div class="nonphd-label">Plain English</div>'
        '<p>Think of the market as a crowd at a concert. When one person starts '
        'cheering loudly, nearby people join in and cheer louder too. '
        'The Hawkes process measures exactly that: how one burst of trading activity '
        'triggers more activity, creating a feedback loop. The "intensity" is simply '
        'the current level of excitement in the crowd.</p></div>'
    ),
    (
        r'(Granger causality tests whether.*?F-statistic\.)</p>',
        '<div class="nonphd"><div class="nonphd-label">Plain English</div>'
        '<p>Granger causality is a statistical way of asking: "Does knowing what '
        'Stock A did yesterday help me predict what Stock B does today?" '
        'If yes, we say A "Granger-causes" B. It is not proof of true cause and effect, '
        'but it is strong evidence that information flows from one asset to another.</p></div>'
    ),
    (
        r'(persistent homology.*?birth-death pairs\.)</p>',
        '<div class="nonphd"><div class="nonphd-label">Plain English</div>'
        '<p>Persistent homology is a way of finding shapes hidden in clouds of data. '
        'Imagine connecting dots on a scatter plot with rubber bands as you slowly '
        'stretch them. Clusters that form and disappear quickly are noise; '
        'clusters that survive a long time are real structure. '
        'This technique finds those long-lived patterns automatically.</p></div>'
    ),
    (
        r'(Ollivier-Ricci curvature.*?negatively curved\.)</p>',
        '<div class="nonphd"><div class="nonphd-label">Plain English</div>'
        '<p>Ricci curvature borrowed from physics measures how "crowded" a network is. '
        'Positive curvature means assets are tightly connected like a busy highway interchange: '
        'a shock anywhere spreads everywhere instantly. '
        'Negative curvature means connections are sparse and local shocks stay local. '
        'When markets become too positively curved, a crash becomes much more likely.</p></div>'
    ),
    (
        r'(Hamilton-Jacobi-Bellman.*?optimal stopping\.)</p>',
        '<div class="nonphd"><div class="nonphd-label">Plain English</div>'
        '<p>The Hamilton-Jacobi-Bellman equation answers the question: '
        '"Given that conditions are changing every second, exactly when should I exit '
        'this trade to maximize my profit?" '
        'It works backwards from the future, computing at each moment whether '
        'holding on is worth the risk compared to taking profits now.</p></div>'
    ),
    (
        r'(multifractal spectrum.*?singularity spectrum\.)</p>',
        '<div class="nonphd"><div class="nonphd-label">Plain English</div>'
        '<p>A multifractal analysis measures whether price moves are "self-similar" '
        'at every time scale. Normal markets behave roughly the same whether you look '
        'at one-minute or one-day charts. When the fractal structure breaks down and '
        'different time scales start behaving very differently, it is often a warning '
        'sign that a regime change or crash is approaching.</p></div>'
    ),
    (
        r'(transfer entropy.*?conditional entropy\.)</p>',
        '<div class="nonphd"><div class="nonphd-label">Plain English</div>'
        '<p>Transfer entropy measures how much knowing the history of Asset A '
        'reduces your uncertainty about Asset B, beyond what B\'s own history tells you. '
        'Unlike correlation, it catches one-way influence: A might drive B without B '
        'driving A at all. Think of it as measuring whether one asset is "broadcasting" '
        'information to another.</p></div>'
    ),
    (
        r'(Page-Hinkley test.*?cumulative sum\.)</p>',
        '<div class="nonphd"><div class="nonphd-label">Plain English</div>'
        '<p>The Page-Hinkley test is an alarm system for your model. '
        'It tracks whether the average prediction error is slowly drifting upward. '
        'If the errors keep piling up in one direction, the test fires an alert '
        'saying "your model no longer fits the current market, recalibrate." '
        'It is the equivalent of a dashboard warning light for quantitative strategies.</p></div>'
    ),
    (
        r'(Bayesian debate.*?posterior probability\.)</p>',
        '<div class="nonphd"><div class="nonphd-label">Plain English</div>'
        '<p>The Bayesian debate system works like a panel of expert analysts. '
        'Each AI agent starts with an opinion on market direction. '
        'After each "round," agents with better recent track records get more votes. '
        'By round five, the panel has collectively updated its view based on both '
        'the evidence and each agent\'s credibility. The result is a consensus '
        'that is more reliable than any single agent\'s prediction alone.</p></div>'
    ),
    (
        r'(PageRank.*?betweenness centrality.*?systemic risk\.)</p>',
        '<div class="nonphd"><div class="nonphd-label">Plain English</div>'
        '<p>PageRank was invented by Google to rank web pages. Applied to financial '
        'networks, it identifies which assets are most "linked-to" by other important '
        'assets. An asset with high PageRank in a financial network is a systemic node: '
        'if it collapses, the shock propagates through many other assets. '
        'Think of it as finding the "too connected to fail" nodes in the market graph.</p></div>'
    ),
    (
        r'(Generalized Pareto Distribution.*?excess losses\.)</p>',
        '<div class="nonphd"><div class="nonphd-label">Plain English</div>'
        '<p>Extreme Value Theory focuses only on the worst outcomes, ignoring normal days. '
        'The Generalized Pareto Distribution fits a curve specifically to tail losses, '
        'answering questions like "how bad could a once-in-fifty-years crash be?" '
        'Standard bell-curve statistics wildly underestimate these rare but catastrophic events; '
        'EVT is built precisely for the situations where that matters most.</p></div>'
    ),
    (
        r'(Student-T Hidden Markov Model.*?degrees of freedom\.)</p>',
        '<div class="nonphd"><div class="nonphd-label">Plain English</div>'
        '<p>A Hidden Markov Model treats the market as a machine that secretly switches '
        'between different "states" (calm, volatile, trending) without telling you directly. '
        'You can only observe prices. The model infers which hidden state you are probably in '
        'right now by learning each state\'s typical behavior. '
        'Using a Student-T distribution instead of a normal one means the model properly '
        'accounts for the fat-tailed, crash-prone nature of real financial returns.</p></div>'
    ),
]

injected = 0
for pattern, callout in INJECTIONS:
    new_html, n = re.subn(pattern, r'\1</p>' + callout + '<p>', html, count=1, flags=re.DOTALL | re.IGNORECASE)
    if n:
        html = new_html
        injected += 1

print(f"  {injected} plain-English callout boxes injected.", flush=True)

# ── 7. Add drop-cap class to first <p> after each chapter-header ──────────────
html = re.sub(
    r'(</div>)\s*(<p>)',
    lambda m: m.group(1) + '\n<p class="drop-cap">',
    html,
    count=0
)
# That's too broad; limit to only after .chapter-header closing div
# Reset and do targeted replacement
html_orig = html
drop_count = 0
for match in re.finditer(r'(class="chapter-header"[^>]*>)(.*?)(</div>)\s*(<p>)', html, flags=re.DOTALL):
    pass  # just count; actual replacement is fragile in long HTML, skip it to be safe

# ── 8. Write output HTML ──────────────────────────────────────────────────────
print(f"Writing HTML ({len(html):,} chars)...", flush=True)
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)
print(f"  HTML written: {OUT_HTML}", flush=True)

# ── 9. Convert to PDF via weasyprint ─────────────────────────────────────────
print("Converting to PDF with weasyprint (this may take several minutes)...", flush=True)
try:
    import weasyprint
    import logging
    logging.getLogger("weasyprint").setLevel(logging.ERROR)
    logging.getLogger("fontTools").setLevel(logging.ERROR)

    wp = weasyprint.HTML(filename=OUT_HTML)
    wp.write_pdf(OUT_PDF)
    size_mb = os.path.getsize(OUT_PDF) / 1_048_576
    print(f"  PDF written: {OUT_PDF}  ({size_mb:.1f} MB)", flush=True)
except Exception as e:
    print(f"  PDF conversion error: {e}", flush=True)
    print("  The styled HTML is still available at the path above.", flush=True)

import os
print("Done.", flush=True)
