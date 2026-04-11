"""
gen_pdf_clean.py
Injects print-hardened CSS into the oxidized HTML and generates a clean PDF.
Fixes: code block overflow, MathJax timing, page-break cropping, image clipping.
"""
import re, os, shutil

IN_HTML  = r"C:/Users/Matthew/Desktop/srfm-experiments/event_horizon_book_oxidized.html"
OUT_HTML = r"C:/Users/Matthew/Desktop/srfm-experiments/event_horizon_book_print.html"
OUT_PDF  = r"C:/Users/Matthew/Desktop/srfm-experiments/event_horizon_book.pdf"
DESK     = r"C:/Users/Matthew/Desktop"

print("Reading HTML...", flush=True)
with open(IN_HTML, "r", encoding="utf-8") as f:
    html = f.read()

# ── 1. Inject print-hardened CSS immediately before </style> ─────────────────
PRINT_CSS = """
/* ======================================================
   PRINT HARDENING — injected by gen_pdf_clean.py
   ====================================================== */

/* Force all images to stay within the page content box */
img {
  max-width: 100% !important;
  width: auto !important;
  height: auto !important;
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.fig img {
  max-width: 96% !important;
  width: 96% !important;
}

/* Code blocks: wrap instead of scroll, smaller font, no horizontal overflow */
pre, .code {
  white-space: pre-wrap !important;
  word-break: break-all !important;
  overflow: visible !important;
  overflow-x: visible !important;
  font-size: 0.72em !important;
  line-height: 1.45 !important;
  max-width: 100% !important;
}

/* Tables: never overflow the page */
table {
  table-layout: fixed !important;
  max-width: 100% !important;
  word-break: break-word;
}
td, th {
  word-break: break-word;
  overflow-wrap: break-word;
}

/* Page-break rules: never cut these mid-element */
.fig               { break-inside: avoid; page-break-inside: avoid; }
.nonphd            { break-inside: avoid; page-break-inside: avoid; }
.takeaway          { break-inside: avoid; page-break-inside: avoid; }
.linkedin          { break-inside: avoid; page-break-inside: avoid; }
.box-info          { break-inside: avoid; page-break-inside: avoid; }
.box-warn          { break-inside: avoid; page-break-inside: avoid; }
.pull-quote        { break-inside: avoid; page-break-inside: avoid; }
.abstract          { break-inside: avoid; page-break-inside: avoid; }
figcaption         { break-before: avoid; page-break-before: avoid; }
h2, h3, h4         { break-after: avoid;  page-break-after: avoid; }
pre, .code         { break-inside: avoid; page-break-inside: avoid; }

/* Part and chapter headers always start on a new page */
.part-header    { break-before: page; page-break-before: always; }
.chapter-header { break-before: page; page-break-before: always; }

/* Body text: justified, proper hyphenation */
body {
  font-size: 11.5pt !important;
  line-height: 1.7 !important;
  max-width: none !important;
  width: 100% !important;
  padding: 0 !important;
  margin: 0 !important;
  text-align: justify;
  hyphens: auto;
  -webkit-hyphens: auto;
}

/* Headings: slightly tighter for print */
h1 { font-size: 2.2em !important; }
h2 { font-size: 1.4em !important; margin-top: 2em !important; }
h3 { font-size: 1.15em !important; }

/* Math display blocks: let them break if needed */
mjx-container[display="true"] {
  max-width: 100% !important;
  overflow: visible !important;
}

/* Ensure the journal meta and footer are visible */
.journal-meta { font-size: 7pt !important; }
footer { font-size: 7pt !important; margin-top: 3em !important; }

/* TOC links: no underline in print */
.toc a { text-decoration: none !important; color: var(--ink-light) !important; }
"""

html = html.replace('</style>', PRINT_CSS + '\n</style>', 1)
print("  Print CSS injected.", flush=True)

# ── 2. Write print-ready HTML ─────────────────────────────────────────────────
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)
print(f"  Written: {OUT_HTML}", flush=True)

# ── 3. Generate PDF with Playwright, waiting for MathJax ─────────────────────
print("Launching Playwright...", flush=True)
from playwright.sync_api import sync_playwright

file_url = "file:///" + OUT_HTML.replace("\\", "/")

with sync_playwright() as p:
    browser = p.chromium.launch()

    # Use a viewport that matches letter-paper content width at 96dpi
    # Letter = 8.5in, 1in margins each side => 6.5in content = 624px at 96dpi
    # Use 816px wide (8.5in) so Chromium can compute proper layout
    page = browser.new_page(viewport={"width": 816, "height": 1056})

    print("  Loading HTML...", flush=True)
    page.goto(file_url, wait_until="networkidle", timeout=120000)

    # Wait for MathJax to fully typeset the document
    print("  Waiting for MathJax...", flush=True)
    try:
        page.wait_for_function(
            "() => typeof MathJax !== 'undefined' && MathJax.typesetPromise !== undefined",
            timeout=15000
        )
        page.evaluate("async () => { await MathJax.typesetPromise(); }")
        page.wait_for_timeout(2000)  # extra buffer for rendering
        print("  MathJax done.", flush=True)
    except Exception as e:
        print(f"  MathJax wait skipped: {e}", flush=True)
        page.wait_for_timeout(3000)

    # Extra wait for all images to fully decode
    page.evaluate("""async () => {
        const imgs = Array.from(document.images);
        await Promise.all(imgs.map(img => img.decode ? img.decode().catch(()=>{}) : Promise.resolve()));
    }""")
    page.wait_for_timeout(1000)

    print("  Generating PDF...", flush=True)
    page.pdf(
        path=OUT_PDF,
        format="Letter",
        margin={"top": "0.9in", "bottom": "0.9in", "left": "1in", "right": "1in"},
        print_background=True,
        prefer_css_page_size=False,
    )
    browser.close()

size_mb = os.path.getsize(OUT_PDF) / 1_048_576
print(f"  PDF: {OUT_PDF}  ({size_mb:.1f} MB)", flush=True)

# ── 4. Copy both files to Desktop ─────────────────────────────────────────────
shutil.copy(OUT_PDF,  os.path.join(DESK, "event_horizon_book.pdf"))
# Also update the desktop HTML to the print version
shutil.copy(OUT_HTML, os.path.join(DESK, "event_horizon_book_oxidized.html"))
print("Copied to Desktop.", flush=True)
print("Done.", flush=True)
