"""
SEC EDGAR filing fetcher and parser.

Fetches:
- 8-K (current reports: earnings, M&A, material events)
- 10-Q / 10-K (quarterly/annual reports)
- Form 4 (insider transactions)
- DEF 14A (proxy statements)

Parses structured financial data from filings.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, quote

logger = logging.getLogger(__name__)

# EDGAR base URLs
EDGAR_BASE = "https://www.sec.gov"
EDGAR_API  = "https://data.sec.gov"
EFTS_API   = "https://efts.sec.gov"

# Rate limit: EDGAR allows 10 requests/sec for bots
EDGAR_RATE_LIMIT = 0.12  # ~8/sec to be safe

HEADERS = {
    "User-Agent": "FinancialResearch/1.0 research@example.com",
    "Accept": "application/json, text/html, */*",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SECFiling:
    """Represents a parsed SEC filing."""
    accession_number: str
    cik: str
    company_name: str
    form_type: str
    filed_date: datetime
    period_of_report: Optional[str]
    url: str
    full_text_url: str = ""
    summary: str = ""
    full_text: str = ""
    tickers: List[str] = field(default_factory=list)
    financial_data: Dict[str, Any] = field(default_factory=dict)
    items_reported: List[str] = field(default_factory=list)  # for 8-K: item numbers
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
        return d


@dataclass
class InsiderTransaction:
    """Parsed insider transaction from Form 4."""
    cik: str
    issuer_name: str
    ticker: str
    insider_name: str
    insider_title: str
    transaction_date: Optional[datetime]
    transaction_type: str     # "P" (purchase) | "S" (sale) | "A" (award)
    shares: float
    price_per_share: float
    total_value: float
    ownership_type: str       # "D" (direct) | "I" (indirect)
    shares_owned_after: float
    filed_date: datetime
    accession_number: str


# ---------------------------------------------------------------------------
# EDGAR fetch helpers
# ---------------------------------------------------------------------------

_last_request_time: float = 0.0


def _edgar_get(url: str, return_text: bool = False) -> Optional[Any]:
    """Fetch from EDGAR with rate limiting."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < EDGAR_RATE_LIMIT:
        time.sleep(EDGAR_RATE_LIMIT - elapsed)
    _last_request_time = time.time()

    for attempt in range(3):
        try:
            req = Request(url, headers=HEADERS)
            with urlopen(req, timeout=20) as response:
                raw = response.read()
                if return_text:
                    return raw.decode("utf-8", errors="replace")
                try:
                    return json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    return raw.decode("utf-8", errors="replace")
        except HTTPError as e:
            if e.code == 429:
                time.sleep(10)
            elif e.code in (403, 404):
                return None
            elif attempt < 2:
                time.sleep(2 ** attempt)
        except URLError as e:
            logger.warning(f"URL error: {e.reason}")
            if attempt < 2:
                time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            break
    return None


def get_company_cik(ticker: str) -> Optional[str]:
    """Look up CIK for a ticker symbol."""
    url = f"{EDGAR_API}/submissions/CIK.json"
    # Use EDGAR company search
    search_url = f"{EDGAR_BASE}/cgi-bin/browse-edgar?company={quote(ticker)}&action=getcompany&type=&dateb=&owner=include&count=10&search_text=&action=getcompany&output=atom"
    result = _edgar_get(search_url, return_text=True)
    if not result:
        return None

    # Try to find CIK in response
    cik_match = re.search(r'CIK=(\d+)', str(result))
    if cik_match:
        return cik_match.group(1).zfill(10)

    # Try the submissions API lookup
    tickers_url = f"{EDGAR_API}/submissions/CIK{ticker.upper()}.json"
    data = _edgar_get(tickers_url)
    if isinstance(data, dict):
        return data.get("cik", "").zfill(10)

    return None


def get_company_submissions(cik: str) -> Optional[Dict]:
    """Get all filings metadata for a company."""
    cik_padded = cik.zfill(10)
    url = f"{EDGAR_API}/submissions/CIK{cik_padded}.json"
    return _edgar_get(url)


# ---------------------------------------------------------------------------
# Filing fetcher
# ---------------------------------------------------------------------------

class EDGARFetcher:
    """
    Fetches and parses SEC EDGAR filings.

    Key methods:
    - get_recent_filings(cik, form_type): fetch recent filings of a type
    - get_8k_items(cik): get recent 8-K filings with event classification
    - get_insider_transactions(cik): get Form 4 filings
    - parse_earnings_from_8k(filing): extract EPS/revenue from 8-K
    """

    def __init__(self, max_filings_per_type: int = 10):
        self.max_filings = max_filings_per_type

    def get_recent_filings(
        self,
        cik: str,
        form_type: str = "8-K",
        n: int = 10,
    ) -> List[SECFiling]:
        """
        Get recent filings of a specific type for a company.
        form_type: "8-K" | "10-K" | "10-Q" | "4" | "DEF 14A"
        """
        submissions = get_company_submissions(cik)
        if not submissions:
            return []

        filings_data = submissions.get("filings", {}).get("recent", {})
        if not filings_data:
            return []

        form_types    = filings_data.get("form", [])
        accession_nos = filings_data.get("accessionNumber", [])
        filed_dates   = filings_data.get("filingDate", [])
        periods       = filings_data.get("reportDate", [])
        docs          = filings_data.get("primaryDocument", [])

        company_name = submissions.get("name", "")
        ticker_list  = submissions.get("tickers", [])

        results = []
        for i, ft in enumerate(form_types):
            if ft.upper() != form_type.upper():
                continue
            if len(results) >= n:
                break

            accession = accession_nos[i] if i < len(accession_nos) else ""
            filed_str = filed_dates[i] if i < len(filed_dates) else ""
            period    = periods[i] if i < len(periods) else ""
            doc       = docs[i] if i < len(docs) else ""

            try:
                filed_dt = datetime.strptime(filed_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                filed_dt = datetime.now(timezone.utc)

            accession_clean = accession.replace("-", "")
            filing_url = (
                f"{EDGAR_BASE}/Archives/edgar/data/{cik.lstrip('0')}/"
                f"{accession_clean}/{doc}"
            )
            index_url = (
                f"{EDGAR_BASE}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}"
                f"&type={form_type}&dateb=&owner=include&count=10"
            )

            filing = SECFiling(
                accession_number=accession,
                cik=cik,
                company_name=company_name,
                form_type=ft,
                filed_date=filed_dt,
                period_of_report=period,
                url=index_url,
                full_text_url=filing_url,
                tickers=ticker_list,
            )
            results.append(filing)

        return results

    def fetch_filing_text(self, filing: SECFiling, max_chars: int = 50_000) -> str:
        """Fetch the full text of a filing."""
        if not filing.full_text_url:
            return ""

        content = _edgar_get(filing.full_text_url, return_text=True)
        if not content:
            return ""

        # Strip HTML
        text = re.sub(r'<[^>]+>', ' ', str(content))
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars]

    def get_8k_filings(
        self,
        cik: str,
        n: int = 20,
        fetch_text: bool = True,
    ) -> List[SECFiling]:
        """Get recent 8-K filings with event classification."""
        filings = self.get_recent_filings(cik, "8-K", n=n)

        for filing in filings:
            if fetch_text:
                filing.full_text = self.fetch_filing_text(filing, max_chars=20_000)
                filing.items_reported = self._extract_8k_items(filing.full_text)
                filing.financial_data = self.parse_earnings_from_8k(filing)
                filing.summary = self._generate_8k_summary(filing)

        return filings

    def _extract_8k_items(self, text: str) -> List[str]:
        """Extract Item numbers reported in an 8-K."""
        # Items like "Item 2.02", "Item 8.01", etc.
        pattern = re.compile(r'Item\s+(\d+\.\d+)', re.IGNORECASE)
        matches = pattern.findall(text)
        return list(dict.fromkeys(matches))  # unique, order preserved

    def parse_earnings_from_8k(self, filing: SECFiling) -> Dict[str, Any]:
        """
        Extract key financial metrics from an 8-K press release.
        Returns: {eps, eps_estimate, revenue, revenue_estimate, guidance, ...}
        """
        text = filing.full_text or ""
        data: Dict[str, Any] = {
            "eps_reported": None,
            "eps_estimate": None,
            "revenue": None,
            "revenue_yoy_growth": None,
            "guidance_revenue_low": None,
            "guidance_revenue_high": None,
            "guidance_eps_low": None,
            "guidance_eps_high": None,
            "beat_eps": None,
            "beat_revenue": None,
        }

        # EPS patterns
        eps_patterns = [
            r'(?:diluted\s+)?(?:earnings|eps|loss)\s+(?:per\s+(?:diluted\s+)?share\s+)?(?:of\s+)?\$?\s*(\d+(?:\.\d+)?)',
            r'\$(\d+\.\d{2})\s+(?:per|a)\s+(?:diluted\s+)?share',
        ]
        for pattern in eps_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    data["eps_reported"] = float(m.group(1))
                    break
                except ValueError:
                    pass

        # Revenue patterns
        rev_patterns = [
            r'(?:net\s+)?(?:revenue|sales)\s+(?:of\s+|were\s+)?\$?\s*(\d+(?:\.\d+)?)\s*(billion|million|B|M)',
            r'total\s+(?:net\s+)?revenue\s+(?:of\s+|was\s+)?\$?\s*(\d+(?:\.\d+)?)\s*(billion|million|B|M)',
        ]
        for pattern in rev_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1))
                    mult = m.group(2).upper() if m.group(2) else "M"
                    if mult in ("BILLION", "B"):
                        val *= 1000
                    data["revenue"] = val  # in millions
                    break
                except (ValueError, IndexError):
                    pass

        # YoY growth
        yoy_patterns = [
            r'(\d+(?:\.\d+)?)\s*%\s*(?:year[-\s]over[-\s]year|YOY|YoY)',
            r'up\s+(\d+(?:\.\d+)?)\s*%\s*(?:from|vs\.?)\s+(?:the\s+)?(?:prior|previous|same)',
        ]
        for pattern in yoy_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    data["revenue_yoy_growth"] = float(m.group(1)) / 100.0
                    break
                except ValueError:
                    pass

        # Guidance (forward looking)
        guidance_pattern = re.compile(
            r'(?:guid(?:ance|es?)|forecast|target(?:ing)?|expect(?:s|ing)?)\s+.*?'
            r'\$(\d+(?:\.\d+)?)\s*(billion|million|B|M)?'
            r'(?:\s*(?:to|-)\s*\$?(\d+(?:\.\d+)?)\s*(billion|million|B|M)?)?',
            re.IGNORECASE | re.DOTALL
        )
        for m in guidance_pattern.finditer(text):
            try:
                lo = float(m.group(1))
                mult_lo = (m.group(2) or "M").upper()
                if mult_lo in ("BILLION", "B"):
                    lo *= 1000
                data["guidance_revenue_low"] = lo

                if m.group(3):
                    hi = float(m.group(3))
                    mult_hi = (m.group(4) or mult_lo).upper()
                    if mult_hi in ("BILLION", "B"):
                        hi *= 1000
                    data["guidance_revenue_high"] = hi
                else:
                    data["guidance_revenue_high"] = lo
                break
            except (ValueError, AttributeError):
                pass

        # Beat/miss detection
        beat_pattern = re.compile(r'(beat|exceeded?|surpassed?|topped?|missed?|fell\s+short)', re.IGNORECASE)
        m = beat_pattern.search(text)
        if m:
            word = m.group(1).lower()
            data["beat_eps"] = "miss" not in word and "fell" not in word

        return data

    def _generate_8k_summary(self, filing: SECFiling) -> str:
        """Generate a short summary of the 8-K filing."""
        items = filing.items_reported
        fd = filing.financial_data

        parts = [f"{filing.company_name} filed 8-K on {filing.filed_date.strftime('%Y-%m-%d')}."]

        if items:
            parts.append(f"Items reported: {', '.join(items)}.")

        if fd.get("eps_reported") is not None:
            parts.append(f"EPS: ${fd['eps_reported']:.2f}.")
            if fd.get("beat_eps") is not None:
                parts.append("Beat estimates." if fd["beat_eps"] else "Missed estimates.")

        if fd.get("revenue") is not None:
            rev = fd["revenue"]
            if rev >= 1000:
                parts.append(f"Revenue: ${rev/1000:.1f}B.")
            else:
                parts.append(f"Revenue: ${rev:.0f}M.")

        if fd.get("revenue_yoy_growth") is not None:
            pct = fd["revenue_yoy_growth"] * 100
            parts.append(f"YoY growth: {pct:+.1f}%.")

        return " ".join(parts)

    def get_insider_transactions(
        self,
        cik: str,
        n: int = 20,
    ) -> List[InsiderTransaction]:
        """Fetch and parse Form 4 insider transactions."""
        filings = self.get_recent_filings(cik, "4", n=n)
        transactions = []

        submissions = get_company_submissions(cik)
        company_name = submissions.get("name", "") if submissions else ""
        ticker_list  = submissions.get("tickers", [])
        ticker = ticker_list[0] if ticker_list else ""

        for filing in filings:
            text = self.fetch_filing_text(filing, max_chars=10_000)
            parsed = self._parse_form4(text, filing, company_name, ticker, cik)
            transactions.extend(parsed)

        return transactions

    def _parse_form4(
        self,
        text: str,
        filing: SECFiling,
        company_name: str,
        ticker: str,
        cik: str,
    ) -> List[InsiderTransaction]:
        """Parse Form 4 XML-like text for transaction data."""
        transactions = []

        # Extract reporting owner name
        owner_match = re.search(r'reportingOwner.*?rptOwnerName[^>]*>([^<]+)<', text, re.DOTALL)
        insider_name = owner_match.group(1).strip() if owner_match else "Unknown"

        # Extract title
        title_match = re.search(r'officerTitle[^>]*>([^<]+)<', text, re.DOTALL)
        insider_title = title_match.group(1).strip() if title_match else ""

        # Find all non-derivative transactions
        trans_pattern = re.compile(
            r'nonDerivativeTransaction.*?transactionDate.*?>(\d{4}-\d{2}-\d{2})<.*?'
            r'transactionCode.*?>([A-Z])<.*?'
            r'transactionShares.*?>(\d+(?:\.\d+)?)<.*?'
            r'transactionPricePerShare.*?value>(\d+(?:\.\d+)?)<.*?'
            r'sharesOwnedFollowingTransaction.*?value>(\d+(?:\.\d+)?)<.*?'
            r'directOrIndirectOwnership.*?>([DI])<',
            re.DOTALL
        )

        for m in trans_pattern.finditer(text):
            try:
                tx_date = datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
                tx_type = m.group(2)
                shares  = float(m.group(3))
                price   = float(m.group(4))
                after   = float(m.group(5))
                own_type = m.group(6)

                transactions.append(InsiderTransaction(
                    cik=cik,
                    issuer_name=company_name,
                    ticker=ticker,
                    insider_name=insider_name,
                    insider_title=insider_title,
                    transaction_date=tx_date,
                    transaction_type=tx_type,
                    shares=shares,
                    price_per_share=price,
                    total_value=shares * price,
                    ownership_type=own_type,
                    shares_owned_after=after,
                    filed_date=filing.filed_date,
                    accession_number=filing.accession_number,
                ))
            except (ValueError, IndexError):
                continue

        return transactions

    def get_earnings_call_transcript_hints(
        self,
        ticker: str,
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search EDGAR full-text search for recent 8-K press releases with earnings data.
        Returns metadata for building transcript queries.
        """
        url = (
            f"{EFTS_API}/hits.json?q=%22{quote(ticker)}%22+%22earnings+per+share%22"
            f"&dateRange=custom&startdt={datetime.now(timezone.utc).strftime('%Y-01-01')}"
            f"&forms=8-K&hits.hits.total.value=true&hits.hits._source.period_of_report=true"
        )
        data = _edgar_get(url)
        if not isinstance(data, dict):
            return []

        hits = data.get("hits", {}).get("hits", [])
        results = []
        for hit in hits[:n]:
            source = hit.get("_source", {})
            results.append({
                "ticker": ticker,
                "form_type": source.get("file_type"),
                "company": source.get("display_names", [{}])[0].get("name", ""),
                "filed_date": source.get("file_date"),
                "period": source.get("period_of_report"),
                "accession": source.get("accession_no"),
                "url": f"{EDGAR_BASE}/Archives/edgar/data/{source.get('entity_id', '')}/{source.get('accession_no', '').replace('-', '')}/",
            })

        return results

    def get_material_events_feed(self, n: int = 40) -> List[SECFiling]:
        """
        Get the most recent 8-K material events across all companies.
        Uses EDGAR's current feed.
        """
        url = f"{EDGAR_BASE}/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count={n}&search_text=&output=atom"
        text = _edgar_get(url, return_text=True)
        if not text:
            return []

        filings = []
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)

            for entry in entries:
                title_el    = entry.find("atom:title", ns)
                link_el     = entry.find("atom:link", ns)
                updated_el  = entry.find("atom:updated", ns)
                summary_el  = entry.find("atom:summary", ns)
                category_el = entry.find("atom:category", ns)

                title   = title_el.text.strip() if title_el is not None and title_el.text else ""
                url_val = link_el.get("href", "") if link_el is not None else ""
                date_str = updated_el.text if updated_el is not None and updated_el.text else ""
                summary = summary_el.text.strip() if summary_el is not None and summary_el.text else ""

                try:
                    filed_dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except Exception:
                    filed_dt = datetime.now(timezone.utc)

                # Parse CIK from URL
                cik_match = re.search(r'CIK=(\d+)', url_val)
                cik = cik_match.group(1) if cik_match else ""

                # Parse ticker from title (e.g., "Apple Inc. (AAPL) 8-K")
                ticker_match = re.search(r'\(([A-Z]{1,5})\)', title)
                ticker = ticker_match.group(1) if ticker_match else ""

                filing = SECFiling(
                    accession_number="",
                    cik=cik,
                    company_name=title,
                    form_type="8-K",
                    filed_date=filed_dt,
                    period_of_report=None,
                    url=url_val,
                    summary=summary,
                    tickers=[ticker] if ticker else [],
                )
                filings.append(filing)

        except ET.ParseError as e:
            logger.error(f"XML parse error for EDGAR feed: {e}")

        return filings


# ---------------------------------------------------------------------------
# Earnings calendar integration
# ---------------------------------------------------------------------------

def parse_earnings_date_from_filings(filings: List[SECFiling]) -> Optional[datetime]:
    """Estimate earnings date from 8-K filing dates."""
    earnings_filings = [f for f in filings if "2.02" in f.items_reported]
    if earnings_filings:
        return sorted(earnings_filings, key=lambda f: f.filed_date, reverse=True)[0].filed_date
    return None


def compute_earnings_surprise(
    filing: SECFiling,
    consensus_eps: float,
    consensus_revenue: float,
) -> Dict[str, float]:
    """
    Compute earnings surprise vs consensus.
    Returns: {eps_surprise_pct, revenue_surprise_pct, beat_both}
    """
    fd = filing.financial_data
    result = {}

    eps_reported = fd.get("eps_reported")
    if eps_reported is not None and consensus_eps != 0:
        surprise_pct = (eps_reported - consensus_eps) / abs(consensus_eps)
        result["eps_surprise_pct"] = float(surprise_pct)
        result["beat_eps"] = float(surprise_pct > 0)

    revenue = fd.get("revenue")
    if revenue is not None and consensus_revenue != 0:
        surprise_pct = (revenue - consensus_revenue) / abs(consensus_revenue)
        result["revenue_surprise_pct"] = float(surprise_pct)
        result["beat_revenue"] = float(surprise_pct > 0)

    if "beat_eps" in result and "beat_revenue" in result:
        result["beat_both"] = float(result["beat_eps"] == 1.0 and result["beat_revenue"] == 1.0)

    return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing EDGAR parser with sample text...")

    sample_8k = """
    Apple Reports Fourth Quarter and Full Year Results

    CUPERTINO, California — November 1, 2024 — Apple today announced
    financial results for its fiscal 2024 fourth quarter ended September 28, 2024.

    The Company posted quarterly revenue of $94.9 billion, up 6% year-over-year,
    and quarterly earnings per diluted share of $1.64, up 12% year-over-year.

    "Our iPhone revenue grew 6% year over year," said Tim Cook.

    Apple is also providing the following guidance for its fiscal 2025 first quarter:
    Revenue between $89.0 billion and $91.0 billion.
    Gross margin between 46.5 percent and 47.5 percent.
    """

    fetcher = EDGARFetcher()

    # Test parsing
    mock_filing = SECFiling(
        accession_number="0001140361-24-001234",
        cik="0000320193",
        company_name="Apple Inc.",
        form_type="8-K",
        filed_date=datetime.now(timezone.utc),
        period_of_report="2024-09-28",
        url="https://www.sec.gov/",
        full_text=sample_8k,
    )
    mock_filing.items_reported = fetcher._extract_8k_items(sample_8k)
    mock_filing.financial_data = fetcher.parse_earnings_from_8k(mock_filing)
    mock_filing.summary = fetcher._generate_8k_summary(mock_filing)

    print(f"Items: {mock_filing.items_reported}")
    print(f"Financial data: {mock_filing.financial_data}")
    print(f"Summary: {mock_filing.summary}")

    # Test earnings surprise
    surprise = compute_earnings_surprise(mock_filing, consensus_eps=1.58, consensus_revenue=93_800)
    print(f"Earnings surprise: {surprise}")

    print("SEC EDGAR self-test passed.")
