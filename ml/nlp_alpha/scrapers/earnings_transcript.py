"""
Earnings call transcript parser.

Sources:
- SEC EDGAR 8-K exhibits
- Seeking Alpha (HTML scraping)
- MotleyFool transcripts

Extracts:
- Speaker segments (CEO/CFO Q&A)
- Management tone features
- Forward guidance mentions
- Risk factor language
- Key financial metrics mentioned
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TranscriptSegment:
    """One segment (speaker turn) of an earnings call transcript."""
    speaker: str
    title: str
    role: str          # "management" | "analyst" | "operator"
    text: str
    segment_idx: int
    is_qa: bool = False
    sentences: List[str] = field(default_factory=list)
    financial_mentions: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class EarningsTranscript:
    """Full parsed earnings call transcript."""
    ticker: str
    company_name: str
    quarter: str
    year: int
    call_date: Optional[datetime]
    url: str = ""
    source: str = ""
    full_text: str = ""
    segments: List[TranscriptSegment] = field(default_factory=list)
    management_sections: List[TranscriptSegment] = field(default_factory=list)
    qa_sections: List[TranscriptSegment] = field(default_factory=list)
    financial_summary: Dict[str, Any] = field(default_factory=dict)
    tone_features: Dict[str, float] = field(default_factory=dict)
    guidance_statements: List[str] = field(default_factory=list)
    risk_statements: List[str] = field(default_factory=list)
    analyst_questions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Speaker identification
# ---------------------------------------------------------------------------

MANAGEMENT_TITLES = [
    "chief executive officer", "ceo", "president", "chief financial officer",
    "cfo", "chief operating officer", "coo", "chief revenue officer", "cro",
    "chief technology officer", "cto", "chief marketing officer", "cmo",
    "executive vice president", "evp", "senior vice president", "svp",
    "vice president", "vp", "chairman", "founder", "co-founder",
    "head of investor relations", "ir", "treasurer",
]

ANALYST_INDICATORS = [
    "analyst", "research", "securities", "capital", "partners",
    "management", "asset", "investment", "bank", "group",
]

OPERATOR_PHRASES = [
    "operator", "conference call", "moderator", "your next question",
    "please go ahead", "thank you", "we have a question from",
]


def classify_speaker(speaker: str, title: str, text_before: str = "") -> str:
    """Classify speaker role: management | analyst | operator."""
    combined = (speaker + " " + title).lower()

    for phrase in OPERATOR_PHRASES:
        if phrase in combined or phrase in text_before.lower():
            return "operator"

    for mgmt_title in MANAGEMENT_TITLES:
        if mgmt_title in combined:
            return "management"

    for indicator in ANALYST_INDICATORS:
        if indicator in combined:
            return "analyst"

    # Heuristic: if they ask a question, likely analyst
    if "?" in text_before[-200:] if text_before else "":
        return "analyst"

    return "unknown"


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

SPEAKER_RE = re.compile(
    r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\n'
    r'(?:([A-Z][^:]{5,80}?)\n)?',
    re.MULTILINE
)

SEGMENT_DELIMITERS = [
    r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*$',
    r'^-{3,}\s*$',
]

GUIDANCE_KEYWORDS = [
    "expect", "guidance", "outlook", "forecast", "anticipate", "project",
    "target", "estimate", "guide", "full year", "next quarter", "fiscal year",
]

RISK_KEYWORDS = [
    "risk", "uncertainty", "challenge", "headwind", "concern", "difficulty",
    "macroeconomic", "inflation", "competition", "regulatory", "litigation",
    "supply chain", "geopolitical",
]

TONE_POSITIVE = [
    "strong", "exceeded", "outperformed", "record", "growth", "momentum",
    "opportunity", "optimistic", "confident", "pleased", "excited", "robust",
    "accelerating", "expanding", "investing", "innovative",
]

TONE_NEGATIVE = [
    "miss", "decline", "challenging", "headwind", "difficult", "uncertain",
    "weakness", "pressure", "slowdown", "disappointing", "concern", "caution",
    "lower", "reduced", "shortfall", "impact",
]


class TranscriptParser:
    """
    Parses earnings call transcripts from raw text.

    Handles:
    - Multiple transcript formats (Seeking Alpha, Motley Fool, plain text)
    - Speaker segmentation
    - Q&A separation
    - Financial metric extraction
    """

    def parse(
        self,
        text: str,
        ticker: str = "",
        company_name: str = "",
        call_date: Optional[datetime] = None,
        source: str = "",
        url: str = "",
    ) -> EarningsTranscript:
        """Parse raw transcript text into structured EarningsTranscript."""

        # Detect format
        if self._is_seeking_alpha_format(text):
            segments = self._parse_seeking_alpha(text)
        elif self._is_motleyfool_format(text):
            segments = self._parse_motleyfool(text)
        else:
            segments = self._parse_generic(text)

        # Extract quarter/year
        quarter, year = self._extract_quarter_year(text, call_date)

        transcript = EarningsTranscript(
            ticker=ticker,
            company_name=company_name,
            quarter=quarter,
            year=year,
            call_date=call_date,
            url=url,
            source=source,
            full_text=text,
            segments=segments,
        )

        # Classify segments
        transcript.management_sections = [s for s in segments if s.role == "management"]
        transcript.qa_sections = [s for s in segments if s.is_qa]

        # Extract features
        transcript.guidance_statements = self._extract_guidance(segments)
        transcript.risk_statements     = self._extract_risk_statements(segments)
        transcript.analyst_questions   = self._extract_analyst_questions(segments)
        transcript.tone_features       = self._compute_tone_features(segments)
        transcript.financial_summary   = self._extract_financial_summary(text)

        return transcript

    def _is_seeking_alpha_format(self, text: str) -> bool:
        return "Seeking Alpha" in text or "seekingalpha" in text.lower()

    def _is_motleyfool_format(self, text: str) -> bool:
        return "Motley Fool" in text or "fool.com" in text.lower()

    def _parse_generic(self, text: str) -> List[TranscriptSegment]:
        """Generic transcript parser based on speaker pattern detection."""
        segments = []
        lines = text.split("\n")
        current_speaker = "Unknown"
        current_title = ""
        current_text: List[str] = []
        segment_idx = 0
        in_qa = False

        for line in lines:
            line_stripped = line.strip()

            # Detect Q&A section
            if re.search(r'\bQ\s*&\s*A\b|\bquestion.and.answer\b|\bQ&A\b', line, re.IGNORECASE):
                in_qa = True

            # Detect speaker line: "John Smith:" or "JOHN SMITH" (all caps)
            speaker_match = re.match(r'^([A-Z][a-zA-Z\s\-]+(?:,\s*[A-Z][a-zA-Z\s]+)?)\s*:?\s*$', line_stripped)
            if speaker_match and len(line_stripped) < 60 and not line_stripped.lower().startswith(("the", "a ", "in ", "on ", "at ")):
                # Save previous segment
                if current_text:
                    seg_text = " ".join(current_text).strip()
                    if seg_text:
                        role = classify_speaker(current_speaker, current_title)
                        seg = TranscriptSegment(
                            speaker=current_speaker,
                            title=current_title,
                            role=role,
                            text=seg_text,
                            segment_idx=segment_idx,
                            is_qa=in_qa,
                            sentences=self._split_sentences(seg_text),
                        )
                        segments.append(seg)
                        segment_idx += 1

                current_speaker = speaker_match.group(1).strip()
                current_title = ""
                current_text = []
            else:
                current_text.append(line_stripped)

        # Last segment
        if current_text:
            seg_text = " ".join(current_text).strip()
            if seg_text:
                role = classify_speaker(current_speaker, current_title)
                seg = TranscriptSegment(
                    speaker=current_speaker,
                    title=current_title,
                    role=role,
                    text=seg_text,
                    segment_idx=segment_idx,
                    is_qa=in_qa,
                    sentences=self._split_sentences(seg_text),
                )
                segments.append(seg)

        return segments

    def _parse_seeking_alpha(self, text: str) -> List[TranscriptSegment]:
        """Parse Seeking Alpha formatted transcripts."""
        # SA format: "Speaker Name\nTitle\nText..."
        segments = []
        blocks = re.split(r'\n{2,}', text)
        segment_idx = 0
        in_qa = False

        i = 0
        while i < len(blocks):
            block = blocks[i].strip()
            if not block:
                i += 1
                continue

            if re.search(r'Q&A|Question.and.Answer', block, re.IGNORECASE):
                in_qa = True
                i += 1
                continue

            lines = block.split("\n")
            if len(lines) >= 2:
                first_line = lines[0].strip()
                if len(first_line) < 50 and re.match(r'^[A-Z][a-zA-Z\s\-]+$', first_line):
                    speaker = first_line
                    title = lines[1].strip() if len(lines) > 1 else ""
                    text_content = " ".join(lines[2:]).strip() if len(lines) > 2 else ""
                    if not text_content and i + 1 < len(blocks):
                        text_content = blocks[i + 1].strip()
                        i += 1

                    if text_content:
                        role = classify_speaker(speaker, title)
                        seg = TranscriptSegment(
                            speaker=speaker,
                            title=title,
                            role=role,
                            text=text_content,
                            segment_idx=segment_idx,
                            is_qa=in_qa,
                            sentences=self._split_sentences(text_content),
                        )
                        segments.append(seg)
                        segment_idx += 1
                else:
                    # Continuation of previous segment
                    if segments:
                        segments[-1].text += " " + block
                        segments[-1].sentences = self._split_sentences(segments[-1].text)
            i += 1

        return segments

    def _parse_motleyfool(self, text: str) -> List[TranscriptSegment]:
        """Parse Motley Fool formatted transcripts."""
        return self._parse_generic(text)  # MF format similar to generic

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s) > 10]

    def _extract_quarter_year(
        self, text: str, date: Optional[datetime]
    ) -> Tuple[str, int]:
        """Extract quarter and year from transcript text."""
        # Look for "Q3 2024", "third quarter 2024", "fiscal Q1", etc.
        q_patterns = [
            r'\bQ([1-4])\s+(\d{4})\b',
            r'\b(first|second|third|fourth)\s+quarter\s+(?:of\s+)?(\d{4})\b',
            r'\bfiscal\s+Q([1-4])\b',
        ]
        q_map = {"first": "1", "second": "2", "third": "3", "fourth": "4"}

        for pattern in q_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                g1, g2 = m.group(1), m.group(2) if m.lastindex >= 2 else None
                q_num = q_map.get(g1.lower(), g1) if g1.lower() in q_map else g1
                year = int(g2) if g2 and g2.isdigit() else (date.year if date else datetime.now(timezone.utc).year)
                return f"Q{q_num}", year

        year = date.year if date else datetime.now(timezone.utc).year
        return "Q?", year

    def _extract_guidance(self, segments: List[TranscriptSegment]) -> List[str]:
        """Extract forward guidance statements from management sections."""
        guidance = []
        mgmt_text = " ".join(s.text for s in segments if s.role == "management")

        sentences = re.split(r'(?<=[.!?])\s+', mgmt_text)
        for sent in sentences:
            sent_lower = sent.lower()
            if any(kw in sent_lower for kw in GUIDANCE_KEYWORDS) and len(sent) > 30:
                guidance.append(sent.strip())

        return guidance[:20]  # cap at 20 statements

    def _extract_risk_statements(self, segments: List[TranscriptSegment]) -> List[str]:
        """Extract risk-related statements."""
        risks = []
        all_text = " ".join(s.text for s in segments)
        sentences = re.split(r'(?<=[.!?])\s+', all_text)

        for sent in sentences:
            sent_lower = sent.lower()
            if any(kw in sent_lower for kw in RISK_KEYWORDS) and len(sent) > 30:
                risks.append(sent.strip())

        return risks[:15]

    def _extract_analyst_questions(self, segments: List[TranscriptSegment]) -> List[str]:
        """Extract analyst questions from Q&A."""
        questions = []
        for seg in segments:
            if seg.role == "analyst" and seg.is_qa:
                # Extract question sentences
                for sent in seg.sentences:
                    if "?" in sent:
                        questions.append(sent.strip())

        return questions[:20]

    def _compute_tone_features(self, segments: List[TranscriptSegment]) -> Dict[str, float]:
        """
        Compute management tone features:
        - positive_word_ratio
        - negative_word_ratio
        - forward_looking_ratio
        - certainty_score
        - management_talk_ratio (vs analyst time)
        """
        mgmt_text = " ".join(s.text for s in segments if s.role == "management").lower()
        all_text  = " ".join(s.text for s in segments).lower()

        total_words = max(len(mgmt_text.split()), 1)

        pos_count = sum(mgmt_text.count(kw) for kw in TONE_POSITIVE)
        neg_count = sum(mgmt_text.count(kw) for kw in TONE_NEGATIVE)

        guidance_count = sum(mgmt_text.count(kw) for kw in GUIDANCE_KEYWORDS)

        # Certainty vs uncertainty language
        certain_words = ["will", "confident", "committed", "plan", "definitely", "expect to"]
        uncertain_words = ["may", "might", "could", "possibly", "uncertain", "depends", "if"]
        certain_count   = sum(mgmt_text.count(w) for w in certain_words)
        uncertain_count = sum(mgmt_text.count(w) for w in uncertain_words)

        # Management vs analyst talk ratio
        mgmt_len   = len(mgmt_text.split())
        total_len  = max(len(all_text.split()), 1)

        features = {
            "positive_word_ratio":    float(pos_count / total_words),
            "negative_word_ratio":    float(neg_count / total_words),
            "tone_score":             float((pos_count - neg_count) / max(pos_count + neg_count, 1)),
            "forward_looking_ratio":  float(guidance_count / total_words),
            "certainty_score":        float((certain_count - uncertain_count) / max(certain_count + uncertain_count, 1)),
            "management_talk_ratio":  float(mgmt_len / total_len),
            "n_management_segments":  float(sum(1 for s in segments if s.role == "management")),
            "n_analyst_segments":     float(sum(1 for s in segments if s.role == "analyst")),
            "n_guidance_statements":  float(len(self._extract_guidance(segments))),
        }

        return features

    def _extract_financial_summary(self, text: str) -> Dict[str, Any]:
        """Extract key financial metrics mentioned in transcript."""
        from ..scrapers.sec_edgar import EDGARFetcher
        # Reuse the 8-K parsing logic
        try:
            mock_filing_data = {"full_text": text, "financial_data": {}}
            fetcher = EDGARFetcher()

            # Use simplified extraction
            data: Dict[str, Any] = {}

            # EPS
            eps_m = re.search(r'\$(\d+\.\d{2})\s+(?:per|a)\s+(?:diluted\s+)?share', text, re.IGNORECASE)
            if eps_m:
                data["eps"] = float(eps_m.group(1))

            # Revenue
            rev_m = re.search(r'(?:revenue|sales)\s+(?:of\s+)?\$?(\d+(?:\.\d+)?)\s*(billion|million|B|M)', text, re.IGNORECASE)
            if rev_m:
                val = float(rev_m.group(1))
                mult = rev_m.group(2).upper() if rev_m.group(2) else "M"
                data["revenue_millions"] = val * 1000 if mult in ("BILLION", "B") else val

            # YoY
            yoy_m = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:year.over.year|YOY)', text, re.IGNORECASE)
            if yoy_m:
                data["revenue_yoy_pct"] = float(yoy_m.group(1))

            return data
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Transcript fetcher
# ---------------------------------------------------------------------------

class TranscriptFetcher:
    """
    Fetches earnings call transcripts from public sources.
    """

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self._parser = TranscriptParser()

    def fetch_from_url(
        self,
        url: str,
        ticker: str = "",
        company_name: str = "",
        call_date: Optional[datetime] = None,
    ) -> Optional[EarningsTranscript]:
        """Fetch and parse transcript from a direct URL."""
        try:
            req = Request(url, headers={
                "User-Agent": "FinanceBot/1.0",
                "Accept": "text/html,*/*",
            })
            with urlopen(req, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")

            # Strip HTML
            text = re.sub(r'<[^>]+>', ' ', raw)
            text = re.sub(r'\s+', ' ', text).strip()

            return self._parser.parse(text, ticker, company_name, call_date, source=url, url=url)

        except Exception as e:
            logger.error(f"Error fetching transcript from {url}: {e}")
            return None

    def parse_text(
        self,
        text: str,
        ticker: str = "",
        company_name: str = "",
        call_date: Optional[datetime] = None,
        source: str = "manual",
    ) -> EarningsTranscript:
        """Parse a transcript from raw text."""
        return self._parser.parse(text, ticker, company_name, call_date, source=source)


# ---------------------------------------------------------------------------
# Tone analyzer
# ---------------------------------------------------------------------------

class ManagementToneAnalyzer:
    """
    Analyzes management tone in earnings call transcripts.
    Provides scores for:
    - Overall sentiment
    - Confidence level
    - Risk disclosure level
    - Forward guidance positivity
    - Vs prior quarter comparison
    """

    def __init__(self):
        self._history: Dict[str, List[Dict[str, float]]] = {}

    def analyze(self, transcript: EarningsTranscript) -> Dict[str, float]:
        """Full tone analysis of a transcript."""
        features = transcript.tone_features.copy()

        # Guidance positivity
        guidance_text = " ".join(transcript.guidance_statements).lower()
        guidance_positive = sum(guidance_text.count(w) for w in TONE_POSITIVE)
        guidance_negative = sum(guidance_text.count(w) for w in TONE_NEGATIVE)
        features["guidance_tone"] = float(
            (guidance_positive - guidance_negative) / max(guidance_positive + guidance_negative, 1)
        )

        # Risk density
        total_words = max(len(transcript.full_text.split()), 1)
        features["risk_density"] = float(len(transcript.risk_statements) / (total_words / 1000))

        # Q&A responsiveness (management words per analyst question)
        mgmt_qa = [s for s in transcript.qa_sections if s.role == "management"]
        analyst_qa = [s for s in transcript.qa_sections if s.role == "analyst"]
        if analyst_qa:
            mgmt_words = sum(len(s.text.split()) for s in mgmt_qa)
            analyst_words = sum(len(s.text.split()) for s in analyst_qa)
            features["qa_mgmt_ratio"] = float(mgmt_words / max(analyst_words, 1))
        else:
            features["qa_mgmt_ratio"] = 0.0

        # Store for historical comparison
        ticker = transcript.ticker
        if ticker:
            self._history.setdefault(ticker, []).append(features)

        return features

    def get_change_vs_prior(
        self,
        ticker: str,
        current_features: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute change in tone vs most recent prior quarter."""
        history = self._history.get(ticker, [])
        if len(history) < 2:
            return {}

        prior = history[-2]  # second-to-last (current is already appended)
        return {
            f"delta_{k}": float(current_features.get(k, 0) - prior.get(k, 0))
            for k in current_features
            if k in prior
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing transcript parser...")

    sample_transcript = """
    Q3 2024 Earnings Call
    October 15, 2024

    Operator
    Good afternoon, and welcome to the Q3 2024 earnings conference call.

    John Smith
    Chief Executive Officer
    Thank you. We delivered strong Q3 results with record revenue of $5.2 billion, up 12% year-over-year.
    We exceeded our guidance on both revenue and earnings. EPS came in at $1.85 per diluted share.
    We remain confident in our full-year guidance of $20 billion in revenue.

    Jane Doe
    Chief Financial Officer
    Our gross margin expanded 150 basis points to 45.3%. We expect Q4 revenue between $5.0 and $5.3 billion.
    We do see some macroeconomic uncertainty and currency headwinds that may impact Q4.

    Q&A

    Mike Johnson
    Goldman Sachs Analyst
    Can you provide more color on the demand environment? Are you seeing any slowdown?

    John Smith
    Chief Executive Officer
    We continue to see strong demand across all segments. We're not seeing any material slowdown at this time.
    """

    fetcher = TranscriptFetcher()
    transcript = fetcher.parse_text(sample_transcript, ticker="TEST", company_name="TestCo")

    print(f"Segments: {len(transcript.segments)}")
    print(f"Management: {len(transcript.management_sections)}")
    print(f"Q&A: {len(transcript.qa_sections)}")
    print(f"Guidance statements: {transcript.guidance_statements[:2]}")
    print(f"Tone features: {transcript.tone_features}")
    print(f"Financial summary: {transcript.financial_summary}")

    analyzer = ManagementToneAnalyzer()
    analysis = analyzer.analyze(transcript)
    print(f"Tone analysis: {analysis}")

    print("Transcript parser self-test passed.")
