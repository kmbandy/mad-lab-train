#!/usr/bin/env python3
"""Generate sentiment analyst fine-tune data from HuggingFace financial datasets.

Sources (downloaded automatically on first run):
  1. financial_phrasebank          — 4,846 financial news sentences (pos/neg/neutral)
  2. zeroshot/twitter-financial-news-sentiment — ticker-tagged financial tweets
  3. nickmuchi/financial-classification       — financial news multi-class
  4. Synthetic gap-fill                       — scenario templates to balance classes

Each source is mapped to our signal schema:
  DIRECTION: LONG | SHORT | HOLD
  CONVICTION: 0.0–1.0
  TIMEFRAME: string
  THESIS: string
  INVALIDATION: string

Outputs: data/sentiment_train.jsonl, data/sentiment_eval.jsonl
"""

import json
import random
import re
from pathlib import Path
from collections import Counter

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = (
    "You are a market sentiment analyst. Analyze news, narrative, and market psychology. "
    "Respond ONLY with a JSON signal: "
    '{"DIRECTION":"LONG|SHORT|HOLD","CONVICTION":0.0-1.0,'
    '"TIMEFRAME":"string","THESIS":"string","INVALIDATION":"string"}'
)

SECTORS = ["Technology", "Financial", "Healthcare", "Energy", "Consumer Cyclical",
           "Consumer Staples", "Communication", "Industrials", "Materials"]

SYMBOLS_BY_SECTOR = {
    "Technology":        ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "CRM", "ADBE", "ORCL"],
    "Financial":         ["JPM", "BAC", "GS", "MS", "WFC", "BLK", "C", "AXP"],
    "Healthcare":        ["JNJ", "PFE", "MRK", "ABBV", "UNH", "CVS", "LLY", "BMY"],
    "Energy":            ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "OXY"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "BKNG"],
    "Consumer Staples":  ["WMT", "TGT", "PG", "KO", "PEP", "COST", "CL"],
    "Communication":     ["DIS", "NFLX", "T", "VZ", "PARA", "SNAP", "SPOT"],
    "Industrials":       ["GE", "CAT", "DE", "HON", "UPS", "FDX", "LMT"],
    "Materials":         ["LIN", "APD", "NEM", "FCX", "ALB", "DD"],
}


def _random_sym():
    sector = random.choice(SECTORS)
    return random.choice(SYMBOLS_BY_SECTOR[sector]), sector


# ── Label helpers ──────────────────────────────────────────────────────────────

def _conviction_from_text(text: str, base_direction: str) -> float:
    """Estimate conviction from linguistic intensity markers."""
    strong_bull = ["surges", "soars", "beats", "upgrades", "raised", "record", "breakthrough",
                   "crush", "outperform", "accelerat", "expan", "buys back", "dividend hike"]
    strong_bear = ["plunges", "collapses", "misses", "cuts", "downgrade", "recall",
                   "lawsuit", "fraud", "probe", "investigate", "restructur", "layoff", "halt"]
    soft = ["slight", "modest", "mixed", "cautious", "uncertain", "concern", "watch", "monitor"]

    txt = text.lower()
    strong_hits = sum(1 for w in (strong_bull if base_direction == "LONG" else strong_bear) if w in txt)
    soft_hits = sum(1 for w in soft if w in txt)

    if strong_hits >= 2:
        return round(random.uniform(0.70, 0.85), 2)
    if strong_hits == 1 and soft_hits == 0:
        return round(random.uniform(0.58, 0.72), 2)
    if soft_hits >= 1:
        return round(random.uniform(0.40, 0.57), 2)
    return round(random.uniform(0.48, 0.65), 2)


def _timeframe(direction: str) -> str:
    if direction == "HOLD":
        return "1D"
    return random.choice(["1D", "2-3D", "1W"])


def _thesis_from_text(text: str, direction: str, sym: str, sector: str) -> str:
    text = text.strip().rstrip(".")
    if direction == "LONG":
        return (f"Positive sentiment for {sym} ({sector}): {text}. "
                f"Market psychology favoring upside — news-driven buying pressure expected.")
    if direction == "SHORT":
        return (f"Negative sentiment for {sym} ({sector}): {text}. "
                f"Bearish narrative building — institutional sellers likely stepping in.")
    return (f"Mixed/neutral sentiment for {sym} ({sector}): {text}. "
            f"No clear directional edge — waiting for cleaner signal.")


def _invalidation(direction: str, sector: str) -> str:
    if direction == "LONG":
        return (f"Sentiment reversal — negative follow-on news, sector-wide risk-off, "
                f"or {sector} macro headwind would invalidate bullish thesis.")
    if direction == "SHORT":
        return (f"Positive catalyst (earnings beat, analyst upgrade, sector rotation into "
                f"{sector}) would override bearish sentiment.")
    return f"Decisive macro or {sector}-specific catalyst breaks neutrality."


def _map_phrasebank(text: str, label: int) -> dict:
    """Map financial_phrasebank label (0=neg, 1=neu, 2=pos) to signal."""
    sym, sector = _random_sym()
    direction = {2: "LONG", 0: "SHORT", 1: "HOLD"}[label]
    conviction = _conviction_from_text(text, direction)
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human",  "value": (
                f"[TICKER: {sym}]\n[SECTOR: {sector}]\n"
                f"[HEADLINE: {text}]\nGenerate a sentiment signal."
            )},
            {"from": "gpt", "value": json.dumps({
                "DIRECTION": direction,
                "CONVICTION": conviction,
                "TIMEFRAME": _timeframe(direction),
                "THESIS": _thesis_from_text(text, direction, sym, sector),
                "INVALIDATION": _invalidation(direction, sector),
            })},
        ]
    }


def _map_twitter_fin(text: str, label: str) -> dict:
    """Map twitter-financial-news-sentiment label to signal."""
    # Labels: "Bearish", "Bullish", "Neutral"
    label_map = {"Bullish": "LONG", "Bearish": "SHORT", "Neutral": "HOLD"}
    direction = label_map.get(label, "HOLD")

    # Try to extract a ticker from the tweet ($AAPL style)
    tickers = re.findall(r'\$([A-Z]{1,5})', text)
    if tickers:
        sym = tickers[0]
        # Find sector
        sector = "Technology"
        for sec, syms in SYMBOLS_BY_SECTOR.items():
            if sym in syms:
                sector = sec
                break
    else:
        sym, sector = _random_sym()

    # Clean tweet text
    clean = re.sub(r'\$[A-Z]{1,5}', '', text).strip()
    clean = re.sub(r'https?://\S+', '', clean).strip()
    clean = re.sub(r'[#@]\w+', '', clean).strip()
    clean = re.sub(r'\s+', ' ', clean).strip()
    if len(clean) < 15:
        clean = text[:120]

    conviction = _conviction_from_text(clean, direction)
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human",  "value": (
                f"[TICKER: {sym}]\n[SECTOR: {sector}]\n"
                f"[HEADLINE: {clean}]\nGenerate a sentiment signal."
            )},
            {"from": "gpt", "value": json.dumps({
                "DIRECTION": direction,
                "CONVICTION": conviction,
                "TIMEFRAME": _timeframe(direction),
                "THESIS": _thesis_from_text(clean, direction, sym, sector),
                "INVALIDATION": _invalidation(direction, sector),
            })},
        ]
    }


def _map_nickmuchi(text: str, label: int) -> dict:
    """Map nickmuchi/financial-classification labels to signal.
    Labels: 0=negative, 1=positive, 2=neutral
    """
    direction = {1: "LONG", 0: "SHORT", 2: "HOLD"}.get(label, "HOLD")
    sym, sector = _random_sym()
    conviction = _conviction_from_text(text, direction)
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human",  "value": (
                f"[TICKER: {sym}]\n[SECTOR: {sector}]\n"
                f"[HEADLINE: {text[:200]}]\nGenerate a sentiment signal."
            )},
            {"from": "gpt", "value": json.dumps({
                "DIRECTION": direction,
                "CONVICTION": conviction,
                "TIMEFRAME": _timeframe(direction),
                "THESIS": _thesis_from_text(text[:200], direction, sym, sector),
                "INVALIDATION": _invalidation(direction, sector),
            })},
        ]
    }


# ── Synthetic gap-fill (keeps HOLD class balanced) ─────────────────────────────

_HOLD_SCENARIOS = [
    "Sector ETF showing mixed flows — institutional positioning unclear ahead of FOMC.",
    "Analyst coverage initiated with Neutral rating; price target in-line with current price.",
    "Trading volumes near 30-day average; no unusual options activity detected.",
    "Company reports in-line quarter — beat on EPS, miss on revenue, guidance unchanged.",
    "Short interest stable at 4.2%; no change in institutional ownership last quarter.",
    "Macro data mixed: strong jobs but weak consumer spending — net neutral for equities.",
    "Sector rotation data inconclusive — outflows from growth offset by value inflows.",
    "Earnings preannouncement: guidance narrowed to midpoint of prior range.",
    "Management change announced — new CEO from within; continuity expected.",
    "Patent litigation settled for undisclosed amount — overhang removed but no windfall.",
]


def _make_hold_sample() -> dict:
    sym, sector = _random_sym()
    text = random.choice(_HOLD_SCENARIOS)
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human",  "value": (
                f"[TICKER: {sym}]\n[SECTOR: {sector}]\n"
                f"[HEADLINE: {text}]\nGenerate a sentiment signal."
            )},
            {"from": "gpt", "value": json.dumps({
                "DIRECTION": "HOLD",
                "CONVICTION": round(random.uniform(0.30, 0.50), 2),
                "TIMEFRAME": "1D",
                "THESIS": (f"No clear directional edge for {sym} ({sector}). {text} "
                           f"Sentiment neutral — monitoring for catalyst before taking a position."),
                "INVALIDATION": (f"Decisive sector or macro catalyst, unusual options activity, "
                                 f"or vol expansion would prompt reassessment."),
            })},
        ]
    }


_SHORT_SCENARIOS = [
    "Company reports net loss, revenue declined {p}% YoY — guidance cut.",
    "SEC investigation opened into accounting irregularities.",
    "CFO departure announced amid restatement of prior financials.",
    "Going concern language added to annual filing.",
    "Debt covenant breach disclosed — restructuring negotiations begin.",
    "Material impairment charge of ${c}M recognized on goodwill.",
    "Quarterly miss: EPS ${e} vs ${g} est, revenue {p}% below consensus.",
    "Mass layoffs announced — {p}% workforce reduction, restructuring charges expected.",
    "Credit rating downgraded to junk — borrowing costs rise.",
    "Key customer representing {p}% of revenue terminates contract.",
    "Product recall issued — liability exposure estimated at ${c}M.",
    "Regulatory fine of ${c}M imposed; further enforcement risk cited.",
    "Activist investor demands breakup; board rejects — stock drops.",
    "CEO arrested on fraud charges — immediate resignation.",
    "Supply chain disruption forces plant shutdown for {p} weeks.",
]


def _make_short_sample() -> dict:
    sym, sector = _random_sym()
    tmpl = random.choice(_SHORT_SCENARIOS)
    text = tmpl.format(
        p=random.randint(5, 35),
        c=random.randint(50, 900),
        e=round(random.uniform(-0.5, 0.8), 2),
        g=round(random.uniform(0.9, 2.5), 2),
    )
    conviction = round(random.uniform(0.58, 0.80), 2)
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human",  "value": (
                f"[TICKER: {sym}]\n[SECTOR: {sector}]\n"
                f"[HEADLINE: {text}]\nGenerate a sentiment signal."
            )},
            {"from": "gpt", "value": json.dumps({
                "DIRECTION": "SHORT",
                "CONVICTION": conviction,
                "TIMEFRAME": _timeframe("SHORT"),
                "THESIS": _thesis_from_text(text, "SHORT", sym, sector),
                "INVALIDATION": _invalidation("SHORT", sector),
            })},
        ]
    }


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_phrasebank() -> list:
    try:
        from datasets import load_dataset
        ds = load_dataset("financial_phrasebank", "sentences_allagree", trust_remote_code=True)
        samples = []
        for row in ds["train"]:
            samples.append(_map_phrasebank(row["sentence"], row["label"]))
        print(f"  financial_phrasebank: {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"  financial_phrasebank SKIP: {e}")
        return []


def load_twitter_fin() -> list:
    try:
        from datasets import load_dataset
        ds = load_dataset("zeroshot/twitter-financial-news-sentiment", trust_remote_code=True)
        samples = []
        for split in ds.values():
            for row in split:
                try:
                    samples.append(_map_twitter_fin(row["text"], row["label"]))
                except Exception:
                    pass
        print(f"  twitter-financial-news-sentiment: {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"  twitter-financial-news-sentiment SKIP: {e}")
        return []


def load_nickmuchi() -> list:
    try:
        from datasets import load_dataset
        ds = load_dataset("nickmuchi/financial-classification", trust_remote_code=True)
        samples = []
        for split in ds.values():
            for row in split:
                try:
                    # column names may vary
                    text = row.get("text") or row.get("sentence") or row.get("news") or ""
                    label = row.get("label") or row.get("labels") or 2
                    if text:
                        samples.append(_map_nickmuchi(text, int(label)))
                except Exception:
                    pass
        print(f"  nickmuchi/financial-classification: {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"  nickmuchi/financial-classification SKIP: {e}")
        return []


_SIC_TO_SECTOR = {
    "01": "Consumer Staples",     # Agriculture
    "10": "Materials",            # Mining
    "13": "Energy",               # Oil & Gas
    "15": "Industrials",          # Construction
    "20": "Consumer Staples",     # Food
    "28": "Materials",            # Chemicals
    "35": "Technology",           # Industrial Machinery
    "36": "Technology",           # Electronic Equipment
    "37": "Consumer Cyclical",    # Motor Vehicles
    "38": "Technology",           # Instruments
    "48": "Communication",        # Telecom
    "49": "Utilities",
    "50": "Industrials",          # Wholesale
    "51": "Industrials",
    "52": "Consumer Cyclical",    # Retail
    "53": "Consumer Staples",
    "54": "Consumer Staples",
    "57": "Consumer Cyclical",
    "58": "Consumer Cyclical",    # Eating/Drinking
    "59": "Consumer Staples",
    "60": "Financial",            # Banking
    "61": "Financial",
    "62": "Financial",
    "63": "Financial",            # Insurance
    "64": "Financial",
    "65": "Financial",            # Real Estate
    "67": "Financial",
    "70": "Consumer Cyclical",    # Hotels
    "73": "Technology",           # Computer Services
    "78": "Communication",        # Entertainment
    "79": "Communication",
    "80": "Healthcare",
    "82": "Education",
    "83": "Healthcare",
    "84": "Communication",
    "87": "Technology",           # Engineering Services
}

_10K_HEADLINE_TEMPLATES = {
    "LONG": [
        "{company} ({sym}) reports record annual revenue, exceeds analyst consensus.",
        "{company} ({sym}) 10-K: net income up {pct}% YoY, operating margins expand to {margin}%.",
        "{company} ({sym}) annual filing highlights strong free cash flow and raised dividend.",
        "{company} ({sym}) 10-K: market share gains, geographic expansion on track.",
        "{company} ({sym}) reports fourth consecutive year of double-digit revenue growth.",
        "{company} ({sym}) 10-K: balance sheet strengthened, debt reduction ahead of schedule.",
        "{company} ({sym}) annual report: new product pipeline exceeds development milestones.",
    ],
    "SHORT": [
        "{company} ({sym}) 10-K discloses material weakness in internal controls.",
        "{company} ({sym}) annual filing: revenue declined {pct}% YoY, restructuring charges taken.",
        "{company} ({sym}) 10-K: going concern language added, covenant compliance at risk.",
        "{company} ({sym}) annual report: impairment charges of ${charges}M, margins compressed.",
        "{company} ({sym}) 10-K: lost key customer representing {pct}% of revenue.",
        "{company} ({sym}) discloses SEC investigation, restatement of prior financials.",
        "{company} ({sym}) 10-K: net loss widens, liquidity runway less than 12 months.",
    ],
    "HOLD": [
        "{company} ({sym}) 10-K: results in-line, guidance reiterated at midpoint.",
        "{company} ({sym}) annual filing: mixed quarter — beat EPS, missed revenue by 1.2%.",
        "{company} ({sym}) 10-K: leadership transition underway, strategic review ongoing.",
        "{company} ({sym}) annual report: capex increased, near-term margin pressure expected.",
        "{company} ({sym}) 10-K: regulatory approval pending, timeline uncertain.",
        "{company} ({sym}) files 10-K with no material changes to previously disclosed guidance.",
    ],
}


def load_edgar_10k(sample_size: int = 3000) -> list:
    """Parse alea-institute/kl3m-index-edgar-filings-10-k metadata index.

    This dataset is an index file (S3 paths + metadata) — actual document text
    lives in S3/EDGAR, not in the HF dataset itself. We use the rich metadata
    (ticker, company name, SIC code, market cap size, filing date) to build
    realistic company-specific synthetic 10-K sentiment scenarios.

    Fields (tab-separated in the index):
      s3_path, cik, company_name, filing_date, form_type, accession,
      ..., ticker, exchange, sic_code, industry_desc, size, ..., fiscal_year_end
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "alea-institute/kl3m-index-edgar-filings-10-k",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        seen, samples = 0, []
        step = max(1, 600000 // (sample_size * 3))

        for row in ds:
            seen += 1
            if seen % step != 0:
                continue
            try:
                # Actual columns: kl3m_id, cik, name, filingDate, tickers,
                #   exchanges, sic, sicDescription, category, ...
                if not isinstance(row, dict):
                    continue
                ticker   = str(row.get("tickers") or "").strip()
                company  = str(row.get("name") or "").strip()
                sic_str  = str(row.get("sic") or "").strip()
                size_cat = str(row.get("category") or "").strip()

                # Derive sector from SIC prefix
                sic_prefix = sic_str[:2] if sic_str else ""
                sector = _SIC_TO_SECTOR.get(sic_prefix, "")
                if not sector:
                    _, sector = _random_sym()

                # Fall back to random sym if ticker not useful
                if not ticker or len(ticker) > 5 or not ticker.replace(".", "").isalpha():
                    sym, sector = _random_sym()
                    company = company or sym
                else:
                    sym = ticker.upper()

                company = company[:40] if company else sym

                # Build a realistic 10-K headline
                direction = random.choice(["LONG", "LONG", "SHORT", "SHORT", "HOLD"])
                tmpl = random.choice(_10K_HEADLINE_TEMPLATES[direction])
                pct = random.randint(8, 32)
                margin = random.randint(12, 38)
                charges = random.randint(50, 800)
                headline = tmpl.format(
                    company=company, sym=sym, pct=pct, margin=margin, charges=charges
                )

                conviction = _conviction_from_text(headline, direction)
                if size_cat in ("Large Accelerated", "Accelerated"):
                    conviction = min(1.0, conviction + 0.05)  # higher conviction for large caps

                samples.append({
                    "conversations": [
                        {"from": "system", "value": SYSTEM_PROMPT},
                        {"from": "human",  "value": (
                            f"[TICKER: {sym}]\n[SECTOR: {sector}]\n"
                            f"[HEADLINE: {headline}]\nGenerate a sentiment signal."
                        )},
                        {"from": "gpt", "value": json.dumps({
                            "DIRECTION": direction,
                            "CONVICTION": round(conviction, 2),
                            "TIMEFRAME": _timeframe(direction),
                            "THESIS": _thesis_from_text(headline, direction, sym, sector),
                            "INVALIDATION": _invalidation(direction, sector),
                        })},
                    ]
                })
            except Exception:
                pass

            if len(samples) >= sample_size:
                break

        print(f"  kl3m edgar-10k (metadata): {len(samples)} samples (streamed {seen} rows)")
        return samples
    except Exception as e:
        print(f"  kl3m edgar-10k SKIP: {e}")
        return []


def load_market_risk_filings(sample_size: int = 3000) -> list:
    """Load DerivedFunction/sec-filings-market-risk-and-derivatives (352K rows).

    Passages are market risk / derivatives disclosures from SEC filings.
    Text is inherently cautious (risk language) → mostly HOLD/SHORT.
    Streams and samples to avoid loading 352K into memory.

    Columns: text (str, 300-1.26k chars), len (int), __index_level_0__ (int)
    """
    # Risk/derivatives-specific keyword sets
    hedge_pos = ["effectively hedged", "mitigates risk", "protected against",
                 "reduced exposure", "interest rate cap", "collar agreement",
                 "fixed rate", "locked in", "favorable terms"]
    risk_neg  = ["unhedged", "significant exposure", "adverse", "losses",
                 "default", "counterparty risk", "exceeds", "volatility",
                 "speculative", "leveraged", "margin call", "covenant breach",
                 "impairment", "write-down", "write-off"]
    risk_neut = ["notional amount", "fair value", "swap agreement", "derivative",
                 "interest rate", "foreign currency", "commodity price",
                 "market risk", "sensitivity analysis"]

    def _score(text: str) -> tuple:
        tl = text.lower()
        pos  = sum(1 for k in hedge_pos if k in tl)
        neg  = sum(1 for k in risk_neg  if k in tl)
        neut = sum(1 for k in risk_neut if k in tl)
        # Risk disclosures are almost never strong bullish signals
        if neg > pos + 1 and neg > 1:
            return "SHORT", round(random.uniform(0.45, 0.62), 2)
        if pos > neg and pos > 1:
            return "HOLD", round(random.uniform(0.40, 0.52), 2)  # hedged = neutral, not LONG
        # Purely procedural risk disclosure → HOLD
        return "HOLD", round(random.uniform(0.30, 0.45), 2)

    try:
        from datasets import load_dataset
        ds = load_dataset(
            "DerivedFunction/sec-filings-market-risk-and-derivatives",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        samples, seen = [], 0
        step = max(1, 352000 // (sample_size * 3))

        for row in ds:
            seen += 1
            if seen % step != 0:
                continue
            try:
                text = str(row.get("text") or "").strip()
                if len(text) < 80:
                    continue
                snippet = text[:350].replace("\n", " ").strip()
                direction, conviction = _score(snippet)
                sym, sector = _random_sym()
                samples.append({
                    "conversations": [
                        {"from": "system", "value": SYSTEM_PROMPT},
                        {"from": "human",  "value": (
                            f"[TICKER: {sym}]\n[SECTOR: {sector}]\n"
                            f"[HEADLINE: {snippet}]\nGenerate a sentiment signal."
                        )},
                        {"from": "gpt", "value": json.dumps({
                            "DIRECTION": direction,
                            "CONVICTION": conviction,
                            "TIMEFRAME": _timeframe(direction),
                            "THESIS": _thesis_from_text(snippet[:200], direction, sym, sector),
                            "INVALIDATION": _invalidation(direction, sector),
                        })},
                    ]
                })
            except Exception:
                pass
            if len(samples) >= sample_size:
                break

        print(f"  sec-filings-market-risk: {len(samples)} samples (streamed {seen} rows)")
        return samples
    except Exception as e:
        print(f"  sec-filings-market-risk SKIP: {e}")
        return []


def load_filings_rag() -> list:
    """Load emirMb/filings_rag_evaluation_bge-m3_llama-3.3-8b-instruct (~225 rows).

    RAG evaluation dataset over real SEC filings. Uses retrieved_docs passages
    as grounded filing text. Only uses high-quality rows (eval_score >= 3).

    Columns: question, correct_answer, generated_answer, retrieved_docs,
             eval_score_llama_3.3_8b_instruct:free, eval_feedback_...
    """
    pos_kw = ["record", "growth", "increased", "exceeded", "strong", "beat", "raised",
              "opportunity", "expansion", "market share", "performance-based", "reward",
              "approved", "adoption", "invested", "surpassed"]
    neg_kw = ["risk", "decline", "loss", "impairment", "restructur", "investigation",
              "uncertainty", "material weakness", "bankruptcy", "default", "departure",
              "terminated", "failed", "shortfall", "adverse"]

    def _score_passage(text: str) -> tuple:
        tl = text.lower()
        pos = sum(1 for k in pos_kw if k in tl)
        neg = sum(1 for k in neg_kw if k in tl)
        if pos > neg + 1:
            return "LONG", round(random.uniform(0.58, 0.74), 2)
        if neg > pos + 1:
            return "SHORT", round(random.uniform(0.55, 0.70), 2)
        return "HOLD", round(random.uniform(0.35, 0.50), 2)

    try:
        from datasets import load_dataset
        ds = load_dataset(
            "emirMb/filings_rag_evaluation_bge-m3_llama-3.3-8b-instruct",
            trust_remote_code=True,
        )
        samples = []
        score_col = "eval_score_llama_3.3_8b_instruct:free"

        for split in ds.values():
            for row in split:
                try:
                    # Only use high-quality rows
                    score = int(row.get(score_col) or 0)
                    if score < 3:
                        continue

                    question    = str(row.get("question") or "").strip()
                    correct_ans = str(row.get("correct_answer") or "").strip()
                    docs        = row.get("retrieved_docs") or []

                    # docs may be a list of strings or a single string
                    if isinstance(docs, str):
                        passages = [docs]
                    elif isinstance(docs, list):
                        passages = [str(d) for d in docs if d]
                    else:
                        passages = []

                    if not passages:
                        continue

                    # Score the first (most relevant) passage
                    primary = passages[0][:400].replace("\n", " ").strip()
                    direction, conviction = _score_passage(primary)

                    # Build a rich headline from question + answer summary
                    if question and correct_ans:
                        headline = f"Q: {question[:120]} — {correct_ans[:180]}"
                    elif question:
                        headline = question[:250]
                    else:
                        headline = primary[:250]

                    sym, sector = _random_sym()
                    # Try to infer ticker from filing text
                    ticker_match = re.search(
                        r'\b([A-Z]{1,5})\s+(?:Corporation|Corp\.|Inc\.|Ltd\.|Co\.)',
                        primary
                    )
                    if ticker_match:
                        candidate = ticker_match.group(1)
                        if candidate not in {"THE", "FOR", "AND", "SEC", "CEO", "CFO", "COO"}:
                            sym = candidate

                    samples.append({
                        "conversations": [
                            {"from": "system", "value": SYSTEM_PROMPT},
                            {"from": "human",  "value": (
                                f"[TICKER: {sym}]\n[SECTOR: {sector}]\n"
                                f"[HEADLINE: {headline}]\nGenerate a sentiment signal."
                            )},
                            {"from": "gpt", "value": json.dumps({
                                "DIRECTION": direction,
                                "CONVICTION": conviction,
                                "TIMEFRAME": _timeframe(direction),
                                "THESIS": _thesis_from_text(headline[:200], direction, sym, sector),
                                "INVALIDATION": _invalidation(direction, sector),
                            })},
                        ]
                    })
                except Exception:
                    pass

        print(f"  emirMb/filings_rag_evaluation: {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"  filings_rag_evaluation SKIP: {e}")
        return []


def load_8k_filings() -> list:
    """Load deazzahra/8k_extracted_filings — 8-K SEC filings with text + summary.

    8-K item numbers carry strong directional signal:
      Bullish items  → LONG  (results beats, acquisitions, share buybacks)
      Bearish items  → SHORT (impairments, restructuring, going concern, delisting)
      Neutral items  → HOLD  (routine disclosures, governance changes)

    Uses the summary column (when available) as the headline + the item text
    as supporting context.
    """
    # Map 8-K item prefixes → direction bias
    # Higher value = stronger signal; items not listed → HOLD
    _ITEM_SIGNAL = {
        # Strongly bullish
        "2.02": ("LONG",  0.72),   # Results of Operations (beat)
        "2.01": ("LONG",  0.68),   # Completion of Acquisition
        "8.01": ("LONG",  0.55),   # Other Events (often positive announcements)
        "1.01": ("LONG",  0.60),   # Entry into Material Agreement
        # Strongly bearish
        "1.03": ("SHORT", 0.80),   # Bankruptcy / Receivership
        "2.04": ("SHORT", 0.75),   # Triggering Events (debt defaults)
        "2.05": ("SHORT", 0.68),   # Costs — Exit/Disposal (restructuring)
        "2.06": ("SHORT", 0.72),   # Material Impairments
        "3.01": ("SHORT", 0.78),   # Notice of Delisting
        "4.01": ("SHORT", 0.65),   # Change in Accountant (auditor concern)
        "1.02": ("SHORT", 0.58),   # Termination of Agreement
        # Neutral / mixed
        "5.02": ("HOLD",  0.42),   # Director/Officer changes
        "5.03": ("HOLD",  0.38),   # Amendments to Articles
        "7.01": ("HOLD",  0.45),   # Regulation FD Disclosure
        "9.01": ("HOLD",  0.35),   # Financial Statements (exhibits)
        "5.01": ("HOLD",  0.48),   # Change in Control
    }

    def _direction_from_item(item_code: str, text: str) -> tuple:
        """Determine direction from item code + keyword cross-check."""
        base_dir, base_conv = _ITEM_SIGNAL.get(item_code, ("HOLD", 0.40))

        # For results items (2.02), keyword-check the text to flip direction on misses
        if item_code == "2.02":
            tl = text.lower()
            miss_kw = ["declined", "decreased", "missed", "below", "loss", "shortfall",
                       "lower than", "fell short", "miss"]
            beat_kw = ["exceeded", "surpassed", "grew", "increased", "record", "beat",
                       "above expectations", "strong growth"]
            misses = sum(1 for k in miss_kw if k in tl)
            beats  = sum(1 for k in beat_kw if k in tl)
            if misses > beats:
                base_dir, base_conv = "SHORT", 0.65
            elif beats >= misses:
                base_dir, base_conv = "LONG", 0.70

        # Jitter conviction slightly
        conv = round(min(0.95, max(0.30, base_conv + random.uniform(-0.05, 0.05))), 2)
        return base_dir, conv

    try:
        from datasets import load_dataset
        ds = load_dataset("deazzahra/8k_extracted_filings", trust_remote_code=True)

        samples = []
        for split in ds.values():
            for row in split:
                try:
                    # Actual columns: input (filing text), summary_en (summary)
                    text    = (row.get("input") or row.get("text") or
                               row.get("filing_text") or row.get("content") or "")
                    summary = (row.get("summary_en") or row.get("summary") or
                               row.get("description") or "")
                    ticker  = (row.get("ticker") or row.get("symbol") or "").strip().upper()
                    company = (row.get("company") or row.get("company_name") or "").strip()

                    # Extract item code from text header "--- ITEM_X.XX ---"
                    item_match = re.search(r'ITEM[_\s]+(\d+\.\d+)', text, re.IGNORECASE)
                    item_code = item_match.group(1) if item_match else ""

                    # Use summary as headline if available, else first 300 chars of text
                    headline = summary.strip() if summary and len(summary) > 20 else ""
                    if not headline:
                        # Strip item header line then take first meaningful sentence
                        body = re.sub(r'---.*?---', '', text).strip()
                        headline = body[:300].replace("\n", " ").strip()

                    if not headline or len(headline) < 20:
                        continue

                    # Resolve ticker / sector
                    if ticker and 1 <= len(ticker) <= 5 and ticker.isalpha():
                        sym = ticker
                        sector = "Unknown"
                        for sec, syms in SYMBOLS_BY_SECTOR.items():
                            if sym in syms:
                                sector = sec
                                break
                        if sector == "Unknown":
                            _, sector = _random_sym()
                    else:
                        sym, sector = _random_sym()
                        if company:
                            # Use company name in headline even if ticker unknown
                            headline = headline.replace("the company", company)

                    direction, conviction = _direction_from_item(item_code, text + " " + headline)

                    samples.append({
                        "conversations": [
                            {"from": "system", "value": SYSTEM_PROMPT},
                            {"from": "human",  "value": (
                                f"[TICKER: {sym}]\n[SECTOR: {sector}]\n"
                                f"[HEADLINE: {headline[:300]}]\nGenerate a sentiment signal."
                            )},
                            {"from": "gpt", "value": json.dumps({
                                "DIRECTION": direction,
                                "CONVICTION": conviction,
                                "TIMEFRAME": _timeframe(direction),
                                "THESIS": _thesis_from_text(headline[:200], direction, sym, sector),
                                "INVALIDATION": _invalidation(direction, sector),
                            })},
                        ]
                    })
                except Exception:
                    pass

        print(f"  deazzahra/8k_extracted_filings: {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"  8k_extracted_filings SKIP: {e}")
        return []


def load_fiqa() -> list:
    """FiQA Sentiment dataset — financial Q&A with aspect-level sentiment."""
    try:
        from datasets import load_dataset
        ds = load_dataset("pauri32/fiqa-2018", trust_remote_code=True)
        samples = []
        for split in ds.values():
            for row in split:
                try:
                    text = row.get("sentence") or row.get("text") or ""
                    score = float(row.get("sentiment_score", 0))
                    if score > 0.1:
                        direction, label = "LONG", 2
                    elif score < -0.1:
                        direction, label = "SHORT", 0
                    else:
                        direction, label = "HOLD", 1
                    if text:
                        samples.append(_map_phrasebank(text[:200], label))
                except Exception:
                    pass
        print(f"  fiqa-2018: {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"  fiqa-2018 SKIP: {e}")
        return []


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading HuggingFace datasets...")
    samples = []
    samples += load_market_risk_filings(sample_size=3000)  # 352K SEC market-risk passages
    samples += load_filings_rag()                  # RAG eval over SEC filings (~225, high quality)
    samples += load_8k_filings()                   # real 8-K SEC filings (~560, high quality)
    samples += load_edgar_10k(sample_size=4000)   # 10-K index metadata → synthetic scenarios
    samples += load_phrasebank()                   # 4,846 labeled sentences
    samples += load_twitter_fin()                  # ticker-tagged tweets
    samples += load_nickmuchi()                    # financial news
    samples += load_fiqa()                         # aspect-level Q&A

    if not samples:
        print("WARNING: No HF datasets loaded. Running pure synthetic fallback.")

    # Count directions
    counts = Counter(
        json.loads(s["conversations"][2]["value"])["DIRECTION"]
        for s in samples
    )
    print(f"  After HF load: {counts}")

    # Pure LONG/SHORT — no HOLD. 50/50 split, 5000 total.
    # Sentiment model is directional-only; HOLD emerges from orchestrator disagreement.
    N_LONG = N_SHORT = 2500
    by_cls: dict = {"LONG": [], "SHORT": [], "HOLD": []}
    for s in samples:
        d = json.loads(s["conversations"][2]["value"])["DIRECTION"]
        by_cls[d].append(s)

    while len(by_cls["SHORT"]) < N_SHORT:
        by_cls["SHORT"].append(_make_short_sample())
    while len(by_cls["LONG"]) < N_LONG:
        by_cls["LONG"].append(random.choice(by_cls["LONG"]))

    samples = (
        random.sample(by_cls["LONG"],  N_LONG) +
        random.sample(by_cls["SHORT"], N_SHORT)
    )
    counts = Counter(
        json.loads(s["conversations"][2]["value"])["DIRECTION"]
        for s in samples
    )
    print(f"  Final: {len(samples)} samples: {dict(counts)}")

    random.shuffle(samples)
    split = int(len(samples) * 0.9)
    train, eval_ = samples[:split], samples[split:]

    (OUT_DIR / "sentiment_train.jsonl").write_text("\n".join(json.dumps(s) for s in train))
    (OUT_DIR / "sentiment_eval.jsonl").write_text("\n".join(json.dumps(s) for s in eval_))
    print(f"Sentiment: {len(train)} train / {len(eval_)} eval → {OUT_DIR}")

    dirs = [s["conversations"][2]["value"] for s in samples]
    for k, v in sorted(Counter(json.loads(d)["DIRECTION"] for d in dirs).items()):
        print(f"  {k}: {v} ({v/len(dirs):.1%})")


if __name__ == "__main__":
    main()
