#!/usr/bin/env python3
"""Generate technical analyst fine-tune data from TimescaleDB price/indicator history.

Outputs: technical_train.jsonl, technical_eval.jsonl
"""

import json
import os
import random
import psycopg2
from pathlib import Path

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = (
    "You are a technical analyst. Analyze price action and indicators. "
    "Respond ONLY with a JSON signal: "
    '{"DIRECTION":"LONG|SHORT|HOLD","CONVICTION":0.0-1.0,'
    '"TIMEFRAME":"string","THESIS":"string","INVALIDATION":"string"}'
)

# ── Rule-based labeling ────────────────────────────────────────────────────────

def label(rsi, macd, ema20, ema50, close):
    """Derive signal direction + conviction from indicator values."""
    bull_ema = close > ema20 > ema50
    bear_ema = close < ema20 < ema50

    # Strong buy
    if rsi < 32 and macd > 0:
        return "LONG", round(random.uniform(0.72, 0.88), 2)
    # Moderate buy
    if rsi < 42 and bull_ema:
        return "LONG", round(random.uniform(0.58, 0.72), 2)
    # Bullish trend continuation
    if 42 <= rsi <= 58 and bull_ema and macd > 0:
        return "LONG", round(random.uniform(0.55, 0.68), 2)
    # Strong sell
    if rsi > 68 and macd < 0:
        return "SHORT", round(random.uniform(0.72, 0.88), 2)
    # Moderate sell
    if rsi > 58 and bear_ema:
        return "SHORT", round(random.uniform(0.58, 0.72), 2)
    # Bearish trend continuation
    if 42 <= rsi <= 58 and bear_ema and macd < 0:
        return "SHORT", round(random.uniform(0.55, 0.68), 2)
    # Oversold but still falling
    if rsi < 32 and macd < 0:
        return "HOLD", round(random.uniform(0.40, 0.55), 2)
    # Overbought but still rising
    if rsi > 68 and macd > 0:
        return "HOLD", round(random.uniform(0.40, 0.55), 2)
    # Neutral
    return "HOLD", round(random.uniform(0.30, 0.50), 2)


def thesis(direction, symbol, rsi, macd, ema20, ema50, close):
    if direction == "LONG":
        if rsi < 32:
            return (f"{symbol} RSI at {rsi:.1f} signals oversold conditions. "
                    f"MACD {'confirming bullish momentum' if macd > 0 else 'yet to confirm'}. "
                    f"Price at {close:.2f} vs EMA20 {ema20:.2f} — watching for reversal.")
        return (f"{symbol} in bullish trend: price {close:.2f} above EMA20 {ema20:.2f} and EMA50 {ema50:.2f}. "
                f"RSI {rsi:.1f} with MACD {macd:.3f} supporting continuation.")
    if direction == "SHORT":
        if rsi > 68:
            return (f"{symbol} RSI at {rsi:.1f} signals overbought conditions. "
                    f"MACD {'confirming bearish reversal' if macd < 0 else 'diverging'}. "
                    f"Price at {close:.2f} extended above EMAs.")
        return (f"{symbol} in bearish trend: price {close:.2f} below EMA20 {ema20:.2f} and EMA50 {ema50:.2f}. "
                f"RSI {rsi:.1f} with MACD {macd:.3f} confirming downside.")
    return (f"{symbol} showing mixed signals. RSI {rsi:.1f}, MACD {macd:.3f}. "
            f"Price {close:.2f} near EMA20 {ema20:.2f}. No clear directional bias.")


def invalidation(direction, ema20, ema50, close):
    if direction == "LONG":
        stop = round(min(ema20, ema50) * 0.985, 2)
        return f"Close below {stop:.2f} (below key EMAs) invalidates bullish thesis."
    if direction == "SHORT":
        stop = round(max(ema20, ema50) * 1.015, 2)
        return f"Close above {stop:.2f} (above key EMAs) invalidates bearish thesis."
    return "Break above EMA20 or below recent support would trigger directional re-evaluation."


def timeframe(direction):
    if direction == "HOLD":
        return "1D"
    return random.choice(["1D", "2-3D", "1W"])


# ── Main ───────────────────────────────────────────────────────────────────────

def build_user_prompt(row):
    return (
        f"[TICKER: {row['symbol']}]\n"
        f"[OHLCV: open={row['open']:.2f} high={row['high']:.2f} "
        f"low={row['low']:.2f} close={row['close']:.2f} vol={int(row['volume'])} ts={row['time']}]\n"
        f"[INDICATORS: RSI={row['rsi']:.2f} MACD={row['macd']:.4f} "
        f"EMA20={row['ema20']:.2f} EMA50={row['ema50']:.2f}]\n"
        f"Generate a technical signal."
    )


def main():
    conn = psycopg2.connect(
        host=os.getenv("TSDB_HOST", "localhost"),
        port=os.getenv("TSDB_PORT", "5432"),
        dbname=os.getenv("TSDB_NAME", "quantdb"),
        user=os.getenv("TSDB_USER", "quantuser"),
        password=os.getenv("TSDB_PASS", "quantpass"),
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT p.symbol, p.time, p.open, p.high, p.low, p.close, p.volume,
               i.rsi, i.macd, i.ema_20 AS ema20, i.ema_50 AS ema50
        FROM price_history p
        JOIN indicators i ON p.symbol = i.ticker AND p.time = i.time
        WHERE i.rsi IS NOT NULL AND i.macd IS NOT NULL
        ORDER BY p.symbol, p.time
    """)
    cols = [d[0] for d in cur.description]
    _float_cols = {"rsi", "macd", "ema20", "ema50", "open", "high", "low", "close"}
    rows = [
        {k: (float(v) if k in _float_cols and v is not None else v) for k, v in zip(cols, r)}
        for r in cur.fetchall()
    ]
    conn.close()
    print(f"Loaded {len(rows)} rows from DB")

    samples = []
    for row in rows:
        direction, conviction = label(row['rsi'], row['macd'], row['ema20'], row['ema50'], row['close'])
        signal = {
            "DIRECTION": direction,
            "CONVICTION": conviction,
            "TIMEFRAME": timeframe(direction),
            "THESIS": thesis(direction, row['symbol'], row['rsi'], row['macd'],
                             row['ema20'], row['ema50'], row['close']),
            "INVALIDATION": invalidation(direction, row['ema20'], row['ema50'], row['close']),
        }
        samples.append({
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": build_user_prompt(row)},
                {"from": "gpt", "value": json.dumps(signal)},
            ]
        })

    # Balance classes: cap HOLD at 2× the minority class count
    from collections import defaultdict
    by_dir = defaultdict(list)
    for s in samples:
        d = json.loads(s['conversations'][2]['value'])['DIRECTION']
        by_dir[d].append(s)
    minority = min(len(by_dir['LONG']), len(by_dir['SHORT']))
    hold_cap = minority * 2
    balanced = (
        by_dir['LONG'] +
        by_dir['SHORT'] +
        random.sample(by_dir['HOLD'], min(hold_cap, len(by_dir['HOLD'])))
    )
    random.shuffle(balanced)
    samples = balanced

    split = int(len(samples) * 0.9)
    train, eval_ = samples[:split], samples[split:]

    (OUT_DIR / "technical_train.jsonl").write_text("\n".join(json.dumps(s) for s in train))
    (OUT_DIR / "technical_eval.jsonl").write_text("\n".join(json.dumps(s) for s in eval_))
    print(f"Technical: {len(train)} train / {len(eval_)} eval → {OUT_DIR}")

    # Print label distribution
    dirs = [s['conversations'][2]['value'] for s in samples]
    from collections import Counter
    for k, v in sorted(Counter(json.loads(d)['DIRECTION'] for d in dirs).items()):
        print(f"  {k}: {v} ({v/len(dirs):.1%})")


if __name__ == "__main__":
    main()
