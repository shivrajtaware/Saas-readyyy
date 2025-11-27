import pandas as pd
import numpy as np


def _s(x):
    """Force Series from DataFrame column."""
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, -1]
    return x


def analyze_patterns(df: pd.DataFrame):
    patterns = []

    # Convert all OHLC columns safely
    o = _s(df["Open"])
    h = _s(df["High"])
    l = _s(df["Low"])
    c = _s(df["Close"])

    # Extract last candle
    last_o = float(o.iloc[-1])
    last_h = float(h.iloc[-1])
    last_l = float(l.iloc[-1])
    last_c = float(c.iloc[-1])

    body = abs(last_c - last_o)
    candle_range = last_h - last_l
    upper_wick = last_h - max(last_o, last_c)
    lower_wick = min(last_o, last_c) - last_l

    # -----------------------------------------------------------
    # BASIC SINGLE-CANDLE PATTERNS
    # -----------------------------------------------------------

    # Hammer
    if (
        candle_range > 0
        and lower_wick >= candle_range * 0.5
        and body <= candle_range * 0.35
        and upper_wick <= candle_range * 0.2
    ):
        patterns.append("Hammer")

    # Shooting Star
    if (
        candle_range > 0
        and upper_wick >= candle_range * 0.5
        and body <= candle_range * 0.35
        and lower_wick <= candle_range * 0.2
    ):
        patterns.append("Shooting Star")

    # Doji
    if candle_range > 0 and body <= candle_range * 0.1:
        patterns.append("Doji")

    # -----------------------------------------------------------
    # ENGULFING PATTERNS
    # -----------------------------------------------------------
    if len(df) >= 2:
        prev_o = float(o.iloc[-2])
        prev_c = float(c.iloc[-2])

        # Bullish Engulfing
        if prev_c < prev_o and last_c > last_o and last_c > prev_o and last_o < prev_c:
            patterns.append("Bullish Engulfing")

        # Bearish Engulfing
        if prev_c > prev_o and last_c < last_o and last_c < prev_o and last_o > prev_c:
            patterns.append("Bearish Engulfing")

    # -----------------------------------------------------------
    # INVERTED HEAD & SHOULDERS (simple version)
    # -----------------------------------------------------------
    if len(df) >= 10:
        closes = c.tail(10).values
        mid = closes[5]
        left = min(closes[1:4])
        right = min(closes[6:9])

        if mid < left and mid < right:
            patterns.append("Inverted Head & Shoulders (basic)")

    # -----------------------------------------------------------
    # DOUBLE BOTTOM
    # -----------------------------------------------------------
    if len(df) >= 12:
        recent = c.tail(12).values
        bottom1 = np.argmin(recent[:6])
        bottom2 = np.argmin(recent[6:])

        if abs(recent[bottom1] - recent[6 + bottom2]) <= recent.mean() * 0.02:
            patterns.append("Double Bottom Pattern")

    # -----------------------------------------------------------
    # BULLISH WEDGE BREAKOUT
    # -----------------------------------------------------------
    if len(df) >= 20:
        highs = h.tail(20).values
        lows = l.tail(20).values
        if highs[0] > highs[-1] and lows[0] > lows[-1]:
            patterns.append("Bullish Falling Wedge")

    # -----------------------------------------------------------
    # ROUNDING BOTTOM (cup-shaped)
    # -----------------------------------------------------------
    if len(df) >= 30:
        closes = c.tail(30).values
        mid = np.argmin(closes)
        if mid > 5 and mid < 25:
            if closes[0] > closes[mid] and closes[-1] > closes[mid]:
                patterns.append("Rounding Bottom")

    # -----------------------------------------------------------
    # BREAKOUT ABOVE MULTIPLE TOPS
    # -----------------------------------------------------------
    if len(df) >= 50:
        closes = c.tail(50).values
        tops = sorted(closes[-10:])
        if last_c > tops[-1]:
            patterns.append("Breakout Above Multiple Tops")

    # -----------------------------------------------------------
    # FLAG PATTERN (bull flag)
    # -----------------------------------------------------------
    if len(df) >= 30:
        closes = c.tail(30).values
        # Sharp rise + consolidation slope downward
        if closes[10] > closes[0] * 1.05:
            if closes[-1] <= closes[20] * 1.02:
                patterns.append("Bull Flag Formation")

    return patterns
