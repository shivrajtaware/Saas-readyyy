from enum import Enum, auto
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from ml_model import ml_predict_direction
from risk_manager import position_sizing

class TradingStyle(Enum):
    INTRADAY = auto()
    BTST = auto()
    SWING = auto()

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float(np.nanmean(x))

def style_params(style: TradingStyle):
    if style == TradingStyle.INTRADAY:
        return dict(t1_mult=0.004, t2_mult=0.008, sl_mult=0.004, breakout_buffer=0.001, holding="Intraday (same day)")
    elif style == TradingStyle.BTST:
        return dict(t1_mult=0.008, t2_mult=0.015, sl_mult=0.006, breakout_buffer=0.0015, holding="BTST (1–3 days)")
    else:
        return dict(t1_mult=0.015, t2_mult=0.03, sl_mult=0.01, breakout_buffer=0.002, holding="Swing (3–10 days)")

def generate_breakout_advice(
    df: pd.DataFrame,
    style: TradingStyle,
    capital: float,
    risk_pct: float
) -> Dict[str, Any] | None:

    if len(df) < 60:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    price = safe_float(last["Close"])
    ema20 = safe_float(last.get("EMA20", price))
    ema50 = safe_float(last.get("EMA50", price))
    rsi_v = safe_float(last.get("RSI", 50))
    macd_v = safe_float(last.get("MACD", 0))
    macd_sig_v = safe_float(last.get("MACD_SIGNAL", 0))
    macd_prev = safe_float(prev.get("MACD", 0))
    macd_sig_prev = safe_float(prev.get("MACD_SIGNAL", 0))
    bb_up = safe_float(last.get("BB_UPPER", price))
    bb_lo = safe_float(last.get("BB_LOWER", price))

    hi_20 = safe_float(df["High"].tail(20).max())
    lo_20 = safe_float(df["Low"].tail(20).min())

    reasons: List[str] = []
    confidence = 50
    direction = "NO_TRADE"

    # Trend logic
    if price > ema20 > ema50:
        direction = "LONG"
        reasons.append("Bullish trend (Close > EMA20 > EMA50).")
        confidence += 15
    else:
        reasons.append("No clean bullish trend.")

    # MACD
    bull_cross = (macd_v > macd_sig_v) and (macd_prev <= macd_sig_prev)
    if bull_cross:
        reasons.append("MACD bullish crossover.")
        confidence += 10
    elif macd_v > macd_sig_v:
        reasons.append("MACD above signal → bullish momentum.")
        confidence += 5
    else:
        reasons.append("MACD not strongly bullish.")

    # RSI
    if rsi_v < 30:
        reasons.append(f"RSI {rsi_v:.1f} oversold → good risk-reward for long.")
        confidence += 8
    elif rsi_v > 70:
        reasons.append(f"RSI {rsi_v:.1f} overbought → chasing high.")
        confidence -= 8
    else:
        reasons.append(f"RSI {rsi_v:.1f} neutral zone.")

    # Bollinger bands
    if price <= bb_lo:
        reasons.append("Near lower Bollinger band → value zone.")
        confidence += 5
    elif price >= bb_up:
        reasons.append("Near upper Bollinger band → stretched upside.")
        confidence -= 5

    # ML view
    ml_view = ml_predict_direction(df)
    reasons.append(f"ML view: {ml_view['label']} (Up={ml_view['prob_up']*100:.1f}%, Down={ml_view['prob_down']*100:.1f}%)")

    if ml_view["label"] == "UP":
        confidence += 10
    elif ml_view["label"] == "DOWN":
        confidence -= 10

    params = style_params(style)

    if direction != "LONG":
        confidence = max(0, min(confidence, 80))
        return {
            "side": "NO_TRADE",
            "message": "No clean long breakout setup. Better to skip.",
            "confidence": int(confidence),
            "holding": params["holding"],
            "reasons": reasons,
            "recent_high": round(hi_20, 2),
            "recent_low": round(lo_20, 2),
            "ml": ml_view
        }

    # Breakout trigger
    trigger = hi_20 * (1 + params["breakout_buffer"])
    t1 = trigger * (1 + params["t1_mult"])
    t2 = trigger * (1 + params["t2_mult"])
    sl = max(lo_20, trigger * (1 - params["sl_mult"]))

    risk_info = position_sizing(capital, risk_pct, trigger, sl)
    rr = (t1 - trigger) / max(trigger - sl, 0.01)
    confidence = max(10, min(confidence, 95))

    return {
        "side": "BUY",
        "trigger": round(trigger, 2),
        "t1": round(t1, 2),
        "t2": round(t2, 2),
        "sl": round(sl, 2),
        "rr": round(rr, 2),
        "confidence": int(confidence),
        "holding": params["holding"],
        "recent_high": round(hi_20, 2),
        "recent_low": round(lo_20, 2),
        "reasons": reasons,
        "ml": ml_view,
        "risk": risk_info
    }

