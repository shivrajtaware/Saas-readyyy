import pandas as pd
import numpy as np

FEATURE_COLS = [
    "ret_1d",
    "vol_5d",
    "ema20_vs_close",
    "ema50_vs_close",
    "rsi",
    "macd",
    "bb_pos",
]


def _to_series(x):
    # Convert DataFrame to Series if needed (fix for Yahoo nested columns)
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, -1]
    return x


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build final feature DataFrame for ML.
    Fixes Yahoo Finance multi-column OHLC structure.
    Ensures all columns used are Series, not DataFrame.
    """

    df = df.copy()

    # Force single column selection
    close = _to_series(df["Close"])
    ema20 = _to_series(df.get("EMA20", close))
    ema50 = _to_series(df.get("EMA50", close))
    rsi = _to_series(df.get("RSI", 50))
    macd = _to_series(df.get("MACD", 0))
    bb_up = _to_series(df.get("BB_UPPER", close))
    bb_lo = _to_series(df.get("BB_LOWER", close))

    # Basic returns & volatility
    df["ret_1d"] = close.pct_change()
    df["vol_5d"] = df["ret_1d"].rolling(5).std()

    # Price vs EMAs
    df["ema20_vs_close"] = (ema20 - close) / close
    df["ema50_vs_close"] = (ema50 - close) / close

    # RSI + MACD use existing processed values
    df["rsi"] = rsi
    df["macd"] = macd

    # Bollinger band position
    df["bb_pos"] = (close - bb_lo) / (bb_up - bb_lo).replace(0, np.nan)

    # Final: return only required feature columns and drop NaN rows
    feats = df[FEATURE_COLS].dropna()

    return feats
