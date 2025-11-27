import pandas as pd
import numpy as np

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    fast_e = ema(series, fast)
    slow_e = ema(series, slow)
    macd_line = fast_e - slow_e
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    return ma, upper, lower

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds EMA20, EMA50, RSI(14), MACD, Bollinger Bands.
    """
    close = df["Close"]

    df["EMA20"] = ema(close, 20)
    df["EMA50"] = ema(close, 50)
    df["EMA9"] = df["Close"].ewm(span=9).mean()
    df["EMA21"] = df["Close"].ewm(span=21).mean()


    m, s, h = macd(close)
    df["MACD"] = m
    df["MACD_SIGNAL"] = s
    df["MACD_HIST"] = h

    df["RSI"] = rsi(close, 14)

    mid, up, low = bollinger(close, 20, 2.0)
    df["BB_MID"] = mid
    df["BB_UPPER"] = up
    df["BB_LOWER"] = low

    return df
