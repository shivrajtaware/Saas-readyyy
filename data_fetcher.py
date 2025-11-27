import pandas as pd
import yfinance as yf
from typing import Optional
from config import DATA_SOURCE

def fetch_price_data(symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data.
    Currently uses YFinance. Later can be switched to SmartAPI / vendor.
    """
    if DATA_SOURCE == "YFINANCE":
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return None
        return df.dropna()
    elif DATA_SOURCE == "SMARTAPI":
        # TODO: implement SmartAPI based fetch here
        raise NotImplementedError("SMARTAPI data source not implemented yet.")
    else:
        raise ValueError(f"Unsupported DATA_SOURCE: {DATA_SOURCE}")
