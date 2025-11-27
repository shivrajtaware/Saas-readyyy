import pandas as pd

def get_key_levels(df: pd.DataFrame):
    """
    Returns support & resistance levels.
    Called by dashboard pack E.
    """
    try:
        closes = df["Close"].tail(200)

        support = closes.min()
        resistance = closes.max()

        mid = (support + resistance) / 2

        return {
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "mid_level": round(mid, 2),
        }
    except Exception as e:
        return None
