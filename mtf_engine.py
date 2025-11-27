import pandas as pd

def analyze_mtf(df: pd.DataFrame):
    """
    Basic multi-timeframe trend strength calculation.
    This is used internally by get_mtf_strength_map.
    """

    try:
        closes = df["Close"]

        strength = {
            "trend_5": "Up" if closes.tail(5).iloc[-1] > closes.tail(5).iloc[0] else "Down",
            "trend_20": "Up" if closes.tail(20).iloc[-1] > closes.tail(20).iloc[0] else "Down",
            "trend_50": "Up" if closes.tail(50).iloc[-1] > closes.tail(50).iloc[0] else "Down",
        }

        score = (
            (1 if strength["trend_5"] == "Up" else -1)
            + (1 if strength["trend_20"] == "Up" else -1)
            + (1 if strength["trend_50"] == "Up" else -1)
        )

        final_view = "Bullish" if score > 0 else "Bearish"

        return {
            "strength": strength,
            "score": score,
            "view": final_view,
        }

    except Exception:
        return None


def get_mtf_strength_map(df: pd.DataFrame):
    """
    EXACT FUNCTION NAME app.py expects.
    Generates the multi-timeframe strength card on the dashboard.
    """

    try:
        mtf = analyze_mtf(df)
        if mtf is None:
            return None

        # Format for UI
        return {
            "5-bar": mtf["strength"]["trend_5"],
            "20-bar": mtf["strength"]["trend_20"],
            "50-bar": mtf["strength"]["trend_50"],
            "score": mtf["score"],
            "final_view": mtf["view"],
        }

    except Exception:
        return None
