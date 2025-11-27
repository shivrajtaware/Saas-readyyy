import os
import joblib
import pandas as pd
from typing import Dict

from config import MODEL_PATH
from features import make_features, FEATURE_COLS

_model = None  # lazy-loaded model

def _load_model():
    global _model
    if _model is not None:
        return _model
    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
    else:
        _model = None
    return _model

def ml_predict_direction(df_with_indicators: pd.DataFrame) -> Dict:
    """
    Returns ML-based view:
      - prob_up
      - prob_down
      - label
    Falls back to simple rule if no model file exists.
    """
    model = _load_model()
    feats = make_features(df_with_indicators)

    if model is None or feats.empty:
        # fallback heuristic: not real ML but safe
        last = df_with_indicators.iloc[-1]
        rsi_v = float(last.get("RSI", 50))
        ema20 = float(last.get("EMA20", last["Close"]))
        ema50 = float(last.get("EMA50", last["Close"]))
        price = float(last["Close"])
        score_up = 0.5
        score_down = 0.5
        if price > ema20 > ema50:
            score_up += 0.2
        if rsi_v < 35:
            score_up += 0.15
        if rsi_v > 70:
            score_down += 0.15
        total = score_up + score_down
        score_up /= total
        score_down /= total
        label = "UP" if score_up > 0.6 else "DOWN" if score_down > 0.6 else "NEUTRAL"
        return {
            "prob_up": round(score_up, 3),
            "prob_down": round(score_down, 3),
            "label": label,
            "source": "heuristic"
        }

    # Real ML prediction
    x_last = feats.iloc[[-1]][FEATURE_COLS]
    proba = model.predict_proba(x_last)[0]
    # assuming model.classes_ = [0, 1] where 1 = up
    if list(model.classes_) == [0, 1]:
        prob_down, prob_up = float(proba[0]), float(proba[1])
    else:
        # just to be safe
        prob_up = float(proba[0])
        prob_down = 1.0 - prob_up

    label = "UP" if prob_up > 0.6 else "DOWN" if prob_down > 0.6 else "NEUTRAL"

    return {
        "prob_up": round(prob_up, 3),
        "prob_down": round(prob_down, 3),
        "label": label,
        "source": "ml"
    }

