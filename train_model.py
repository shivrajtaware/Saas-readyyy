import os
import joblib
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

from core.indicators import add_indicators
from core.features import make_features, FEATURE_COLS
from config import MODEL_PATH

# You can extend this list
TRAIN_SYMBOLS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS"]

def build_training_data() -> pd.DataFrame:
    frames = []
    for sym in TRAIN_SYMBOLS:
        print(f"Downloading {sym}...")
        df = yf.download(sym, period="3y", interval="1d", progress=False)
        if df is None or df.empty:
            continue
        df = df.dropna()
        df = add_indicators(df)

        feats = make_features(df)
        # Align feats with df index
        df = df.loc[feats.index]

        # target: 1 if next day's close > today's close else 0
        next_close = df["Close"].shift(-1)
        target = (next_close > df["Close"]).astype(int)
        # Drop last row with NaN target
        valid = target.dropna().index
        feats = feats.loc[valid]
        target = target.loc[valid]

        tmp = feats.copy()
        tmp["target"] = target
        tmp["symbol"] = sym
        frames.append(tmp)

    if not frames:
        raise RuntimeError("No training data built.")
    data = pd.concat(frames, axis=0)
    return data

def main():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    data = build_training_data()
    print("Training data shape:", data.shape)

    X = data[FEATURE_COLS]
    y = data["target"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print("Model saved to:", MODEL_PATH)

if __name__ == "__main__":
    main()
