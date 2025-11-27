def position_sizing(capital: float, risk_pct: float, entry: float, stop: float) -> dict:
    """
    Simple risk model:
      - risk_pct of capital per trade
      - Qty based on (entry - stop)
    """
    risk_per_trade = capital * (risk_pct / 100.0)
    risk_per_share = max(entry - stop, 0.01)
    qty = int(risk_per_trade // risk_per_share)

    return {
        "risk_per_share": round(risk_per_share, 2),
        "max_loss": round(risk_per_trade, 2),
        "quantity": qty
    }
