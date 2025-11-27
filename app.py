import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any
import streamlit.components.v1 as components

# ---------------------------
# Your core modules (main)
# ---------------------------
from core.data_fetcher import fetch_price_data
from core.indicators import add_indicators
from core.signal_engine import TradingStyle, generate_breakout_advice, safe_float

# ---------------------------
# Advanced features imports (robust)
# ---------------------------

# pattern_detector -> analyze_patterns
try:
    from core.pattern_detector import analyze_patterns
except Exception:
    try:
        from core.pattern_detector import analyze_patterns
    except Exception:
        def analyze_patterns(df: pd.DataFrame):
            return []

# key_levels -> get_key_levels
try:
    from core.key_levels import get_key_levels
except Exception:
    try:
        from core.key_levels import calc_key_levels as get_key_levels
    except Exception:
        def get_key_levels(df: pd.DataFrame):
            return {}

# mtf_engine -> get_mtf_strength_map
try:
    from core.mtf_engine import get_mtf_strength_map
except Exception:
    try:
        from core.mtf_engine import analyze_mtf as _analyze_mtf

        def get_mtf_strength_map(df: pd.DataFrame):
            m = _analyze_mtf(df) or {}
            return {
                "5-bar": m.get("strength", {}).get("trend_5", "N/A"),
                "20-bar": m.get("strength", {}).get("trend_20", "N/A"),
                "50-bar": m.get("strength", {}).get("trend_50", "N/A"),
                "score": m.get("score", 0),
                "final_view": m.get("view", "N/A"),
            }
    except Exception:
        def get_mtf_strength_map(df: pd.DataFrame):
            return {}

# Optional: trend scanner & risk model & volume heatmap (fallback stubs)
try:
    from core.trend_scanner import scan_trends as scan_trends
except Exception:
    def scan_trends(df: pd.DataFrame) -> Dict[str, Any]:
        # lightweight fallback: EMA cross quick check
        out = {}
        try:
            close = df["Close"]
            ema20 = close.ewm(span=20).mean().iloc[-1]
            ema50 = close.ewm(span=50).mean().iloc[-1]
            out["ema20"] = float(ema20)
            out["ema50"] = float(ema50)
            out["signal"] = "Bullish" if ema20 > ema50 else "Bearish"
        except Exception:
            out = {}
        return out

try:
    from core.risk_manager import evaluate_risk as evaluate_risk
except Exception:
    def evaluate_risk(df: pd.DataFrame, capital: float, risk_pct: float) -> Dict[str, Any]:
        # Very simple fallback risk sizing
        try:
            last = float(df["Close"].iloc[-1])
            risk_amount = capital * (risk_pct / 100.0)
            qty = max(int(risk_amount / (last * 0.02)), 0)  # assume 2% SL per share
            return {"risk_amount": round(risk_amount, 2), "suggested_qty": qty}
        except Exception:
            return {}

try:
    from core.volume_heatmap import build_volume_heatmap as build_volume_heatmap
except Exception:
    def build_volume_heatmap(df: pd.DataFrame):
        # fallback: return basic volume summary
        try:
            v = df["Volume"].tail(50)
            return {"avg_volume": float(v.mean()), "last_volume": float(v.iloc[-1])}
        except Exception:
            return {}

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="SephTrade – AI Stock Advisor", layout="wide")
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppToolbar {display: none;}
    .st-emotion-cache-18ni7ap {display: none;}  /* sometimes toolbar ID */
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

TIMEFRAME_OPTIONS = {
    "1m": ("7d", "1m"),
    "5m": ("7d", "5m"),
    "15m": ("30d", "15m"),
    "30m": ("60d", "30m"),
    "1h": ("60d", "60m"),
    # use 60m and resample to 4h (Yahoo often breaks on 240m)
    "4h": ("90d", "60m"),
    "1d": ("1y", "1d"),
    "1wk": ("5y", "1wk"),
}

# default TF for each style
STYLE_DEFAULT_TF = {
    "Intraday": "15m",
    "BTST": "1d",
    "Swing": "1d",
}

# holding hint for style mode
STYLE_HOLDING_HINT = {
    "Intraday": "Intraday (same day)",
    "BTST": "BTST (1–2 days)",
    "Swing": "Swing (few days to weeks)",
}

# holding hint for manual timeframe
TF_HOLDING_HINT = {
    "1m": "Ultra-short scalp",
    "5m": "Scalp / very short-term",
    "15m": "Intraday trade",
    "30m": "Intraday trade",
    "1h": "Intraday / short swing",
    "4h": "Short swing (1–5 days)",
    "1d": "BTST / Swing (days–weeks)",
    "1wk": "Positional / medium term",
}

# ----------------- THEME LOADER -----------------
def apply_premium_theme():
    try:
        with open("assets/theme.css", "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


# ----------------- MAIN APP -----------------
def main():
    apply_premium_theme()

    # ---- SIDEBAR BRAND ----
    with st.sidebar:
        st.markdown(
            """
            <div style="padding-bottom:6px;">
                <div class="seph-logo">★ SEPH<span style="color:#e5e7eb;">TRADE</span></div>
                <div class="seph-subtitle">AI Hybrid Breakout Advisor</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

    # ================= SIDEBAR CONTROLS ==================
    mode = st.sidebar.radio(
        "Mode",
        ["Trading Style", "Manual Timeframe"],
        index=0,
        key="mode_radio",
    )

    # defaults
    style_label = "Intraday"
    style = TradingStyle.INTRADAY
    tf_label = "1d"

    style_map = {
        "Intraday": TradingStyle.INTRADAY,
        "BTST": TradingStyle.BTST,
        "Swing": TradingStyle.SWING,
    }

    if mode == "Trading Style":
        style_label = st.sidebar.selectbox(
            "Trading Style",
            ["Intraday", "BTST", "Swing"],
            key="style_select",
        )
        style = style_map[style_label]

        tf_label = STYLE_DEFAULT_TF.get(style_label, "1d")
        st.sidebar.markdown(
            f"**Timeframe (auto):** `{tf_label}` based on **{style_label}**"
        )
    else:
        style_label = "Manual"
        style = TradingStyle.INTRADAY  # internal engine behaviour
        tf_label = st.sidebar.selectbox(
            "Manual Timeframe",
            list(TIMEFRAME_OPTIONS.keys()),
            index=6,
            key="tf_manual_select",
        )
        st.sidebar.markdown(
            f"**Timeframe (manual):** `{tf_label}` – custom user selection"
        )

    period, interval = TIMEFRAME_OPTIONS[tf_label]

    symbol = st.sidebar.text_input(
        "NSE Symbol (e.g. RELIANCE.NS)",
        "RELIANCE.NS",
        key="symbol_input",
    )

    capital = st.sidebar.number_input(
        "Capital (₹)",
        min_value=1000.0,
        value=50000.0,
        step=1000.0,
        key="capital_input",
    )

    risk_pct = st.sidebar.slider(
        "Risk per trade (%)",
        min_value=0.5,
        max_value=9.0,
        value=2.0,
        step=0.5,
        key="risk_slider",
    )

    st.sidebar.markdown("### Chart Panels")
    show_rsi = st.sidebar.checkbox("Show RSI panel", value=True, key="show_rsi_cb")
    show_macd = st.sidebar.checkbox("Show MACD panel", value=True, key="show_macd_cb")
    show_bb = st.sidebar.checkbox("Show Bollinger Bands", value=True, key="show_bb_cb")

    run = st.sidebar.button("🚀 Generate Advice", key="run_button")

    # ---- HEADER ----
    # st.markdown("## 📈 SephTrade – Luxury AI Stock Advisor")
    # st.caption(
    #     "Hybrid ML + Breakout engine. Educational use only. For paid advisory you still need SEBI registration & broker agreement."
    # )

    if not run:
        st.info("Choose symbol, mode, style/timeframe and click **Generate Advice**.")
        return

    # ================= DATA & INDICATORS ==================
    with st.spinner("Fetching data & computing indicators..."):
        df = fetch_price_data(symbol, period, interval)
        if df is None or df.empty:
            st.error("Could not fetch data. Check symbol or internet.")
            return

        # resample for 4h if needed
        if tf_label == "4h" and interval == "60m":
            try:
                df = df.resample("4h").agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                ).dropna()
            except Exception:
                pass

        df = add_indicators(df)

    # --- NORMALIZE SIGNAL (robust fallback if core returns minimal fields) ---
    raw_signal = generate_breakout_advice(df, style, capital, risk_pct)

    if not raw_signal or not isinstance(raw_signal, dict):
        raw_signal = {
            "side": "NO_TRADE",
            "message": "Signal engine returned no output.",
            "confidence": 0,
            "reasons": ["Signal engine produced no valid output."],
            "ml": None,
        }

    signal: Dict[str, Any] = {}
    signal["side"] = raw_signal.get("side", "NO_TRADE")
    signal["message"] = raw_signal.get("message", "")
    signal["confidence"] = raw_signal.get("confidence", 0)
    signal["reasons"] = raw_signal.get("reasons", [])
    signal["ml"] = raw_signal.get("ml", None)

    last_price = safe_float(df["Close"].iloc[-1])

    signal["trigger"] = raw_signal.get("trigger", round(last_price * 1.01, 2))
    signal["t1"] = raw_signal.get("t1", round(signal["trigger"] * 1.03, 2))
    signal["t2"] = raw_signal.get("t2", round(signal["trigger"] * 1.06, 2))
    signal["sl"] = raw_signal.get("sl", round(last_price * 0.98, 2))

    rr_num = signal["t1"] - signal["sl"]
    rr_den = max(signal["trigger"] - signal["sl"], 1e-6)
    signal["rr"] = raw_signal.get("rr", round(rr_num / rr_den, 2))

    # Holding logic based on mode
    if mode == "Trading Style":
        holding_hint = STYLE_HOLDING_HINT.get(style_label, "Intraday/Swing")
        timeframe_base = f"Auto – {tf_label} (by {style_label} style)"
    else:
        holding_hint = TF_HOLDING_HINT.get(tf_label, "Custom timeframe")
        timeframe_base = f"Manual – {tf_label}"

    signal["holding"] = raw_signal.get("holding", holding_hint)
    signal["timeframe_base"] = timeframe_base
    signal["recent_high"] = raw_signal.get(
        "recent_high",
        round(df["High"].tail(20).max(), 2) if "High" in df else last_price,
    )
    signal["recent_low"] = raw_signal.get(
        "recent_low",
        round(df["Low"].tail(20).min(), 2) if "Low" in df else last_price,
    )

    # Risk object: try to use raw risk if provided, otherwise simple sizing
    risk = raw_signal.get("risk", None)
    if not risk or not isinstance(risk, dict):
        try:
            computed = evaluate_risk(df, capital, risk_pct)
            risk = {
                "risk_amount": computed.get("risk_amount", 0),
                "risk_per_share": round(
                    (signal["trigger"] - signal["sl"])
                    if (signal["trigger"] > signal["sl"])
                    else 0,
                    2,
                ),
                "max_loss": round(computed.get("risk_amount", 0), 2),
                "quantity": int(computed.get("suggested_qty", 0)),
            }
        except Exception:
            risk_amount = capital * (risk_pct / 100.0)
            risk = {
                "risk_amount": round(risk_amount, 2),
                "risk_per_share": round(
                    (signal["trigger"] - signal["sl"])
                    if (signal["trigger"] > signal["sl"])
                    else 0,
                    2,
                ),
                "max_loss": round(risk_amount, 2),
                "quantity": 0,
            }
    signal["risk"] = risk

    ml_view = signal["ml"] if signal is not None and "ml" in signal else None

    # ================= TABS (MAIN NAV) ==================
    tab_dashboard, tab_advice, tab_ml, tab_chart, tab_data, tab_pro = st.tabs(
        ["🏠 Dashboard", "🎯 Advice", "🧠 ML Insight", "📈 Chart", "📊 Data", "🔍 Pro Insights"]
    )

    # -------------- DASHBOARD TAB --------------
    with tab_dashboard:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class="seph-card">
                    <div class="seph-label">Instrument</div>
                    <div class="seph-value">{symbol}</div>
                    <div class="seph-label" style="margin-top:6px;">Style / TF</div>
                    <div class="seph-value">{style_label} · {tf_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            side_txt = "No Signal"
            if signal is not None:
                if signal["side"] == "BUY":
                    side_txt = "BUY BREAKOUT"
                elif signal["side"] == "NO_TRADE":
                    side_txt = "NO TRADE"

            pill_class = "pill-gold"
            if signal is not None:
                if signal["side"] == "BUY":
                    pill_class = "pill-green"
                elif signal["side"] == "NO_TRADE":
                    pill_class = "pill-red"

            conf = signal["confidence"] if signal is not None else 0

            st.markdown(
                f"""
                <div class="seph-card">
                    <div class="seph-label">Last Price</div>
                    <div class="seph-value">₹{last_price:.2f}</div>
                    <div style="margin-top:8px;">
                        <span class="pill {pill_class}">{side_txt}</span>
                        <span class="pill pill-gold" style="margin-left:6px;">CONF {conf}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Quick Snapshot")
        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric(
                "ML Up Probability",
                f"{ml_view['prob_up']*100:.1f}%" if ml_view else "NA",
            )
        with col4:
            st.metric(
                "ML Down Probability",
                f"{ml_view['prob_down']*100:.1f}%" if ml_view else "NA",
            )
        with col5:
            st.metric("ML Label", ml_view["label"] if ml_view else "NA")

        if signal is not None:
            st.markdown("#### Market View")
            if signal["side"] == "NO_TRADE":
                st.write("🔸 Market not giving a clean long breakout setup right now.")
            else:
                st.write("🟢 Market is suitable for long breakout entries above trigger.")

    # -------------- ADVICE TAB --------------
    with tab_advice:
        st.subheader(f"🎯 Trade Advice – {symbol}")

        if signal is None:
            st.warning("Not enough data to generate a valid setup.")
            st.stop()

        # =================== NO TRADE CASE ===================
        if signal["side"] == "NO_TRADE":
            st.error("❌ NO TRADE ZONE")
            st.write("**Message:**", signal["message"])

            st.markdown(
                f"""
                <div style="
                    width:140px; height:140px;
                    border-radius:50%;
                    margin:15px 0;
                    background: conic-gradient(#ef4444 {signal['confidence']*3.6}deg, #1e293b 0deg);
                    display:flex; align-items:center; justify-content:center;
                    font-size:1.7rem; font-weight:700; color:white;">
                    {signal['confidence']}%
                </div>
                """,
                unsafe_allow_html=True,
            )

            if ml_view:
                st.markdown("### 🧠 ML View")
                st.write(
                    f"**{ml_view['label']}**\n\n"
                    f"Up={ml_view['prob_up']*100:.1f}% | Down={ml_view['prob_down']*100:.1f}%"
                )

            st.markdown("### 🧠 Technical Reasoning")
            for r in signal.get("reasons", []):
                st.write("•", r)

            st.stop()

        # =================== BUY / LONG BREAKOUT SIGNAL ===================
        col_left, col_right = st.columns([2.3, 1])

        advice_html = f"""
        <div style="
            padding:24px;
            border-radius:20px;
            backdrop-filter: blur(14px);
            background: linear-gradient(135deg, rgba(15,23,42,0.9) 0%, rgba(30,41,59,0.85) 100%);
            border: 1px solid rgba(255,255,255,0.1);
            color:white;
            font-size:16px;
            line-height:1.6;">
            <div style="
                background:#22c55e;
                color:black;
                padding:6px 14px;
                border-radius:10px;
                display:inline-block;
                font-weight:700;
                font-size:15px;">
                🟢 ACTIVE LONG BREAKOUT
            </div>

            <h2 style="margin-top:10px; font-weight:700;">
                BUY ABOVE ₹{signal['trigger']}
            </h2>

            Entry becomes valid only if price breaks & sustains above trigger.
            <br><br>

            <b>🎯 Targets</b>
            <ul style="margin-top:4px;">
                <li><b>T1:</b> ₹{signal['t1']}</li>
                <li><b>T2:</b> ₹{signal['t2']}</li>
            </ul>

            <p><b>🛑 Stop Loss:</b> ₹{signal['sl']}</p>
            <p><b>📊 Risk/Reward:</b> {signal['rr']}:1</p>
            <p><b>⏳ Holding:</b> {signal['holding']}</p>
            <p><b>🕐 Timeframe Base:</b> {signal['timeframe_base']}</p>
            <p><b>📍 Recent High / Low:</b> ₹{signal['recent_high']} / ₹{signal['recent_low']}</p>
        </div>
        """

        with col_left:
            components.html(advice_html, height=780, scrolling=True)

        with col_right:
            components.html(
                f"""
                <div style="
                    width:160px; height:160px;
                    border-radius:50%;
                    margin:auto;
                    background: conic-gradient(#22c55e {signal['confidence']*3.6}deg, #1e293b 0deg);
                    display:flex; align-items:center; justify-content:center;
                    font-size:32px; font-weight:700; color:white;">
                    {signal['confidence']}%
                </div>
                """,
                height=260,
            )

        # =================== RISK MODEL ===================
        st.markdown("### 📏 Position Sizing (Risk Model)")

        try:
            risk_amount = capital * (risk_pct / 100.0)
            trigger_px = float(signal.get("trigger", last_price))
            sl_px = float(signal.get("sl", last_price * 0.98))

            risk_per_share = abs(trigger_px - sl_px)
            if risk_per_share <= 0:
                risk_per_share = last_price * 0.01

            qty = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
            if qty * trigger_px > capital and trigger_px > 0:
                qty = int(capital / trigger_px)
            qty = max(qty, 0)

            risk_calc = {
                "risk_amount": round(risk_amount, 2),
                "risk_per_share": round(risk_per_share, 2),
                "max_loss": round(qty * risk_per_share, 2),
                "quantity": qty,
            }
        except Exception:
            risk_calc = {"risk_amount": 0, "risk_per_share": 0, "max_loss": 0, "quantity": 0}

        st.metric("Risk / Share", f"₹{risk_calc['risk_per_share']}")
        st.metric("Max Loss", f"₹{risk_calc['max_loss']}")
        st.metric(
            "Suggested Qty",
            f"{risk_calc['quantity']} shares" if risk_calc["quantity"] > 0 else "Too risky",
        )

        # =================== ML VIEW ===================
        if ml_view:
            st.markdown("### 🧠 ML View")
            st.write(
                f"**{ml_view['label']}**\n\n"
                f"Up={ml_view['prob_up']*100:.1f}% | Down={ml_view['prob_down']*100:.1f}%"
            )

        # =================== TECHNICAL REASONS ===================
        st.markdown("### 🧠 Technical Reasoning")
        for r in signal.get("reasons", []):
            st.write("•", r)

    # -------------- ML INSIGHT TAB --------------
    with tab_ml:
        st.subheader("🧠 ML Insight")
        if ml_view is None:
            st.info("ML view not available.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("ML Label", ml_view["label"])
                st.metric("Up Probability", f"{ml_view['prob_up']*100:.2f}%")
                st.metric("Down Probability", f"{ml_view['prob_down']*100:.2f}%")
                st.write(f"Source: `{ml_view.get('source','model')}`")
            with c2:
                prob_fig = go.Figure()
                prob_fig.add_trace(
                    go.Bar(
                        x=["Up", "Down"],
                        y=[ml_view["prob_up"] * 100, ml_view["prob_down"] * 100],
                    )
                )
                prob_fig.update_layout(height=300, yaxis=dict(range=[0, 100]))
                st.plotly_chart(prob_fig, use_container_width=True)
        st.markdown("### How ML is used here?")
        st.write(
            "- Price + indicators → feature vector\n"
            "- Model outputs Up/Down probabilities\n"
            "- Final advice = ML + indicator rules + risk filters"
        )

    # -------------- CHART TAB --------------
    with tab_chart:
        st.subheader("📈 Price Chart")
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            )
        )
        if "EMA20" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20"))
        if "EMA50" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50"))
        if show_bb and "BB_UPPER" in df and "BB_LOWER" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_UPPER"], name="BB Upper"))
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOWER"], name="BB Lower"))
        fig.update_layout(
            height=480,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        if show_rsi and "RSI" in df:
            st.subheader("RSI (14)")
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI"]))
            rsi_fig.update_layout(height=240, yaxis=dict(range=[0, 100]))
            st.plotly_chart(rsi_fig, use_container_width=True)

        if show_macd and {"MACD", "MACD_SIGNAL", "MACD_HIST"}.issubset(set(df.columns)):
            st.subheader("MACD")
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
            macd_fig.add_trace(
                go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal")
            )
            macd_fig.add_trace(
                go.Bar(x=df.index, y=df["MACD_HIST"], name="Hist")
            )
            macd_fig.update_layout(height=260)
            st.plotly_chart(macd_fig, use_container_width=True)

        st.caption(
            "Realtime candles will be added later via broker API (SmartAPI / IIFL)."
        )

    # -------------- DATA TAB --------------
    with tab_data:
        st.subheader("📊 Raw Data + Indicators (Last 100 rows)")
        st.dataframe(df.tail(100))

    # -------------- PRO INSIGHTS TAB --------------
    with tab_pro:
        st.markdown(
            "<div class='top-navbar'>🔍 <strong>Pro Insights</strong></div>",
            unsafe_allow_html=True,
        )
        st.markdown("### Premium Market Packs (A → F)")

        c1, c2 = st.columns(2)

        # Pack A — Volume Heatmap
        with c1:
            st.markdown("<div class='seph-card'>", unsafe_allow_html=True)
            st.markdown("### 📊 Pack A — Volume Heatmap")
            try:
                vh = build_volume_heatmap(df)
                if isinstance(vh, dict):
                    for k, v in vh.items():
                        st.write(f"**{k.replace('_',' ').title()}:** {v}")
                else:
                    st.write(
                        "Volume heatmap available (visualization will be added)."
                    )
            except Exception as e:
                st.write("Volume heatmap error:", e)
            st.markdown("</div>", unsafe_allow_html=True)

        # Pack B — MTF Strength Map
        with c2:
            st.markdown("<div class='seph-card'>", unsafe_allow_html=True)
            st.markdown("### 🧭 Pack B — Multi-Timeframe Strength Map")
            try:
                mtf = get_mtf_strength_map(df)
                if mtf:
                    st.write(f"Final view: **{mtf.get('final_view','NA')}**")
                    st.write(f"Score: {mtf.get('score','NA')}")
                    st.write("Details:")
                    st.json(mtf)
                else:
                    st.write("MTF map not available.")
            except Exception as e:
                st.write("MTF error:", e)
            st.markdown("</div>", unsafe_allow_html=True)

        c3, c4 = st.columns(2)

        # Pack C — Trend Scanner
        with c3:
            st.markdown("<div class='seph-card'>", unsafe_allow_html=True)
            st.markdown("### 🔎 Pack C — Trend Scanner")
            try:
                ts = scan_trends(df)
                if ts:
                    st.write("Signal:", ts.get("signal", "NA"))
                    st.write("EMA20:", ts.get("ema20", "NA"))
                    st.write("EMA50:", ts.get("ema50", "NA"))
                else:
                    st.write("Trend scanner not available.")
            except Exception as e:
                st.write("Trend scanner error:", e)
            st.markdown("</div>", unsafe_allow_html=True)

        # Pack D — Risk Model
        with c4:
            st.markdown("<div class='seph-card'>", unsafe_allow_html=True)
            st.markdown("### 🛡 Pack D — Risk Model")
            try:
                rm = evaluate_risk(df, capital, risk_pct)
                if rm:
                    st.write("Risk Amount:", rm.get("risk_amount", "NA"))
                    st.write("Suggested Quantity:", rm.get("suggested_qty", "NA"))
                else:
                    st.write("Risk model not available.")
            except Exception as e:
                st.write("Risk model error:", e)
            st.markdown("</div>", unsafe_allow_html=True)

        c5, c6 = st.columns(2)

        # Pack E — Key Levels & Zones
        with c5:
            st.markdown("<div class='seph-card'>", unsafe_allow_html=True)
            st.markdown("### 🧱 Pack E — Key Levels & Zones")
            try:
                kl = get_key_levels(df)
                if kl and isinstance(kl, dict):
                    st.write("Support:", kl.get("support", "NA"))
                    st.write("Resistance:", kl.get("resistance", "NA"))
                    st.write("Mid Level:", kl.get("mid_level", "NA"))
                else:
                    st.write("Key levels not available.")
            except Exception as e:
                st.write("Key levels error:", e)
            st.markdown("</div>", unsafe_allow_html=True)

        # Pack F — Pattern Detection
        with c6:
            st.markdown("<div class='seph-card'>", unsafe_allow_html=True)
            st.markdown("### 🕯️ Pack F — Pattern Detection")
            try:
                patterns = analyze_patterns(df)
                if patterns:
                    for p in patterns:
                        st.markdown(f"- {p}")
                else:
                    st.write("No strong patterns detected.")
            except Exception as e:
                st.write("Pattern detection error:", e)
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
