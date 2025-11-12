import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="StochRSI Backtest", layout="wide")

# -------- Helpers --------
def pick_close(df: pd.DataFrame) -> pd.Series:
    """Hole die Close-Spalte robust (egal ob 'Close'/'close', MultiIndex, etc.)."""
    if df is None or df.empty:
        return pd.Series(dtype=float)

    # MultiIndex-Spalten flach ziehen (bei Einzelticker holen wir die unterste Ebene)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # Case-insensitive Mapping
    cols = list(map(str, df.columns))
    lower_map = {c.lower(): c for c in cols}
    if "close" in lower_map:
        col = lower_map["close"]
        ser = df[col]
    else:
        # Fallback: Wenn nur eine Spalte vorhanden ist (z.B. bereits Close)
        if df.shape[1] == 1:
            ser = df.iloc[:, 0]
        else:
            raise KeyError("Keine 'close'-Spalte gefunden.")

    # Sicherstellen: 1D-Serie, sauberer Zeitindex ohne TZ
    ser = pd.Series(ser.squeeze(), index=df.index)
    if getattr(ser.index, "tz", None) is not None:
        ser.index = ser.index.tz_convert(None)
    ser.index = pd.to_datetime(ser.index).tz_localize(None)
    ser.name = "close"
    return ser.astype(float)

# -------- Indicator --------
def stoch_rsi(close, rsi_len=14, stoch_len=14, k=3, d=3):
    close = pd.Series(close.squeeze(), index=close.index)
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    roll_up   = pd.Series(gain, index=close.index).rolling(rsi_len).mean()
    roll_down = pd.Series(loss, index=close.index).rolling(rsi_len).mean()
    rs  = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))

    rsi_min = rsi.rolling(stoch_len).min()
    rsi_max = rsi.rolling(stoch_len).max()
    srs   = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-12)

    k_line = srs.rolling(k).mean() * 100
    d_line = k_line.rolling(d).mean()
    out = pd.DataFrame({"K": k_line, "D": d_line}, index=close.index)
    return out.dropna()

# -------- Backtest --------
def run_backtest(df):
    df["bull_cross"] = (df["K"].shift(1) < df["D"].shift(1)) & (df["K"] > df["D"])
    df["bear_cross"] = (df["K"].shift(1) > df["D"].shift(1)) & (df["K"] < df["D"])
    # Extremzonen-Filter reduziert Fehlsignale
    df["bull_cross"] &= (df["K"].shift(1) < 20)
    df["bear_cross"] &= (df["K"].shift(1) > 80)

    cash, position = 10000.0, 0.0
    trades = []
    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        if row["bull_cross"] and cash > 0:
            position = cash / price
            cash = 0.0
            trades.append(("BUY", row.name, price))
        elif row["bear_cross"] and position > 0:
            cash = position * price
            position = 0.0
            trades.append(("SELL", row.name, price))

    final_value = cash + position * float(df["close"].iloc[-1])
    perf = (final_value / 10000.0 - 1.0) * 100.0
    return final_value, perf, trades

# -------- UI --------
st.title("📊 StochRSI Backtest Dashboard")
col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.text_input("Symbol (z. B. BTC-USD, ETH-USD, AAPL)", "BTC-USD")
with col2:
    interval = st.selectbox("Zeiteinheit", ["1h", "4h", "1d", "1wk"], index=2)
with col3:
    years = st.slider("Zeitraum (Jahre)", 1, 10, 3)

# -------- Run --------
try:
    # yfinance-Periodenlimits für Intraday beachten
    period = f"{years}y"
    if interval in ("1h", "4h"):
        period = "60d" if interval == "1h" else "730d"

    df_raw = yf.download(symbol, period=period, interval=interval, group_by="column")
    if df_raw is None or df_raw.empty:
        st.warning("Keine Daten geladen. Versuch ein anderes Symbol/Intervall.")
    else:
        close = pick_close(df_raw)
        df = pd.DataFrame({"close": close})

        stoch = stoch_rsi(df["close"])
        df = pd.concat([df, stoch], axis=1).dropna()

        final_value, perf, trades = run_backtest(df)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df["close"], label="Kurs")
        ax.scatter(df.index[df["bull_cross"]], df["close"][df["bull_cross"]], label="Buy", marker="^")
        ax.scatter(df.index[df["bear_cross"]], df["close"][df["bear_cross"]], label="Sell", marker="v")
        ax.set_title(f"{symbol} – StochRSI Cross Signale ({interval})")
        ax.legend()
        st.pyplot(fig)

        st.success(f"💰 Endwert: **{final_value:,.2f}**  |  Gewinn: **{perf:.2f}%**")
        if trades:
            trade_df = pd.DataFrame(trades, columns=["Typ", "Datum", "Preis"]).set_index("Datum")
            st.dataframe(trade_df)
        else:
            st.info("Keine Signale im gewählten Zeitraum.")

except Exception as e:
    st.error(f"Fehler: {e}")