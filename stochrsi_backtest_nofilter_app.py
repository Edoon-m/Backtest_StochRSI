import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="StochRSI Cross Signale", layout="wide")

# ---------------------------------------------------------
# 🔥 TradingView-RSI (Wilder) & StochRSI (TV-kompatibel)
# ---------------------------------------------------------

def rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's Moving Average (wie TradingView für RSI)."""
    return series.ewm(alpha=1/length, adjust=False).mean()

def stoch_rsi_tv_like(close, rsi_len=14, stoch_len=14, k=3, d=3):
    """
    StochRSI identisch zu TradingView:
    1) Wilder RSI
    2) Stochastic über RSI
    3) K & D geglättet mit einfachen SMA
    """
    close = pd.Series(close).astype(float)
    delta = close.diff()

    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    avg_gain = rma(up, rsi_len)
    avg_loss = rma(down, rsi_len)

    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))

    lowest_rsi = rsi.rolling(stoch_len).min()
    highest_rsi = rsi.rolling(stoch_len).max()

    stoch = (rsi - lowest_rsi) / (highest_rsi - lowest_rsi + 1e-12) * 100

    k_line = stoch.rolling(k).mean()
    d_line = k_line.rolling(d).mean()

    return pd.DataFrame({"K": k_line, "D": d_line}).dropna()

# ---------------------------------------------------------
# Cross-Berechnung (immer bei Schnitt)
# ---------------------------------------------------------

def find_cross_signals(df):
    """Buy = K kreuzt D von unten. Sell = K kreuzt D von oben."""
    df["bull_cross"] = (df["K"].shift(1) < df["D"].shift(1)) & (df["K"] > df["D"])
    df["bear_cross"] = (df["K"].shift(1) > df["D"].shift(1)) & (df["K"] < df["D"])
    return df

# ---------------------------------------------------------
# Backtest (immer Buy/Sell bei jedem Kreuz)
# ---------------------------------------------------------

def run_backtest(df):
    cash = 10000
    position = 0
    trades = []

    for i in range(len(df)):
        price = df["close"].iloc[i]

        if df["bull_cross"].iloc[i] and cash > 0:
            position = cash / price
            cash = 0
            trades.append(("BUY", df.index[i], price))

        elif df["bear_cross"].iloc[i] and position > 0:
            cash = position * price
            position = 0
            trades.append(("SELL", df.index[i], price))

    final_value = cash + position * df["close"].iloc[-1]
    perf = (final_value / 10000 - 1) * 100
    return final_value, perf, trades

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.title("📊 TradingView-genauer StochRSI – Kauf/Verkauf bei jedem Cross")

col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.text_input("Symbol (BTC-USD, ETH-USD, AAPL, ISWD.L ...)", "BTC-USD")
with col2:
    interval = st.selectbox("Intervall", ["1h", "4h", "1d", "1wk"], index=2)
with col3:
    years = st.slider("Zeitraum (Jahre)", 1, 10, 3)

try:
    df = yf.download(symbol, period=f"{years}y", interval=interval)
    df = df.rename(columns=str.lower).dropna()

    if "close" not in df.columns:
        st.error("Fehler: Keine 'close'-Spalte gefunden.")
        st.write(df.head())
        st.stop()

    stoch = stoch_rsi_tv_like(df["close"])
    df = df.join(stoch).dropna()

    df = find_cross_signals(df)

    final_value, perf, trades = run_backtest(df)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["close"], label="Preis")
    ax.scatter(df.index[df["bull_cross"]], df["close"][df["bull_cross"]], color="green", marker="^", s=80, label="Buy")
    ax.scatter(df.index[df["bear_cross"]], df["close"][df["bear_cross"]], color="red", marker="v", s=80, label="Sell")
    ax.legend()
    ax.set_title(f"StochRSI Cross Signale – {symbol} ({interval})")

    st.pyplot(fig)

    st.success(f"💰 Endwert: **{final_value:,.2f} USD**  |  Gewinn: **{perf:.2f}%**")

    if trades:
        st.dataframe(pd.DataFrame(trades, columns=["Typ", "Datum", "Preis"]).set_index("Datum"))
    else:
        st.info("Keine Cross-Signale im Zeitraum.")

except Exception as e:
    st.error(f"Fehler: {e}")