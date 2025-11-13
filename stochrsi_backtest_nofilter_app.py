import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="TradingView StochRSI – Buy/Sell", layout="wide")

# ---------------------------------------------------------
# TradingView Wilder RSI + StochRSI
# ---------------------------------------------------------
def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1/length, adjust=False).mean()

def stoch_rsi_tv(close, rsi_len=14, stoch_len=14, k=3, d=3):
    close = close.astype(float)
    delta = close.diff()

    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    avg_gain = rma(up, rsi_len)
    avg_loss = rma(down, rsi_len)

    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - 100 / (1 + rs)

    lowest = rsi.rolling(stoch_len).min()
    highest = rsi.rolling(stoch_len).max()

    stoch = (rsi - lowest) / (highest - lowest + 1e-12) * 100

    K = stoch.rolling(k).mean()
    D = K.rolling(d).mean()

    return K, D

# ---------------------------------------------------------
# Cross Signale
# ---------------------------------------------------------
def detect_crosses(df):
    df["bull_cross"] = (df["K"].shift(1) < df["D"].shift(1)) & (df["K"] > df["D"])
    df["bear_cross"] = (df["K"].shift(1) > df["D"].shift(1)) & (df["K"] < df["D"])
    return df

# ---------------------------------------------------------
# Backtest
# ---------------------------------------------------------
def run_backtest(df):
    cash = 10000
    position = 0
    trades = []

    for i in range(len(df)):
        price = df["close"].iloc[i]
        row = df.iloc[i]

        if bool(row["bull_cross"]) and cash > 0:
            position = cash / price
            cash = 0
            trades.append(("BUY", df.index[i], price))

        if bool(row["bear_cross"]) and position > 0:
            cash = position * price
            position = 0
            trades.append(("SELL", df.index[i], price))

    final_value = cash + position * df["close"].iloc[-1]
    gain = (final_value / 10000 - 1) * 100
    return final_value, gain, trades

# ---------------------------------------------------------
st.title("📊 TradingView-StochRSI – Buy/Sell bei jedem Cross")

col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.text_input("Symbol (BTC-USD, ETH-USD, AAPL, ISWD.L ...)", "BTC-USD")
with col2:
    interval = st.selectbox("Intervall", ["1h", "4h", "1d", "1wk"], index=2)
with col3:
    years = st.slider("Zeitraum (Jahre)", 1, 10, 3)

try:
    df = yf.download(symbol, period=f"{years}y", interval=interval)
    df = df.rename(columns=str.lower)

    if df.empty:
        st.error("Keine Daten geladen!")
        st.stop()

    # Close-Spalte sicherstellen
    close = df["close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)

    # TradingView StochRSI
    K, D = stoch_rsi_tv(close)
    df["K"] = K.values
    df["D"] = D.values

    df = df.dropna()
    df = detect_crosses(df)

    # Backtest
    final_value, perf, trades = run_backtest(df)

    # Chart
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df.index, df["close"], label="Kurs")
    ax.scatter(df.index[df["bull_cross"]], df["close"][df["bull_cross"]],
               marker="^", color="green", s=80, label="Buy")
    ax.scatter(df.index[df["bear_cross"]], df["close"][df["bear_cross"]],
               marker="v", color="red", s=80, label="Sell")
    ax.legend()
    st.pyplot(fig)

    st.success(f"💰 Endwert: **{final_value:,.2f} USD** | Gewinn: **{perf:.2f}%**")

    if trades:
        st.dataframe(pd.DataFrame(trades, columns=["Typ", "Datum", "Preis"]).set_index("Datum"))
    else:
        st.info("Keine Signale gefunden.")

except Exception as e:
    st.error(f"Fehler: {e}")