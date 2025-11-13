import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="StochRSI Cross – Clean Version", layout="wide")

# ---------------------------------------------------------
# StochRSI (TradingView Style)
# ---------------------------------------------------------
def rma(x, length):
    return x.ewm(alpha=1/length, adjust=False).mean()

def stoch_rsi(close, rsi_len=14, stoch_len=14, smoothK=3, smoothD=3):

    close = pd.Series(close.astype(float))

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

    K = stoch.rolling(smoothK).mean()
    D = K.rolling(smoothD).mean()

    return K, D

# ---------------------------------------------------------
# Cross detection (100% safe)
# ---------------------------------------------------------
def detect_cross(df):
    bull = (df["K"].shift(1) < df["D"].shift(1)) & (df["K"] > df["D"])
    bear = (df["K"].shift(1) > df["D"].shift(1)) & (df["K"] < df["D"])

    df["bull_cross"] = bull.fillna(False).astype(bool)
    df["bear_cross"] = bear.fillna(False).astype(bool)
    return df

# ---------------------------------------------------------
# Backtest (absolut safe)
# ---------------------------------------------------------
def run_backtest(df):

    cash = 10000
    position = 0
    trades = []

    for i in range(len(df)):
        price = float(df["close"].iloc[i])
        buy = bool(df["bull_cross"].iloc[i])
        sell = bool(df["bear_cross"].iloc[i])

        if buy and cash > 0:
            position = cash / price
            cash = 0
            trades.append(("BUY", df.index[i], price))

        if sell and position > 0:
            cash = position * price
            position = 0
            trades.append(("SELL", df.index[i], price))

    final = cash + position * float(df["close"].iloc[-1])
    perf = (final / 10000 - 1) * 100
    return final, perf, trades

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("📉 Clean StochRSI – Buy/Sell bei jedem Cross")

symbol = st.text_input("Symbol", "BTC-USD")
interval = st.selectbox("Intervall", ["1h", "4h", "1d", "1wk"], index=2)
years = st.slider("Zeitraum (Jahre)", 1, 10, 3)

try:
    df = yf.download(symbol, period=f"{years}y", interval=interval)
    df = df.rename(columns=str.lower)

    close = df["close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    K, D = stoch_rsi(close)
    df["K"] = K.values
    df["D"] = D.values
    df = df.dropna()

    df = detect_cross(df)

    final, perf, trades = run_backtest(df)

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df.index, df["close"])
    ax.scatter(df.index[df["bull_cross"]], df["close"][df["bull_cross"]], color="green", marker="^")
    ax.scatter(df.index[df["bear_cross"]], df["close"][df["bear_cross"]], color="red", marker="v")
    st.pyplot(fig)

    st.success(f"Endwert: {final:,.2f} USD | Performance: {perf:.2f}%")

    if trades:
        st.dataframe(pd.DataFrame(trades, columns=["Typ","Datum","Preis"]).set_index("Datum"))
    else:
        st.info("Keine Trades gefunden.")

except Exception as e:
    st.error(f"Fehler: {e}")