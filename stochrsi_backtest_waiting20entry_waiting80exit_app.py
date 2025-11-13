import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Clean StochRSI – Cross Trades", layout="wide")

# ----------------- Hilfsfunktionen -----------------

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

def detect_cross(df):
    bull = (df["K"].shift(1) < df["D"].shift(1)) & (df["K"] > df["D"])
    bear = (df["K"].shift(1) > df["D"].shift(1)) & (df["K"] < df["D"])

    df["bull_cross"] = bull.fillna(False).astype(bool)
    df["bear_cross"] = bear.fillna(False).astype(bool)
    return df

def run_backtest_filtered(df, k_buy_level=20, k_sell_level=80):
    """
    Logik:
    - Bullish Cross wird NUR 'scharf', wenn er unter k_buy_level (z.B. 20) passiert.
      -> buy_armed = True
      -> Kauf erst, wenn K > k_buy_level.
    - Bearish Cross wird NUR 'scharf', wenn er über k_sell_level (z.B. 80) passiert.
      -> sell_armed = True
      -> Verkauf erst, wenn K < k_sell_level.
    """

    start_capital = 10000.0
    cash = start_capital
    position = 0.0
    trades = []

    buy_armed = False   # wartet auf K > k_buy_level
    sell_armed = False  # wartet auf K < k_sell_level

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        K = float(row["K"])

        bull_cross = bool(row["bull_cross"])
        bear_cross = bool(row["bear_cross"])

        # --- Signale "scharf machen" NUR in Extremzone ---
        # Bullish Cross nur zählen, wenn er UNTER k_buy_level (z.B. 20) passiert
        if bull_cross and K < k_buy_level:
            buy_armed = True

        # Bearish Cross nur zählen, wenn er ÜBER k_sell_level (z.B. 80) passiert
        if bear_cross and K > k_sell_level:
            sell_armed = True

        # --- Einstieg: erst NACH dem Cross, wenn K > k_buy_level ---
        if cash > 0 and buy_armed and K > k_buy_level:
            position = cash / price
            cash = 0.0
            buy_armed = False   # Signal verbraucht
            trades.append(("BUY", df.index[i], price))

        # --- Ausstieg: erst NACH dem Cross, wenn K < k_sell_level ---
        if position > 0 and sell_armed and K < k_sell_level:
            cash = position * price
            position = 0.0
            sell_armed = False  # Signal verbraucht
            trades.append(("SELL", df.index[i], price))

    # Depotwert am Ende
    if len(df) == 0:
        return None, None, trades

    final_value = cash + position * float(df["close"].iloc[-1])
    perf = (final_value / start_capital - 1) * 100.0

    return final_value, perf, trades


# ----------------- UI -----------------

st.title("📉 Clean StochRSI – Buy/Sell bei gefilterten Crosses")

symbol = st.text_input("Symbol", "BTC-USD")
interval = st.selectbox("Intervall", ["1h", "4h", "1d", "1wk"], index=2)
years = st.slider("Zeitraum (Jahre)", 1, 10, 3)

# Levels einstellbar machen (optional)
col_a, col_b = st.columns(2)
with col_a:
    k_buy_level = st.slider("K-Buy-Level (Oversold-Bereich)", 0, 50, 20)
with col_b:
    k_sell_level = st.slider("K-Sell-Level (Overbought-Bereich)", 50, 100, 80)

try:
    df = yf.download(symbol, period=f"{years}y", interval=interval)

    if df.empty:
        st.error("⚠️ Für dieses Symbol/Intervall konnten keine Daten geladen werden.")
        st.stop()

    df = df.rename(columns=str.lower)

    close = df["close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # StochRSI berechnen
    K, D = stoch_rsi(close)
    df["K"] = K.values
    df["D"] = D.values

    df = df.dropna()
    if df.empty:
        st.error("⚠️ Zu wenige Daten für StochRSI-Berechnung.")
        st.stop()

    # Cross-Signale
    df = detect_cross(df)

    # Backtest mit deiner erweiterten Logik
    final, perf, trades = run_backtest_filtered(df, k_buy_level=k_buy_level, k_sell_level=k_sell_level)

    # Plot: Kurs + tatsächliche Buy/Sell-Trades
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["close"], label="Kurs")

    # tatsächliche Trades (nicht alle Crosses)
    buy_dates = [t[1] for t in trades if t[0] == "BUY"]
    sell_dates = [t[1] for t in trades if t[0] == "SELL"]

    if buy_dates:
        ax.scatter(buy_dates, df.loc[buy_dates, "close"], color="green", marker="^", s=80, label="Buy")
    if sell_dates:
        ax.scatter(sell_dates, df.loc[sell_dates, "close"], color="red", marker="v", s=80, label="Sell")

    ax.legend()
    ax.set_title(f"{symbol} – gefilterte StochRSI-Trades ({interval})")
    st.pyplot(fig)

    # Ergebnis anzeigen
    if final is None:
        st.error("⚠️ Keine Daten für Backtest.")
    else:
        st.success(f"Endwert: **{final:,.2f} USD** | Performance: **{perf:.2f}%**")

    # Trade-Tabelle
    if trades:
        trade_df = pd.DataFrame(trades, columns=["Typ", "Datum", "Preis"]).set_index("Datum")
        st.dataframe(trade_df)
    else:
        st.info("Keine Trades gefunden.")

except Exception as e:
    st.error(f"Fehler: {e}")