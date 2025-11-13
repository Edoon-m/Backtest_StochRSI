import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="StochRSI Signale", layout="wide")

# ----------------- Hilfsfunktionen -----------------

def rma(x, length):
    """Wilder's RMA (wie bei TradingView)."""
    return x.ewm(alpha=1 / length, adjust=False).mean()


def stoch_rsi(close,
              rsi_len=14,
              stoch_len=14,
              smoothK=3,
              smoothD=3):
    """TradingView-ähnlicher StochRSI (K & D)."""

    close = pd.Series(close.astype(float))

    # RSI (Wilder)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    avg_gain = rma(up, rsi_len)
    avg_loss = rma(down, rsi_len)

    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - 100 / (1 + rs)

    # Stochastik auf RSI
    lowest = rsi.rolling(stoch_len).min()
    highest = rsi.rolling(stoch_len).max()
    stoch = (rsi - lowest) / (highest - lowest + 1e-12) * 100

    K = stoch.rolling(smoothK).mean()
    D = K.rolling(smoothD).mean()

    K.name = "K"
    D.name = "D"
    return K, D


def detect_cross(df: pd.DataFrame) -> pd.DataFrame:
    """Bull/Bear-Cross zwischen K und D finden."""
    bull = (df["K"].shift(1) < df["D"].shift(1)) & (df["K"] > df["D"])
    bear = (df["K"].shift(1) > df["D"].shift(1)) & (df["K"] < df["D"])

    df["bull_cross"] = bull.fillna(False).astype(bool)
    df["bear_cross"] = bear.fillna(False).astype(bool)
    return df


def run_backtest_filtered(df: pd.DataFrame,
                          k_buy_level: float = 20,
                          k_sell_level: float = 80):
    """
    Strategie:
    - Bullish Cross wird NUR 'scharf', wenn er UNTER k_buy_level (z.B. 20) passiert.
      -> buy_armed = True
      -> Kauf erfolgt erst, wenn K > k_buy_level.
    - Bearish Cross wird NUR 'scharf', wenn er ÜBER k_sell_level (z.B. 80) passiert.
      -> sell_armed = True
      -> Verkauf erfolgt erst, wenn K < k_sell_level.
    """

    start_capital = 10000.0
    cash = start_capital
    position = 0.0
    trades: list[tuple[str, pd.Timestamp, float]] = []

    buy_armed = False
    sell_armed = False

    if df.empty:
        return None, None, trades

    # Jede Zeile als NamedTuple
    for row in df.itertuples():
        price = float(row.close)
        K = float(row.K)

        bull_cross = bool(row.bull_cross)
        bear_cross = bool(row.bear_cross)

        # --- Signale "scharf machen" NUR in Extremzone ---
        if bull_cross and K < k_buy_level:
            buy_armed = True
        if bear_cross and K > k_sell_level:
            sell_armed = True

        # --- Einstieg: nach bullischem Cross, wenn K > k_buy_level ---
        if cash > 0 and buy_armed and K > k_buy_level:
            position = cash / price
            cash = 0.0
            buy_armed = False
            trades.append(("BUY", row.Index, price))

        # --- Ausstieg: nach bearish Cross, wenn K < k_sell_level ---
        if position > 0 and sell_armed and K < k_sell_level:
            cash = position * price
            position = 0.0
            sell_armed = False
            trades.append(("SELL", row.Index, price))

    last_price = float(df["close"].iloc[-1])
    final_value = cash + position * last_price
    perf = (final_value / start_capital - 1) * 100.0

    return final_value, perf, trades


def fix_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Repariert ALLE yfinance-Formate:
    - MultiIndex-Spalten
    - doppelte Ticker-Spalten (z.B. ['btc-usd', ...])
    - normalisiert auf ['open','high','low','close','adj close','volume'] oder ähnlich
    """

    # Fall 1: MultiIndex → nimm die innere Ebene
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    cols = [str(c).lower() for c in df.columns]

    # Fall 2: alle Spalten identisch (nur Ticker-Name)
    if len(set(cols)) == 1:
        # Versuche anhand der Spaltenanzahl OHLCV zuzuweisen
        if df.shape[1] >= 6:
            df.columns = ["open", "high", "low", "close", "adj close", "volume"]
        elif df.shape[1] == 5:
            df.columns = ["open", "high", "low", "close", "volume"]
        else:
            raise ValueError(f"Unbekanntes Spaltenformat von yfinance: {df.columns}")
    else:
        df.columns = cols

    return df


# ----------------- UI -----------------

st.title("📉 StochRSI Signale – gefilterte Cross-Strategie")

symbol = st.text_input("Symbol", "BTC-USD")
interval = st.selectbox("Intervall", ["1h", "4h", "1d", "1wk"], index=2)
years = st.slider("Zeitraum (Jahre)", 1, 10, 3)

col_a, col_b = st.columns(2)
with col_a:
    k_buy_level = st.slider("K-Buy-Level (Oversold-Bereich)", 0, 50, 20)
with col_b:
    k_sell_level = st.slider("K-Sell-Level (Overbought-Bereich)", 50, 100, 80)

try:
    # --------------------------------------------------
    # Daten laden
    # --------------------------------------------------
    df = yf.download(symbol, period=f"{years}y", interval=interval)

    if df.empty:
        st.error("⚠️ Für dieses Symbol/Intervall konnten keine Daten geladen werden.")
        st.stop()

    df = df.copy()
    df = fix_yf_columns(df)

    if "close" not in df.columns:
        st.error(f"❌ Keine 'close'-Spalte gefunden. Spalten nach Fix: {list(df.columns)}")
        st.stop()

    close = df["close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # --------------------------------------------------
    # StochRSI berechnen und andocken
    # --------------------------------------------------
    K, D = stoch_rsi(close)

    kd_df = pd.DataFrame({"K": K, "D": D})
    df = df.join(kd_df, how="left")

    required_cols = ["close", "K", "D"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"❌ Fehlende Spalten nach Join: {missing}")
        st.stop()

    df = df.dropna(subset=required_cols)
    if df.empty:
        st.error("⚠️ Zu wenige Daten für StochRSI-Berechnung.")
        st.stop()

    # --------------------------------------------------
    # Cross-Signale & Backtest
    # --------------------------------------------------
    df = detect_cross(df)

    final, perf, trades = run_backtest_filtered(
        df,
        k_buy_level=float(k_buy_level),
        k_sell_level=float(k_sell_level),
    )

    # --------------------------------------------------
    # Plot: Kurs + Trades
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["close"], label="Kurs")

    buy_dates = [t[1] for t in trades if t[0] == "BUY"]
    sell_dates = [t[1] for t in trades if t[0] == "SELL"]

    if buy_dates:
        ax.scatter(buy_dates,
                   df.loc[buy_dates, "close"],
                   marker="^", s=80, label="Buy")
    if sell_dates:
        ax.scatter(sell_dates,
                   df.loc[sell_dates, "close"],
                   marker="v", s=80, label="Sell")

    ax.legend()
    ax.set_title(f"{symbol} – gefilterte StochRSI-Trades ({interval})")
    st.pyplot(fig)

    # Ergebnis
    if final is None:
        st.error("⚠️ Keine Daten für Backtest.")
    else:
        st.success(f"Endwert: **{final:,.2f}** | Performance: **{perf:.2f}%**")

    # Trades anzeigen
    if trades:
        trade_df = pd.DataFrame(trades, columns=["Typ", "Datum", "Preis"]).set_index("Datum")
        st.dataframe(trade_df)
    else:
        st.info("Keine Trades gefunden.")

except Exception as e:
    st.error(f"Fehler: {e}")