import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="StochRSI Advanced Signale", layout="wide")

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


def compute_atr(df, length=14):
    """ATR auf Basis von High/Low/Close."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = rma(tr, length)
    return atr


def detect_cross(df):
    """Bull/Bear-Cross zwischen K und D finden."""
    bull = (df["K"].shift(1) < df["D"].shift(1)) & (df["K"] > df["D"])
    bear = (df["K"].shift(1) > df["D"].shift(1)) & (df["K"] < df["D"])

    df["bull_cross"] = bull.fillna(False).astype(bool)
    df["bear_cross"] = bear.fillna(False).astype(bool)
    return df


def run_backtest_advanced(
    df,
    k_buy_level=20,
    k_sell_level=80,
    atr_sl_mult=2.0,
    atr_tp_mult=3.0,
    min_atr_pct=0.0,
):
    """
    Erweiterte Strategie:

    - Bullish Cross wird NUR 'scharf', wenn er UNTER k_buy_level passiert.
      -> buy_armed = True
      -> Kauf erst, wenn K > k_buy_level UND Aufwärtstrend UND Volatilität ok.
    - Bearish Cross wird NUR 'scharf', wenn er ÜBER k_sell_level passiert.
      -> sell_armed = True
      -> Verkauf, wenn:
          * Kurs SL (ATR-basiert) trifft
          * Kurs TP (ATR-basiert) trifft
          * oder sell_armed & K < k_sell_level (normales Sell-Signal)

    - Trendfilter: nur Longs, wenn close > ema_trend
    - Volatilitätsfilter: nur Longs, wenn atr_pct >= min_atr_pct
    """

    start_capital = 10000.0
    cash = start_capital
    position = 0.0
    trades = []

    buy_armed = False
    sell_armed = False
    in_position = False
    sl = None
    tp = None

    if df.empty:
        return None, None, trades

    for row in df.itertuples():
        price = float(row.close)
        K = float(row.K)
        ema = float(row.ema_trend)
        atr = float(row.atr)
        atr_pct = float(row.atr_pct)

        bull_cross = bool(row.bull_cross)
        bear_cross = bool(row.bear_cross)

        uptrend = price > ema
        vol_ok = atr_pct >= min_atr_pct if not np.isnan(atr_pct) else False

        # --- Signale scharf machen ---
        if bull_cross and K < k_buy_level:
            buy_armed = True

        if bear_cross and K > k_sell_level:
            sell_armed = True

        # --- Wenn Position offen: Exits prüfen ---
        if in_position and position > 0:
            exit_reason = None

            # ATR-basierter Stop-Loss
            if sl is not None and price <= sl:
                exit_reason = "SL"

            # ATR-basiertes Take-Profit
            elif tp is not None and price >= tp:
                exit_reason = "TP"

            # Normales Sell-Signal (nach armed Bear-Cross + K < Level)
            elif sell_armed and K < k_sell_level:
                exit_reason = "Signal"

            if exit_reason:
                cash = position * price
                position = 0.0
                in_position = False
                sell_armed = False  # Verbrauchtes Sell-Signal
                trades.append(("SELL", row.Index, price, exit_reason))
                continue  # Nächste Kerze

        # --- Wenn KEINE Position offen: Einstiege prüfen ---
        if (not in_position) and cash > 0:
            # Einstieg nur bei:
            # - buy_armed
            # - K > k_buy_level (nach dem Cross)
            # - Aufwärtstrend (Trendfilter)
            # - ausreichender Volatilität
            # - ATR vorhanden
            if (
                buy_armed
                and K > k_buy_level
                and uptrend
                and vol_ok
                and not np.isnan(atr)
                and atr > 0
            ):
                position = cash / price
                cash = 0.0
                in_position = True

                sl = price - atr_sl_mult * atr
                tp = price + atr_tp_mult * atr

                buy_armed = False  # Signal verbraucht
                trades.append(("BUY", row.Index, price, "Entry"))
                continue

    last_price = float(df["close"].iloc[-1])
    final_value = cash + position * last_price
    perf = (final_value / start_capital - 1) * 100.0

    return final_value, perf, trades


def fix_yf_columns(df):
    """
    Repariert yfinance-Formate:
    - MultiIndex-Spalten
    - doppelte Ticker-Spalten (z.B. ['btc-usd', 'btc-usd', ...])
    - normalisiert auf ['open','high','low','close','adj close','volume'] o.ä.
    """

    # MultiIndex → innere Ebene
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    cols = [str(c).lower() for c in df.columns]

    # Wenn alle Spalten gleich heißen (nur Ticker)
    if len(set(cols)) == 1:
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

st.title("📉 StochRSI – Advanced Cross-Strategie")

symbol = st.text_input("Symbol (z.B. BTC-USD, ETH-USD, AAPL, ISWD.L ...)", "BTC-USD")
interval = st.selectbox("Intervall", ["1h", "4h", "1d", "1wk"], index=2)
years = st.slider("Zeitraum (Jahre)", 1, 10, 3)

col_levels = st.columns(2)
with col_levels[0]:
    k_buy_level = st.slider("K-Buy-Level (Oversold-Bereich)", 0, 50, 20)
with col_levels[1]:
    k_sell_level = st.slider("K-Sell-Level (Overbought-Bereich)", 50, 100, 80)

col_risk = st.columns(3)
with col_risk[0]:
    atr_sl_mult = st.slider("ATR-Multiplikator Stop-Loss", 0.5, 5.0, 2.0, 0.5)
with col_risk[1]:
    atr_tp_mult = st.slider("ATR-Multiplikator Take-Profit", 0.5, 10.0, 3.0, 0.5)
with col_risk[2]:
    min_atr_pct = st.slider("Min. ATR in % (Volatilitätsfilter)", 0.0, 5.0, 0.0, 0.1)

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

    needed_ohlc = {"open", "high", "low", "close"}
    if not needed_ohlc.issubset(df.columns):
        st.error(f"❌ Benötigte Spalten fehlen. Vorhanden: {list(df.columns)}")
        st.stop()

    close = df["close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # --------------------------------------------------
    # Indikatoren berechnen
    # --------------------------------------------------
    # StochRSI
    K, D = stoch_rsi(close)
    kd_df = pd.DataFrame({"K": K, "D": D})
    df = df.join(kd_df, how="left")

    # Trendfilter EMA200
    df["ema_trend"] = df["close"].ewm(span=200).mean()

    # ATR & ATR in %
    df["atr"] = compute_atr(df, length=14)
    df["atr_pct"] = df["atr"] / df["close"] * 100.0

    # Nur Zeilen mit vollständigen Daten
    required_cols = ["close", "K", "D", "ema_trend", "atr", "atr_pct"]
    df = df.dropna(subset=required_cols)
    if df.empty:
        st.error("⚠️ Zu wenige Daten für Indikator-Berechnung.")
        st.stop()

    # Cross-Signale
    df = detect_cross(df)

    # --------------------------------------------------
    # Backtest
    # --------------------------------------------------
    final, perf, trades = run_backtest_advanced(
        df,
        k_buy_level=float(k_buy_level),
        k_sell_level=float(k_sell_level),
        atr_sl_mult=float(atr_sl_mult),
        atr_tp_mult=float(atr_tp_mult),
        min_atr_pct=float(min_atr_pct),
    )

    # --------------------------------------------------
    # Plot: Kurs + Trades
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["close"], label="Kurs")

    buy_dates = [t[1] for t in trades if t[0] == "BUY"]
    sell_dates = [t[1] for t in trades if t[0] == "SELL"]

    if buy_dates:
        ax.scatter(
            buy_dates,
            df.loc[buy_dates, "close"],
            marker="^",
            s=80,
            label="Buy",
        )
    if sell_dates:
        ax.scatter(
            sell_dates,
            df.loc[sell_dates, "close"],
            marker="v",
            s=80,
            label="Sell",
        )

    ax.legend()
    ax.set_title(f"{symbol} – Advanced StochRSI-Trades ({interval})")
    st.pyplot(fig)

    # Ergebnis
    if final is None:
        st.error("⚠️ Keine Daten für Backtest.")
    else:
        st.success(f"Endwert: **{final:,.2f}** | Performance: **{perf:.2f}%**")

    # Trades anzeigen
    if trades:
        trade_df = pd.DataFrame(trades, columns=["Typ", "Datum", "Preis", "Grund"]).set_index("Datum")
        st.dataframe(trade_df)
    else:
        st.info("Keine Trades gefunden.")

except Exception as e:
    st.error(f"Fehler: {e}")