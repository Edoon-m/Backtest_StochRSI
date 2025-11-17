import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="S1 – Trendfolge Pullback", layout="wide")


# ---------- Helper ----------

def fix_yf_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    cols = [str(c).lower() for c in df.columns]
    if len(set(cols)) == 1:
        if df.shape[1] >= 6:
            df.columns = ["open", "high", "low", "close", "adj close", "volume"]
        elif df.shape[1] == 5:
            df.columns = ["open", "high", "low", "close", "volume"]
        else:
            raise ValueError(f"Unbekanntes Spaltenformat: {df.columns}")
    else:
        df.columns = cols
    return df

def rma(x, length):
    return x.ewm(alpha=1/length, adjust=False).mean()

def compute_atr(df, length=14):
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return rma(tr, length)


def backtest_trend_pullback(df, ema_trend_len=200, ema_fast_len=20,
                            atr_len=14, atr_sl_mult=2.0):

    df = df.copy()
    df["ema_trend"] = df["close"].ewm(span=ema_trend_len).mean()
    df["ema_fast"] = df["close"].ewm(span=ema_fast_len).mean()
    df["atr"] = compute_atr(df, atr_len)

    df = df.dropna(subset=["close", "ema_trend", "ema_fast", "atr"])
    if df.empty:
        return None, None, None, []

    cash = 10000.0
    position = 0.0
    entry_price = None
    sl = None

    equity = []
    trades = []
    in_pos = False

    prev_close = None
    prev_ema_fast = None

    for row in df.itertuples():
        price = float(row.close)
        ema_trend = float(row.ema_trend)
        ema_fast = float(row.ema_fast)
        atr = float(row.atr)

        uptrend = price > ema_trend

        # Ausstieg, wenn in Position
        if in_pos and position > 0:
            exit_reason = None
            if sl is not None and price <= sl:
                exit_reason = "SL"
            elif price < ema_fast:
                exit_reason = "EMA_fast_break"

            if exit_reason:
                cash = position * price
                position = 0.0
                in_pos = False
                trades.append(("SELL", row.Index, price, exit_reason))

        # Einstieg, wenn NICHT in Position
        if (not in_pos) and cash > 0 and uptrend:
            # Pullback-Bedingung:
            # vorher unter EMA20, jetzt darüber
            if prev_close is not None and prev_ema_fast is not None:
                crossed_up = (prev_close < prev_ema_fast) and (price > ema_fast)
            else:
                crossed_up = False

            if crossed_up and not np.isnan(atr) and atr > 0:
                position = cash / price
                cash = 0.0
                in_pos = True
                entry_price = price
                sl = price - atr_sl_mult * atr
                trades.append(("BUY", row.Index, price, "Pullback"))

        # Equity berechnen
        cur_equity = cash + position * price
        equity.append((row.Index, cur_equity))

        # für nächste Iteration
        prev_close = price
        prev_ema_fast = ema_fast

    final_equity = equity[-1][1]
    perf = (final_equity / 10000 - 1) * 100

    eq_df = pd.DataFrame(equity, columns=["Datum", "Equity"]).set_index("Datum")
    return final_equity, perf, eq_df, trades


# ---------- UI ----------

st.title("📈 S1 – Trendfolge mit Pullback (EMA200 + EMA20)")

symbol = st.text_input("Symbol", "BTC-USD")
interval = st.selectbox("Intervall", ["1d", "4h", "1h", "1wk"], index=0)
years = st.slider("Zeitraum (Jahre)", 1, 10, 5)

col1, col2 = st.columns(2)
with col1:
    ema_trend_len = st.slider("Trend-EMA (z.B. 200)", 50, 300, 200, 10)
    ema_fast_len = st.slider("Pullback-EMA (z.B. 20)", 5, 50, 20, 1)
with col2:
    atr_len = st.slider("ATR-Länge", 5, 50, 14, 1)
    atr_sl_mult = st.slider("ATR Stop-Loss x", 0.5, 5.0, 2.0, 0.5)

try:
    df = yf.download(symbol, period=f"{years}y", interval=interval)
    if df.empty:
        st.error("Keine Daten geladen.")
        st.stop()

    df = fix_yf_columns(df)
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(df.columns):
        st.error(f"Fehlende OHLC-Spalten, vorhanden: {df.columns}")
        st.stop()

    final_val, perf, eq_df, trades = backtest_trend_pullback(
        df,
        ema_trend_len=ema_trend_len,
        ema_fast_len=ema_fast_len,
        atr_len=atr_len,
        atr_sl_mult=atr_sl_mult,
    )

    if final_val is None:
        st.error("Backtest nicht möglich.")
        st.stop()

    # Equity-Chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eq_df.index, eq_df["Equity"], label="Equity")
    ax.set_title(f"Equity Curve – Endwert: {final_val:,.2f} | {perf:.2f}%")
    ax.legend()
    st.pyplot(fig)

    # Preis + Trades
    df_close = df.loc[eq_df.index]  # gleiche Range
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df_close.index, df_close["close"], label="Kurs")

    buy_dates = [t[1] for t in trades if t[0] == "BUY"]
    sell_dates = [t[1] for t in trades if t[0] == "SELL"]
    if buy_dates:
        ax2.scatter(buy_dates, df_close.loc[buy_dates, "close"], marker="^", s=60, label="Buy")
    if sell_dates:
        ax2.scatter(sell_dates, df_close.loc[sell_dates, "close"], marker="v", s=60, label="Sell")
    ax2.legend()
    ax2.set_title(f"{symbol} – Trades im Chart")
    st.pyplot(fig2)

    st.success(f"Endwert: {final_val:,.2f}  |  Performance: {perf:.2f}%")

    if trades:
        trades_df = pd.DataFrame(trades, columns=["Typ", "Datum", "Preis", "Grund"]).set_index("Datum")
        st.dataframe(trades_df)
    else:
        st.info("Keine Trades.")

except Exception as e:
    st.error(f"Fehler: {e}")