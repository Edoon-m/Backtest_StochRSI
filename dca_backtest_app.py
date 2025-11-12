import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Monatlicher DCA mit Startkapital", layout="wide")
st.title("💰 Monatlicher DCA (mit Startkapital)")

# Eingaben
c1, c2, c3 = st.columns(3)
with c1:
    symbol = st.text_input("Symbol (z. B. BTC-USD, ETH-USD, AAPL)", "BTC-USD")
with c2:
    years = st.slider("Zeitraum (Jahre)", 1, 20, 5)
with c3:
    st.caption("Käufe am Monatsanfang auf Basis des Schlusskurses")

c4, c5 = st.columns(2)
with c4:
    monthly_invest = st.number_input("Monatliche Investition (€)", min_value=0.0, value=200.0, step=50.0)
with c5:
    start_capital = st.number_input("Startkapital (einmalig, €)", min_value=0.0, value=0.0, step=100.0)

try:
    # --- Daten laden ---
    df = yf.download(symbol, period=f"{years}y", interval="1d", auto_adjust=True)

    if df.empty:
        st.warning("Keine Daten gefunden. Bitte anderes Symbol probieren.")
    else:
        # MultiIndex oder verschachtelte Spalten in einfache umwandeln
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        # Close-Spalte robust suchen
        close_candidates = [c for c in df.columns if "close" in c.lower() or symbol in c]
        if not close_candidates:
            raise KeyError(f"Keine Close-Spalte gefunden. Spalten: {list(df.columns)}")

        close = df[close_candidates[0]].squeeze()
        close = pd.to_numeric(close, errors='coerce').dropna()

        if not isinstance(close, pd.Series):
            close = pd.Series(close)

        close.index = pd.to_datetime(close.index)

        # Monatsanfangsdaten
        monthly = close.resample("MS").first().dropna()

        # --- DCA-Logik ---
        units = 0.0
        invested = 0.0
        records = []

        for i, (date, price) in enumerate(monthly.items()):
            price = float(price)
            if i == 0 and start_capital > 0:
                units += start_capital / price
                invested += start_capital
            if monthly_invest > 0:
                units += monthly_invest / price
                invested += monthly_invest
            value = units * price
            records.append([date, price, units, invested, value])

        hist = pd.DataFrame(records, columns=["Datum", "Preis", "Anteile", "Investiert", "Depotwert"]).set_index("Datum")

        # Ergebnisse
        total_invested = hist["Investiert"].iloc[-1]
        final_value = hist["Depotwert"].iloc[-1]
        profit = final_value - total_invested
        perf = (final_value / total_invested - 1) * 100 if total_invested > 0 else 0.0

        # KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("Gesamt investiert", f"{total_invested:,.2f} €")
        c2.metric("Endwert", f"{final_value:,.2f} €")
        c3.metric("Rendite", f"{perf:.2f}%")

        # Chart
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(hist.index, hist["Depotwert"], label="Depotwert")
        ax.plot(hist.index, hist["Investiert"], "--", label="Kumuliert investiert")
        ax.set_title(f"{symbol} – DCA über {years} Jahre | "
                     f"{monthly_invest:.0f} €/Monat, Start {start_capital:.0f} €")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Kaufhistorie")
        st.dataframe(hist.round(2))

except Exception as e:
    st.error(f"Fehler: {e}")