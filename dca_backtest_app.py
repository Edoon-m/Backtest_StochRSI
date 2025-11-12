import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Monatlicher DCA-Simulator", layout="wide")
st.title("💰 Monatlicher DCA (mit Startkapital)")

# Eingaben
c1, c2, c3 = st.columns(3)
with c1:
    symbol = st.text_input("Symbol (z. B. BTC-USD, ETH-USD, AAPL)", "BTC-USD")
with c2:
    years = st.slider("Zeitraum (Jahre)", 1, 20, 5)
with c3:
    interval_info = st.caption("Datenbasis: Tageskurs, Käufe am Monatsanfang")

c4, c5 = st.columns(2)
with c4:
    monthly_invest = st.number_input("Monatliche Investition (€)", min_value=0.0, value=200.0, step=50.0)
with c5:
    start_capital = st.number_input("Startkapital (einmalig, €)", min_value=0.0, value=0.0, step=100.0)

try:
    # Tagesdaten laden
    df = yf.download(symbol, period=f"{years}y", interval="1d")
    if df is None or df.empty:
        st.warning("Keine Daten geladen – anderes Symbol/Zeitraum probieren.")
    else:
        df = df.rename(columns=str.lower)
        df.index = pd.to_datetime(df.index)
        close = df["close"].dropna()

        # Monatsanfang (MS = Month Start) -> erster handelbarer Tag des Monats
        monthly = close.resample("MS").first().dropna()

        units = 0.0
        invested = 0.0
        rows = []

        for i, (date, price) in enumerate(monthly.items()):
            # beim allerersten Monat: ggf. Startkapital investieren
            if i == 0 and start_capital > 0:
                units += start_capital / price
                invested += start_capital

            # jeden Monat: fixer DCA-Betrag
            if monthly_invest > 0:
                units += monthly_invest / price
                invested += monthly_invest

            depotwert = units * price
            rows.append([date, price, units, invested, depotwert])

        hist = pd.DataFrame(rows, columns=["Datum", "Preis", "Anteile", "Investiert", "Depotwert"]).set_index("Datum")

        if hist.empty:
            st.info("Zu wenige Daten für den gewählten Zeitraum.")
        else:
            final_value = float(hist["Depotwert"].iloc[-1])
            total_invested = float(hist["Investiert"].iloc[-1])
            profit = final_value - total_invested
            perf_pct = (final_value / total_invested - 1) * 100 if total_invested > 0 else 0.0
            avg_cost = (total_invested / hist["Anteile"].iloc[-1]) if hist["Anteile"].iloc[-1] > 0 else np.nan
            last_price = float(hist["Preis"].iloc[-1])

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Gesamt investiert", f"{total_invested:,.2f} €")
            k2.metric("Endwert", f"{final_value:,.2f} €")
            k3.metric("Gewinn/Verlust", f"{profit:,.2f} €", f"{perf_pct:.2f}%")
            k4.metric("Ø Einstand / Letzter Preis", f"{avg_cost:,.2f} €", f"{last_price:,.2f} €")

            # Chart: Depotwert vs. kumuliertes Investment
            fig, ax = plt.subplots(figsize=(12,5))
            ax.plot(hist.index, hist["Depotwert"], label="Depotwert")
            ax.plot(hist.index, hist["Investiert"], label="Kumuliert investiert", linestyle="--")
            ax.set_title(f"{symbol} – DCA über {years} Jahre | {monthly_invest:.0f} €/Monat, Start {start_capital:.0f} €")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Kaufhistorie (Monatsanfang)")
            st.dataframe(hist.round(2))

except Exception as e:
    st.error(f"Fehler: {e}")