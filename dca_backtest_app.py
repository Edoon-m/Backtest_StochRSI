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
    # --- Kursdaten laden ---
    df = yf.download(symbol, period=f"{years}y", interval="1d", group_by="column")

    if df.empty:
        st.warning("Keine Daten geladen – anderes Symbol probieren.")
    else:
        # Falls MultiIndex-Spalten vorhanden → flach machen
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

        # Close-Spalte finden
        cols = [c for c in df.columns if "close" in c.lower()]
        if not cols:
            raise KeyError(f"Keine 'Close'-Spalte gefunden: {list(df.columns)}")
        close = df[cols[0]].astype(float)

        # Monatsanfang-Daten (erster Handelstag des Monats)
        monthly = close.resample("MS").first().dropna()

        # --- DCA-Berechnung ---
        units = 0.0
        invested = 0.0
        history = []

        for i, (date, price) in enumerate(monthly.items()):
            price = float(price)
            if i == 0 and start_capital > 0:
                units += start_capital / price
                invested += start_capital
            if monthly_invest > 0:
                units += monthly_invest / price
                invested += monthly_invest
            depotwert = units * price
            history.append([date, price, units, invested, depotwert])

        hist = pd.DataFrame(history, columns=["Datum", "Preis", "Anteile", "Investiert", "Depotwert"]).set_index("Datum")

        if hist.empty:
            st.info("Zu wenige Daten im gewählten Zeitraum.")
        else:
            total_invested = hist["Investiert"].iloc[-1]
            final_value = hist["Depotwert"].iloc[-1]
            profit = final_value - total_invested
            perf = (final_value / total_invested - 1) * 100 if total_invested > 0 else 0

            # KPIs anzeigen
            k1, k2, k3 = st.columns(3)
            k1.metric("Gesamt investiert", f"{total_invested:,.2f} €")
            k2.metric("Endwert", f"{final_value:,.2f} €")
            k3.metric("Rendite", f"{perf:.2f}%")

            # Plot
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(hist.index, hist["Depotwert"], label="Depotwert")
            ax.plot(hist.index, hist["Investiert"], label="Kumuliert investiert", linestyle="--")
            ax.set_title(f"{symbol} – DCA über {years} Jahre | "
                         f"{monthly_invest:.0f} €/Monat, Start {start_capital:.0f} €")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Kaufhistorie (Monatsanfang)")
            st.dataframe(hist.round(2))

except Exception as e:
    st.error(f"Fehler: {e}")