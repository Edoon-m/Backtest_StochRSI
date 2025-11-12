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
    # --- Daten abrufen ---
    df = yf.download(symbol, period=f"{years}y", interval="1d", group_by=False, auto_adjust=True)

    if df.empty:
        st.warning("Keine Daten geladen – anderes Symbol probieren.")
    else:
        # Einige BTC-USD-Downloads haben Spalten wie ['BTC-USD','BTC-USD.1',...]
        # → Wir suchen die erste Spalte, die 'Close' enthält oder wie eine Kursreihe aussieht
        if "Close" in df.columns:
            close = df["Close"]
        elif symbol in df.columns:
            close = df[symbol]
        else:
            # falls yfinance MultiIndex liefert (z. B. [('BTC-USD', 'Close')])
            try:
                close = df.xs("Close", level=-1, axis=1)
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
            except Exception:
                raise KeyError(f"Keine gültige Close-Spalte gefunden: {list(df.columns)}")

        close = pd.to_numeric(close, errors="coerce").dropna()
        close.index = pd.to_datetime(close.index)

        # Monatsanfangsdaten
        monthly = close.resample("MS").first().dropna()

        # --- DCA-Logik ---
        units, invested = 0.0, 0.0
        records = []

        for i, (date, price) in enumerate(monthly.items()):
            if i == 0 and start_capital > 0:
                units += start_capital / price
                invested += start_capital
            if monthly_invest > 0:
                units += monthly_invest / price
                invested += monthly_invest
            value = units * price
            records.append([date, price, units, invested, value])

        hist = pd.DataFrame(records, columns=["Datum", "Preis", "Anteile", "Investiert", "Depotwert"]).set_index("Datum")

        if hist.empty:
            st.info("Keine ausreichenden Daten im gewählten Zeitraum.")
        else:
            total_invested = hist["Investiert"].iloc[-1]
            final_value = hist["Depotwert"].iloc[-1]
            profit = final_value - total_invested
            perf = (final_value / total_invested - 1) * 100 if total_invested else 0

            k1, k2, k3 = st.columns(3)
            k1.metric("Gesamt investiert", f"{total_invested:,.2f} €")
            k2.metric("Endwert", f"{final_value:,.2f} €")
            k3.metric("Rendite", f"{perf:.2f}%")

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