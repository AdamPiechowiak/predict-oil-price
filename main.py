import streamlit as st
import pandas as pd


data = pd.read_csv("oil_price.csv",parse_dates=True,index_col="Date")

st.write("""
## App to predict oil price
Shown are the stock prices and volume of Oil!
""")


st.write("""
## Open price
""")

st.line_chart(data["Open"])

st.write("""
## Close price
""")

st.line_chart(data["Close"])

st.write("""
## High price
""")

st.line_chart(data["High"])


st.write("""
## Low price
""")

st.line_chart(data["Low"])

st.write("""
## Volume
""")

st.line_chart(data["Volume"])


