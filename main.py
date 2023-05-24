import streamlit as st
import yfinance as yf
import datetime
import pandas as pd
import numpy as np
from keras.layers import GRU, Input, LSTM, Dropout
from keras.models import Model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from keras.layers import TimeDistributed, Dense
from keras.utils.vis_utils import plot_model
import copy


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


