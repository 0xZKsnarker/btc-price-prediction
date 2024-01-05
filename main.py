import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

crypto_currency = 'BTC'
against_currency = 'USD'

start = dt.datetime(2016, 2, 2)
end = dt.datetime.now()

# Using yfinance to fetch data
data = yf.download(f'{crypto_currency}-{against_currency}', start=start, end=end)

print(data.head())
