import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_stock_data(ticker="AAPL"):
    data = yf.download(ticker, start="2015-01-01", end="2024-01-01")
    return data['Close'].values.reshape(-1,1)

def preprocess_data(data, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(data)

    X, y = [], []

    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i])
        y.append(scaled[i])

    return np.array(X), np.array(y), scaler
