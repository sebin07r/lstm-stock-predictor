import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_stock_data(ticker="AAPL"):
    data = yf.download(
        ticker,
        start="2015-01-01",
        end="2024-01-01"
    )

    return data['Close'].values.reshape(-1, 1)


def preprocess_data(data, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    X, y = [], []

    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i])
        y.append(scaled[i])

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    return X_train, X_test, y_train, y_test, scaler
