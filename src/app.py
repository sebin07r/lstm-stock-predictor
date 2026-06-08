import streamlit as st
import matplotlib.pyplot as plt
from src.data_pipeline import load_stock_data, preprocess_data
from src.model_train import build_model

st.title("📈 LSTM Stock Price Predictor")

ticker = st.text_input("Enter Stock Ticker", "AAPL")

if st.button("Run Prediction"):

    data = load_stock_data(ticker)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

    model = build_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    preds = model.predict(X_test)

    preds = scaler.inverse_transform(preds)
    actual = scaler.inverse_transform(y_test.reshape(-1,1))

    fig, ax = plt.subplots()
    ax.plot(actual, label="Actual")
    ax.plot(preds, label="Predicted")
    ax.legend()

    st.pyplot(fig)
