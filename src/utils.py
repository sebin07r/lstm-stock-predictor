def plot_predictions(actual, predicted):
    plt.figure(figsize=(12,6))

    plt.plot(actual, label="Actual Price", color="blue")
    plt.plot(predicted, label="Predicted Price", color="red")

    plt.title("LSTM Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)

    plt.show()
