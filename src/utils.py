import matplotlib.pyplot as plt

def plot_predictions(actual, predicted):

    plt.figure(figsize=(10,5))

    plt.plot(actual, label="Actual Price")
    plt.plot(predicted, label="Predicted Price")

    plt.title("Stock Price Prediction using LSTM")
    plt.xlabel("Time")
    plt.ylabel("Price")

    plt.legend()
    plt.show()
