from src.data_pipeline import load_stock_data, preprocess_data
from src.model_train import build_model
from src.model_eval import evaluate_model
from src.utils import plot_predictions

data = load_stock_data()

X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

model = build_model(X_train.shape[1:])

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

predictions = model.predict(X_test)

# Convert scaled values back to original stock prices
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

evaluate_model(actual, predictions)

plot_predictions(actual, predictions)

model.save("lstm_model.h5")

print("Model saved successfully!")
