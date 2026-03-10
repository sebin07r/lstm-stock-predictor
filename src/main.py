from src.data_pipeline import load_stock_data, preprocess_data
from src.model_train import build_model

data = load_stock_data()

X, y, scaler = preprocess_data(data)

model = build_model((X.shape[1],1))

model.fit(X, y, epochs=10, batch_size=32)

model.save("lstm_model.h5")
