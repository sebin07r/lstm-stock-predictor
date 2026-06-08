def preprocess_data(data, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0,1))
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
