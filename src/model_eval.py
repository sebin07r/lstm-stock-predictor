from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_model(y_true, y_pred):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("RMSE:", rmse)

    return rmse
