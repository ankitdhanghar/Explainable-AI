import numpy as np
from sklearn.metrics import mean_absolute_error,r2_score

def evaluate_model(model,model_name,X_test, y_test,config):
    if model_name =='ANN':
        predictions = model.predict(X_test).flatten()
        y_test = np.array(y_test).flatten()
    else:
         predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(
        ((y_test - predictions)**2).mean()
    )
    r2 = r2_score(y_test, predictions)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }