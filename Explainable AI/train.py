from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
from scikeras.wrappers import KerasRegressor

def train_rf(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

def train_xgb(X_train, y_train):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model

def build_ann(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mse'
    )
    return model

def train_ann(X_train, y_train, config):
    
    model = KerasRegressor(
    model=build_ann,
    input_dim=X_train.shape[1],
    epochs=config.ann_epochs,
    batch_size=config.ann_batch_size,
    verbose=0
)
    model.fit(
        X_train,
        y_train
    )
    return model