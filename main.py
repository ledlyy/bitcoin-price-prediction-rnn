import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional
import matplotlib.dates as mdates

# Global parameters
TIME_STEP = 5
MODEL_PATH = 'models/model.h5'
DATA_PATH = 'data/BTC-Daily.csv'
FEATURES = ['open', 'high', 'low', 'close']
VOLUME_COLUMNS = ['Volume BTC', 'Volume USD']


def load_data():
    data = pd.read_csv(DATA_PATH)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data.sort_index(inplace=True)
    return data


def get_features(data):
    for vol in VOLUME_COLUMNS:
        if vol in data.columns:
            volume_column = vol
            break
    else:
        raise KeyError("No volume column found. Check column names.")
    
    selected = FEATURES + [volume_column]
    return data[selected]


def normalize_data(train_values, test_values):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_values)
    scaled_test = scaler.transform(test_values)
    return scaled_train, scaled_test, scaler


def create_dataset(data, time_step=TIME_STEP):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        Y.append(data[i + time_step, 3])  # 'close' is at index 3
    return np.array(X), np.array(Y)


def build_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(100, return_sequences=True, input_shape=input_shape)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(GRU(100, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(LSTM(100))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train():
    data = load_data()
    data = get_features(data)
    train_data = data['2014':'2020']
    test_data = data['2020':]

    train_values = train_data.values
    test_values = test_data.values

    scaled_train, scaled_test, scaler = normalize_data(train_values, test_values)
    X_train, y_train = create_dataset(scaled_train)

    model = build_model((TIME_STEP, X_train.shape[2]))
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")


def predict():
    data = load_data()
    data = get_features(data)
    train_data = data['2014':'2020']
    test_data = data['2020':]

    train_values = train_data.values
    test_values = test_data.values

    scaled_train, scaled_test, scaler = normalize_data(train_values, test_values)
    X_test, y_test = create_dataset(scaled_test)

    model = load_model(MODEL_PATH)
    predictions = model.predict(X_test)

    # Inverse scale
    dummy_full = np.zeros((len(predictions), X_test.shape[2]))
    dummy_full[:, 3] = predictions[:, 0]  # Only 'close'
    predicted_values = scaler.inverse_transform(dummy_full)[:, 3]

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['close'], color='red', label='Actual BTC Prices')
    plt.plot(test_data.index[TIME_STEP:], predicted_values, color='blue', label='Predicted BTC Prices')

    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('BTC Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.xlim([pd.Timestamp('2020-01-01'), pd.Timestamp('2022-12-31')])
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bitcoin Price Prediction using RNN')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'],
                        help='Mode to run: train or predict')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'predict':
        predict()