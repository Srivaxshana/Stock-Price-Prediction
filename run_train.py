"""
run_train.py
Simple script to train an LSTM model for a selected stock from `all_stocks_5yr.csv`.
Usage:
    python run_train.py --symbol AAPL --epochs 30
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def train(symbol='AAPL', sequence_length=30, epochs=20, batch_size=32, model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)
    df = pd.read_csv('all_stocks_5yr.csv')
    df['date'] = pd.to_datetime(df['date'])
    df_symbol = df[df['Name'] == symbol].sort_values('date')
    prices = df_symbol['close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)

    X, y = create_sequences(prices_scaled, sequence_length)
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    model_path = os.path.join(model_dir, f'lstm_{symbol}.h5')
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    mc = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[es, mc])

    scaler_path = os.path.join(model_dir, f'scaler_{symbol}.bin')
    joblib.dump(scaler, scaler_path)
    print('Training complete.')
    print('Model saved to:', model_path)
    print('Scaler saved to:', scaler_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default='AAPL')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--sequence_length', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    args = p.parse_args()
    train(symbol=args.symbol, sequence_length=args.sequence_length, epochs=args.epochs, batch_size=args.batch_size)
