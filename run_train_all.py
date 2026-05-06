"""
run_train_all.py
----------------
Downloads live data for top 50 S&P 500 companies using yfinance,
trains one general LSTM model on all of them, and saves it to
models/lstm_general.keras

Run:
    pip install yfinance
    python run_train_all.py

Training takes approximately 30-60 minutes depending on your machine.
"""

import math
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
LOOKBACK    = 60
EPOCHS      = 100
BATCH_SIZE  = 64
MODEL_SAVE  = "models/lstm_general.keras"
START_DATE  = "2013-01-01"

TOP_50 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM",  "V",    "JNJ",
    "WMT",  "PG",   "MA",   "UNH",  "HD",
    "BAC",  "XOM",  "ABBV", "MRK",  "CVX",
    "LLY",  "AVGO", "COST", "PEP",  "KO",
    "ADBE", "CSCO", "TMO",  "ACN",  "MCD",
    "CRM",  "NEE",  "DHR",  "TXN",  "NFLX",
    "QCOM", "AMD",  "LOW",  "BMY",  "ORCL",
    "PM",   "RTX",  "HON",  "SBUX", "IBM",
    "GE",   "CAT",  "BA",   "MMM",  "GS",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# ── Download data ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  General LSTM Model Training")
print("  Top 50 S&P 500 Companies | yfinance | 2013 to today")
print("=" * 55)

try:
    import yfinance as yf
except ImportError:
    print("\nERROR: yfinance not installed.")
    print("Run:  pip install yfinance  then try again.")
    exit(1)

os.makedirs("models", exist_ok=True)

end_date   = pd.Timestamp.today().strftime("%Y-%m-%d")
all_X      = []
all_y      = []
successful = []

for idx, ticker in enumerate(TOP_50, 1):
    print(f"[{idx:02d}/{len(TOP_50)}] {ticker:<6} ", end="", flush=True)
    try:
        raw = yf.download(ticker, start=START_DATE, end=end_date,
                          interval="1d", progress=False, auto_adjust=True)
        if raw.empty or len(raw) < LOOKBACK + 20:
            print("skipped (not enough data)")
            continue

        close  = raw["Close"].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(close)

        X, y = create_sequences(scaled, LOOKBACK)
        all_X.append(X)
        all_y.append(y)
        successful.append(ticker)
        print(f"OK  ({len(X):,} sequences, {len(raw):,} days)")
    except Exception as e:
        print(f"error — {e}")

print(f"\nSuccessfully processed: {len(successful)} / {len(TOP_50)} companies")
print(f"Companies: {', '.join(successful)}\n")

if not all_X:
    print("No data collected. Check your internet connection and try again.")
    exit(1)

# ── Combine + shuffle ─────────────────────────────────────────────────────────
X_all = np.concatenate(all_X, axis=0).reshape(-1, LOOKBACK, 1)
y_all = np.concatenate(all_y, axis=0)

idx   = np.random.permutation(len(X_all))
X_all = X_all[idx]
y_all = y_all[idx]

split   = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

print(f"Total sequences : {len(X_all):,}")
print(f"Training        : {len(X_train):,}")
print(f"Testing         : {len(X_test):,}")

# ── Build model ───────────────────────────────────────────────────────────────
print("\nBuilding LSTM model...")

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1),
])
model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

early_stop = EarlyStopping(
    monitor="val_loss", patience=10,
    restore_best_weights=True, verbose=1
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nTraining started — this will take 30-60 minutes...")
print("(Early stopping will end it sooner if the model converges)\n")

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
y_pred = model.predict(X_test, verbose=0)
mse    = mean_squared_error(y_test, y_pred)
rmse   = math.sqrt(mse)

print("=" * 55)
print(f"  Test MSE  : {mse:.6f}")
print(f"  Test RMSE : {rmse:.6f}")
print(f"  Epochs    : {len(history.history['loss'])}")
print("=" * 55)

# ── Save ──────────────────────────────────────────────────────────────────────
model.save(MODEL_SAVE)
size_kb = os.path.getsize(MODEL_SAVE) / 1024
print(f"\nModel saved : {MODEL_SAVE}  ({size_kb:.0f} KB)")
print("\nDone! Now run:  streamlit run app.py")
print("The dashboard will automatically use the new general model.")
