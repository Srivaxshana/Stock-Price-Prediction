# Stock-Price-Prediction
AI Mini Project: Stock Price Prediction using LSTM

This repository contains exploratory analysis, a simple Streamlit dashboard, and training utilities for an LSTM-based stock price prediction mini-project.

Quick start

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Train a model for a specific ticker (example `AAPL`):
```bash
python run_train.py --symbol AAPL --epochs 20
```
3. Run the dashboard:
```bash
streamlit run app.py
```

Notes
- Place trained models and scalers under `models/` (script saves as `models/lstm_<TICKER>.h5` and `models/scaler_<TICKER>.bin`).
- The notebook `stock-price-direction-forecasting-with-lstm.ipynb` contains EDA and a training cell you can run interactively.
- The LaTeX proposal source is `proposal.tex` — compile locally with `pdflatex proposal.tex` to produce the PDF required for submission.

