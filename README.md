#  Advanced Stock Prediction & Analysis System

> **Minor Project — Paid Internship**
> Author: **Spoorthi S Patil** | Version: **2.1** | Platform: **Google Colab / Python**

---

##  Overview

A dual-model machine learning pipeline that predicts short-term stock prices using 3 years of historical market data, three technical indicators, and an interactive command-line interface. The system trains both an **XGBoost** and a **Bidirectional LSTM** model, automatically selects the better performer, and outputs next-day predictions alongside rolling 7-day forecasts — complete with confidence scores, trend signals, and human-readable reasoning.

---

##  Features

| Feature | Description |
|---|---|
| **Dual-Model Training** | XGBoost (gradient boosting) and BiLSTM compete; best RMSE wins |
| **3 Technical Indicators** | MA10, RSI (14-day), and MACD for rich feature engineering |
| **3 Evaluation Metrics** | MAE, RMSE, and MAPE reported on every run |
| **7-Day Rolling Forecast** | Autoregressive forecast with ±2% confidence bands |
| **Confidence Score** | Error-based confidence, reduced automatically for long-horizon forecasts |
| **Trend Signal** | Bullish / Bearish / Neutral with plain-English reasoning |
| **Volatility Warning** | Flags high-volatility regimes (>3% daily std dev) |
| **Model Persistence** | Saves trained models to disk for production reuse |
| **24 Supported Tickers** | TSLA, AAPL, MSFT, META, AMZN, GOOG, AMD, and more |

---

##  Tech Stack

- **Language:** Python 3.10+
- **Data Source:** Yahoo Finance via `yfinance`
- **ML Framework:** TensorFlow / Keras (BiLSTM), XGBoost
- **Data Processing:** NumPy, Pandas, Scikit-learn
- **Visualisation:** Matplotlib (`fivethirtyeight` style)
- **Platform:** Google Colab (also runs locally)

---

##  Installation

```bash
pip install yfinance xgboost tensorflow scikit-learn matplotlib pandas numpy joblib
```

> **In Google Colab:** all major libraries are pre-installed. Only `yfinance` and `xgboost` may need installation:
> ```bash
> !pip install yfinance xgboost --quiet
> ```

---

##  Usage

### Run in Google Colab
1. Upload `stock_predictions_minor.py` to your Colab session
2. Run all cells
3. Enter a ticker when prompted (e.g. `TSLA`, `AAPL`, `MSFT`)
4. Wait for both models to train (~60–90 seconds)
5. Use the interactive menu:

```
═══════════════════════════════════════
        STOCK COMMAND CENTER
═══════════════════════════════════════
  [P] Predict Next Day | [F] 7-Day Forecast | [Q] Quit :
```

### Run Locally
```bash
python stock_predictions_minor.py
```

---

##  Project Structure

```
stock_predictions_minor.py       ← Main script
saved_models/
    {TICKER}_xgb.json            ← Saved XGBoost model
    {TICKER}_lstm.keras          ← Saved BiLSTM model
```

---

##  Technical Indicators

### MA10 — 10-Day Moving Average
Simple average of the last 10 closing prices. Smooths daily noise to reveal the short-term trend direction.
- Price > MA10 → Upward momentum
- Price < MA10 → Downward pressure

### RSI — Relative Strength Index (14-day)
Momentum oscillator measuring the speed and magnitude of recent price changes.
- RSI > 70 → **Overbought** (potential reversal down)
- RSI < 30 → **Oversold** (potential reversal up)

### MACD — Moving Average Convergence/Divergence *(added v2.1)*
Difference between the 12-day and 26-day Exponential Moving Averages.
- MACD > 0 → **Bullish** momentum (fast EMA above slow EMA)
- MACD < 0 → **Bearish** momentum (fast EMA below slow EMA)

---

## Model Details

### XGBoost Regressor
```
n_estimators     = 200
learning_rate    = 0.05
max_depth        = 5
subsample        = 0.8
colsample_bytree = 0.8
```
Input is the 20 × 8 feature window **flattened** into a 1D vector (160 features).

### Bidirectional LSTM
```
Architecture:
  Input → BiLSTM(64, return_sequences=True)
        → Dropout(0.2)
        → LSTM(32)
        → Dropout(0.1)
        → Dense(1)

Optimizer   : Adam
Loss        : MSE
Epochs      : Up to 20 (EarlyStopping, patience=3)
Val split   : 10%
Batch size  : 32
```
Input is the raw 3D sequence: `(1, TIME_STEPS=20, FEATURES=8)`.

---

##  Evaluation Metrics

| Metric | Formula | Best For |
|---|---|---|
| **MAE** | mean(\|actual − predicted\|) | Intuitive USD error |
| **RMSE** | √mean((actual − predicted)²) | Model selection — penalises outliers |
| **MAPE** | mean(\|actual − predicted\| / actual) × 100 | Cross-stock comparison (scale-free %) |

> **Model selection is based on RMSE** — its sensitivity to large errors makes it the most responsible choice for financial applications.

---

##  Sample Output

```
══════════════════════════════════════════════════
   ANALYSIS REPORT — TSLA
══════════════════════════════════════════════════
  Current Price  : $248.72
  Predicted Next : $253.15  (+1.78%)
  Confidence     : 78.4%
  Trend Signal   : 📈 Bullish (Short-term)
  Reason         : Based on MACD is positive (Bullish crossover)
                   and Price holding above 10-day MA (Upward momentum)
══════════════════════════════════════════════════

7-DAY FORECAST — TSLA
══════════════════════════════════════════════════
  Day 1: $251.00  ▲  (+0.92% vs today)
  Day 2: $253.15  ▲  (+1.78% vs today)
  Day 3: $254.80  ▲  (+2.45% vs today)
  Day 4: $256.20  ▲  (+3.01% vs today)
  Day 5: $255.50  ▲  (+2.73% vs today)
  Day 6: $257.10  ▲  (+3.37% vs today)
  Day 7: $258.40  ▲  (+3.89% vs today)
══════════════════════════════════════════════════
```

---

##  Disclaimer

This project is built for **educational purposes** as part of an academic internship. Stock price predictions are inherently uncertain and should **not** be used as financial advice. The models make simplified assumptions and do not account for macroeconomic events, earnings surprises, or market sentiment.

---

##  Future Improvements

- Add Transformer / Attention-based architecture
- Include sentiment analysis from financial news (NLP pipeline)
- Extend to Indian market tickers (NSE/BSE via `nsepy`)
- Deploy as a Flask / Streamlit web application
- Add hyperparameter tuning with Optuna or GridSearchCV

---

##  License

This project was created for internship evaluation purposes. All rights reserved © Spoorthi S Patil.
