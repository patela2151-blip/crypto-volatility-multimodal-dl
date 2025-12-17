# Cryptocurrency Volatility Prediction Using Multimodal Deep Learning

## 📌 Overview
This project implements a **multimodal deep learning framework** to predict **Bitcoin price volatility** by combining:
- **Financial time-series data** (Bitcoin prices and returns)
- **Social media sentiment** extracted from Bitcoin-related Twitter data

An **LSTM (Long Short-Term Memory)** network is used to model temporal dependencies, while **BERT-based sentiment analysis** captures market sentiment from tweets. The project demonstrates a complete, optimised, and reproducible deep learning pipeline suitable for academic coursework and professional portfolios.

---

## 🎯 Objectives
- Predict short-term cryptocurrency volatility
- Combine numerical market data with textual sentiment features
- Apply deep learning techniques to real-world financial data
- Optimise computational performance for large-scale NLP tasks

---

## 🧠 Technologies Used
- **Python**
- **PyTorch** – deep learning framework
- **LSTM** – time-series modelling
- **HuggingFace Transformers** – BERT-based sentiment analysis
- **Pandas / NumPy** – data processing
- **Matplotlib** – visualisation
- **yFinance** – Bitcoin market data

---

## 📊 Data Sources
- **Bitcoin Tweets Dataset** – public Twitter dataset
- **Bitcoin price data** (`BTC-USD`) from Yahoo Finance

### Sentiment Processing Optimisation
| Stage | Before | After |
|------|--------|-------|
| Tweet count | ~170,000 | ~10,000–20,000 |
| Sentiment inference | 1-by-1 | Batch |
| Runtime | 3–4 hours | ~10–15 minutes |

To ensure computational feasibility, a fixed number of tweets per day were sampled and sentiment inference was performed in batches.

---

## ⚙️ Methodology
1. Load and clean Bitcoin-related tweets
2. Sample tweets per day to construct representative sentiment proxies
3. Extract sentiment scores using a pre-trained BERT model
4. Aggregate daily sentiment statistics (mean, standard deviation, volume)
5. Download Bitcoin price data and compute returns and realised volatility
6. Merge sentiment and numerical features
7. Construct rolling sequences for LSTM input
8. Train an LSTM-based regression model
9. Evaluate using regression metrics and diagnostic plots

---

## 📈 Model Evaluation
The model is evaluated using:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**

### Visual Analysis
The project includes multiple diagnostic plots:
- Training vs validation loss curve
- Actual vs predicted volatility
- Residual error time series
- Residual distribution histogram
- Predicted vs actual scatter plot
- Rolling RMSE for temporal stability

These plots support interpretability and performance analysis.

---

## 📂 Project Structure
