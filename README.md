# Cryptocurrency Volatility Prediction Using Multimodal Deep Learning

This repository presents a multimodal deep learning framework for predicting Bitcoin volatility by integrating numerical market indicators with social media sentiment extracted from Twitter. The system combines an LSTM-based time-series model with transformer-based sentiment analysis to capture both quantitative and qualitative market drivers.

---

## 📌 Project Overview

- **Task**: Bitcoin volatility prediction
- **Data Sources**:
  - BTC-USD market data (Yahoo Finance)
  - Twitter sentiment (Bitcoin-related tweets)
- **Models**:
  - BERT-based sentiment analysis
  - LSTM for volatility forecasting
- **Frameworks**: PyTorch, Transformers
- **Runtime**: ~4 minutes (CPU)

---

## 🧠 Methodology

1. **Data Preprocessing**
   - Tweet cleaning and daily sampling
   - Batch sentiment inference
   - Daily sentiment aggregation

2. **Feature Engineering**
   - Log returns
   - Rolling volatility
   - Mean and variance of sentiment

3. **Model Architecture**
   - LSTM with rolling window inputs
   - Smooth L1 loss for robustness
   - Adam optimizer

4. **Evaluation**
   - MSE, RMSE, MAE
   - Residual analysis
   - Rolling RMSE

---

## 📊 Results

| Metric | Value |
|------|------|
| RMSE | 0.00517 |
| MAE  | 0.00380 |
| Runtime | 3.84 minutes |

The model produces smooth and stable volatility forecasts while capturing medium-term market dynamics.

---

## 📂 Repository Structure

```text
src/        → Core implementation  
figures/    → Output visualisations  
data/       → Processed sentiment data  
models/     → Trained LSTM model  
report/     → IEEE-formatted final report  
