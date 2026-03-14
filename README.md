# Crypto Volatility Forecasting with Multimodal Deep Learning

This repository contains a multimodal deep learning project for forecasting **Bitcoin realised volatility** by combining:

- **structured market time-series data** (BTC-USD price and returns)
- **unstructured Twitter sentiment data**
- **lagged volatility features**

The project was developed as an end-to-end, leakage-safe forecasting pipeline and includes data preprocessing, sentiment extraction, multimodal feature fusion, LSTM-based modelling, baseline comparison, ablation studies, and evaluation outputs.

## Project Overview

Cryptocurrency markets are highly volatile, non-stationary, and influenced by both internal price dynamics and external public sentiment. This project investigates whether combining market data with Twitter-derived sentiment can improve **Bitcoin realised volatility forecasting**.

The final implementation uses:

- **BTC-USD daily market data** from `yfinance`
- **Bitcoin-related tweet data**
- **Transformer-based sentiment extraction**
- **PyTorch LSTM regression**
- **Chronological train/validation/test split**
- **Aligned comparison against strong baselines**

## Main Objective

The goal of this project is to forecast **daily Bitcoin realised volatility** using a multimodal pipeline that integrates:

1. market-based numerical features  
2. sentiment-derived text features  
3. temporal volatility memory through lagged features  

## Key Features

- Leakage-safe preprocessing
- Daily log-return and realised-volatility construction
- Transformer-based tweet sentiment extraction
- Daily sentiment aggregation
- Multimodal feature fusion
- LSTM regression model in PyTorch
- Baseline comparison:
  - Persistence
  - Rolling mean
  - EWMA
- Ablation experiments:
  - Market only
  - No lag features
  - Price and return only
  - No log-target version
- Diagnostic plots and saved outputs
- Reproducible configuration-based workflow

## Repository Structure

```text
crypto-volatility-multimodal-dl/
├── report/
│   └── IEEE_Conference_Template_4.pdf
├── src/
│   └── crypto_volatility_updated_final.py
├── outputs/
│   ├── tables/
│   ├── figures/
│   └── metrics/
├── data/
│   └── Bitcoin_tweets_dataset_2.csv   # if included locally
├── README.md
└── .gitignore
