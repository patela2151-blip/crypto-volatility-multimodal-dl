# 📊 Bitcoin Tweets Dataset (Feb–Mar 2023)

## 🧾 Overview
This dataset contains Bitcoin-related tweets collected over a short time window from **February 25, 2023 to March 5, 2023**. It is designed for use in:

- Sentiment Analysis (NLP)
- Cryptocurrency Market Research
- Feature Engineering for Time-Series Models
- Social Media Analytics

This dataset is particularly useful for combining **textual sentiment with financial data** (e.g., Bitcoin price/volatility prediction models).

---

## 📁 Dataset File

- **File Name:** `Bitcoin_tweets_dataset_2.csv`
- **Format:** CSV (Comma-Separated Values)
- **Records:** ~174,000 tweets
- **Attributes:** 13 columns

---

## 📌 Features Description

| Column Name | Description |
|------------|------------|
| `user_name` | Twitter username |
| `user_location` | User's location (self-reported) |
| `user_description` | Bio of the user |
| `user_created` | Account creation date |
| `user_followers` | Number of followers |
| `user_friends` | Number of followings |
| `user_favourites` | Number of likes |
| `user_verified` | Verified account (True/False) |
| `date` | Tweet timestamp |
| `text` | Tweet content |
| `hashtags` | Hashtags used |
| `source` | Posting platform (e.g., iPhone, Web) |
| `is_retweet` | Retweet flag |

---

## ⏱️ Time Range

- **Start Date:** 2023-02-25  
- **End Date:** 2023-03-05  

---

## 🧠 Use Cases

This dataset can be used for:

### 🔹 1. Sentiment Analysis
- Classify tweets into positive, negative, or neutral sentiment
- Use models like VADER, TextBlob, or BERT

### 🔹 2. Crypto Market Prediction
- Aggregate daily sentiment scores
- Combine with Bitcoin price data for forecasting

### 🔹 3. NLP Tasks
- Text cleaning and preprocessing
- Tokenization, embeddings, topic modeling

### 🔹 4. Social Media Insights
- Identify trending hashtags
- Analyze user influence (followers vs engagement)

---

## ⚙️ How to Load Dataset

```python
import pandas as pd

df = pd.read_csv("Bitcoin_tweets_dataset_2.csv")
print(df.head())
