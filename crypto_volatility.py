# ============================================================
# Cryptocurrency Volatility Prediction (Robust Final Version)
# MSc Data Science Coursework – University of Roehampton
# ============================================================

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")   # keep CPU for coursework reproducibility
WINDOW_SIZE = 30
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
start_time = time.time()

# ============================================================
# HELPER: SAVE FIGURES
# ============================================================
FIG_ID = 1

def save_figure(title, filename):
    global FIG_ID
    plt.title(f"Figure {FIG_ID}: {title}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()
    FIG_ID += 1

# ============================================================
# 1. LOAD TWITTER DATA (ROBUST CSV PARSING)
# ============================================================
tweets = pd.read_csv(
    "Bitcoin_tweets_dataset_2.csv",
    engine="python",
    encoding="utf-8",
    on_bad_lines="skip"
)

print(f"Tweets loaded (raw): {len(tweets)}")

# Detect columns safely
possible_date_cols = ["date", "created_at", "timestamp"]
possible_text_cols = ["text", "tweet"]

date_col = next(col for col in possible_date_cols if col in tweets.columns)
text_col = next(col for col in possible_text_cols if col in tweets.columns)

tweets = tweets[[date_col, text_col]].rename(
    columns={date_col: "date", text_col: "text"}
)

tweets["date"] = pd.to_datetime(tweets["date"], errors="coerce").dt.date
tweets = tweets.dropna()

# Daily sampling (max 100 tweets/day)
tweets = (
    tweets.groupby("date", group_keys=False)
    .sample(n=100, random_state=SEED)
)

print(f"Tweets after optimisation: {len(tweets)}")

# ============================================================
# 2. SENTIMENT ANALYSIS (BERT, BATCHED)
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment"
)
sent_model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment"
).to(DEVICE)

sent_model.eval()

def batch_sentiment(texts):
    scores = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Batch sentiment"):
        batch = texts[i:i+BATCH_SIZE]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            logits = sent_model(**enc).logits
            probs = torch.softmax(logits, dim=1)
            scores.extend((probs[:, 2] - probs[:, 0]).cpu().numpy())
    return scores

tweets["sentiment"] = batch_sentiment(tweets["text"].astype(str).tolist())

daily_sentiment = tweets.groupby("date").agg(
    sentiment_mean=("sentiment", "mean"),
    sentiment_std=("sentiment", "std"),
    tweet_volume=("sentiment", "count")
).fillna(0).reset_index()

print(f"Daily sentiment rows: {len(daily_sentiment)}")

# ============================================================
# 3. LOAD BITCOIN PRICE DATA (FIX MULTIINDEX)
# ============================================================
btc = yf.download(
    "BTC-USD",
    start="2021-01-01",
    end="2023-12-31",
    progress=False
)

# Flatten MultiIndex columns
if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.get_level_values(0)

btc = btc.reset_index()
btc["date"] = pd.to_datetime(btc["Date"]).dt.date

btc["return"] = np.log(btc["Close"] / btc["Close"].shift(1))
btc["volatility"] = btc["return"].rolling(7).std()

btc = btc.dropna()
btc = btc[["date", "Close", "return", "volatility"]]

print(f"BTC rows: {len(btc)}")

# ============================================================
# 4. MERGE DATA (SAFE EVEN WITH SPARSE SENTIMENT)
# ============================================================
df = pd.merge(btc, daily_sentiment, on="date", how="left")
df[["sentiment_mean", "sentiment_std", "tweet_volume"]] = (
    df[["sentiment_mean", "sentiment_std", "tweet_volume"]].fillna(0)
)

# Lag features (important)
df["vol_lag_1"] = df["volatility"].shift(1)
df["vol_lag_7"] = df["volatility"].shift(7)

df.dropna(inplace=True)

print(f"Merged dataset rows: {len(df)}")

# ============================================================
# 5. TARGET TRANSFORMATION (LOG VOLATILITY)
# ============================================================
df["log_volatility"] = np.log(df["volatility"] + 1e-8)

features = [
    "Close", "return",
    "sentiment_mean", "sentiment_std", "tweet_volume",
    "vol_lag_1", "vol_lag_7"
]

X = df[features].values
y = df[["log_volatility"]].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# ============================================================
# 6. BUILD SEQUENCES
# ============================================================
def make_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = make_sequences(X, y, WINDOW_SIZE)

split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ============================================================
# 7. LSTM MODEL (STRONGER)
# ============================================================
class VolatilityLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 64,
            num_layers=2,
            dropout=0.3,
            batch_first=True
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = VolatilityLSTM(X_train.shape[2]).to(DEVICE)
criterion = nn.SmoothL1Loss(beta=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ============================================================
# 8. TRAINING
# ============================================================
train_losses, val_losses = [], []

print("\n[TRAINING]")
print("Epoch   Train Loss     Val Loss")

for epoch in range(EPOCHS):
    model.train()
    batch_losses = []

    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    train_loss = np.mean(batch_losses)

    model.eval()
    with torch.no_grad():
        val_preds = model(X_test)
        val_loss = criterion(val_preds, y_test).item()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(f"{epoch:<7} {train_loss:<13.6f} {val_loss:.6f}")

# ============================================================
# 9. EVALUATION
# ============================================================
model.eval()
with torch.no_grad():
    preds_scaled = model(X_test).numpy()

y_pred = scaler_y.inverse_transform(preds_scaled)
y_true = scaler_y.inverse_transform(y_test.numpy())

y_pred_vol = np.exp(y_pred) - 1e-8
y_true_vol = np.exp(y_true) - 1e-8

mse = mean_squared_error(y_true_vol, y_pred_vol)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true_vol, y_pred_vol)

print("\nFINAL METRICS")
print(f"MSE : {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE : {mae:.6f}")

# ============================================================
# 10. FIGURES
# ============================================================
plt.figure()
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.legend()
save_figure("Training and Validation Loss Curve", "fig1_loss.png")

plt.figure()
plt.plot(y_true_vol, label="Actual")
plt.plot(y_pred_vol, linestyle="--", label="Predicted")
plt.legend()
save_figure("Actual vs Predicted Volatility", "fig2_actual_vs_pred.png")

residuals = y_true_vol - y_pred_vol

plt.figure()
plt.plot(residuals)
plt.axhline(0, linestyle="--", color="red")
save_figure("Residual Errors Over Time", "fig3_residuals.png")

plt.figure()
plt.hist(residuals, bins=30, edgecolor="black")
save_figure("Residual Distribution", "fig4_residual_dist.png")

plt.figure()
plt.scatter(y_true_vol, y_pred_vol, alpha=0.6)
plt.plot(
    [y_true_vol.min(), y_true_vol.max()],
    [y_true_vol.min(), y_true_vol.max()],
    "r--"
)
save_figure("Predicted vs Actual Scatter", "fig5_scatter.png")

rolling_rmse = pd.Series(
    np.sqrt((y_true_vol - y_pred_vol) ** 2).flatten()
).rolling(20).mean()

plt.figure()
plt.plot(rolling_rmse)
save_figure("Rolling RMSE (Window=20)", "fig6_rmse.png")

# ============================================================
# 11. SUMMARY
# ============================================================
runtime = (time.time() - start_time) / 60

print("\n================ SUMMARY ================")
print(f"Runtime           : {runtime:.2f} minutes")
print(f"Raw tweets        : {len(tweets)}")
print(f"Sentiment days    : {len(daily_sentiment)}")
print(f"Final rows        : {len(df)}")
print("========================================")
