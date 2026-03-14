# ============================================================
# Cryptocurrency Volatility Prediction (FINAL SUBMISSION VERSION)
# Leakage-safe + Experiments + Fair Baseline Alignment + Bootstrap CIs
#
# Outputs created in current working directory:
#   outputs/, figures/, models/
# ============================================================

import os
import time
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

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

@dataclass
class Config:
    seed: int = 42
    device: str = "cpu"

    # Data
    start_date: str = "2021-01-01"
    end_date: str = "2023-12-31"
    ticker: str = "BTC-USD"

    # Tweets (optional)
    tweets_csv: str = "Bitcoin_tweets_dataset_2.csv"
    daily_tweet_cap: int = 100
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment"
    sentiment_batch_size: int = 32
    sentiment_max_length: int = 128

    # Volatility + target
    vol_window: int = 7
    eps: float = 1e-8

    # Sequence model
    seq_len: int = 30

    # Chronological split
    train_frac: float = 0.70
    val_frac: float = 0.15

    # Model/training
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    batch_size: int = 32
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 0.0
    shuffle_train: bool = False
    patience: int = 8
    min_delta: float = 1e-4

    # Baselines
    rolling_k: int = 7
    ewma_alpha: float = 0.94

    # Bootstrap CIs
    n_boot: int = 1000   # set 0 to disable
    ci_alpha: float = 0.05

    # Output
    out_dir: str = "outputs"
    fig_dir: str = "figures"
    model_dir: str = "models"
    run_name: str = "btc_vol"

    # Plot handling
    show_plots: bool = False
    open_saved_figs: bool = False


# ============================================================
# UTILITIES
# ============================================================

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.fig_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)

class FigureSaver:
    def __init__(self, fig_dir: str):
        self.fig_dir = fig_dir
        self.fig_id = 1

    def save(self, title: str, filename: str) -> None:
        plt.title(f"Figure {self.fig_id}: {title}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, filename), dpi=300)
        plt.close()
        self.fig_id += 1

def safe_find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of {candidates} found in columns: {list(df.columns)}")


# ============================================================
# TWEETS + SENTIMENT (OPTIONAL)
# ============================================================

def load_and_prepare_tweets(cfg: Config) -> pd.DataFrame:
    tweets = pd.read_csv(
        cfg.tweets_csv,
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip",
    )
    date_col = safe_find_col(tweets, ["date", "created_at", "timestamp"])
    text_col = safe_find_col(tweets, ["text", "tweet", "content"])

    tweets = tweets[[date_col, text_col]].rename(columns={date_col: "date", text_col: "text"})
    tweets["date"] = pd.to_datetime(tweets["date"], errors="coerce").dt.date
    tweets = tweets.dropna(subset=["date", "text"])
    tweets["text"] = tweets["text"].astype(str)

    def _sample_day(g):
        k = min(cfg.daily_tweet_cap, len(g))
        return g.sample(n=k, random_state=cfg.seed)

    tweets = tweets.groupby("date", group_keys=False).apply(_sample_day).reset_index(drop=True)
    return tweets

def infer_sentiment(cfg: Config, tweets: pd.DataFrame) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(cfg.sentiment_model)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.sentiment_model).to(cfg.device)
    model.eval()

    scores: List[float] = []
    texts = tweets["text"].tolist()

    for i in tqdm(range(0, len(texts), cfg.sentiment_batch_size), desc="Batch sentiment"):
        batch = texts[i:i + cfg.sentiment_batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=cfg.sentiment_max_length,
            return_tensors="pt",
        ).to(cfg.device)

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)
            # labels: [neg, neu, pos]
            batch_scores = (probs[:, 2] - probs[:, 0]).detach().cpu().numpy()
            scores.extend(batch_scores.tolist())

    tmp = tweets.copy()
    tmp["sentiment"] = scores

    daily = tmp.groupby("date").agg(
        sentiment_mean=("sentiment", "mean"),
        sentiment_std=("sentiment", "std"),
        tweet_volume=("sentiment", "count"),
    ).reset_index()

    daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)
    return daily


# ============================================================
# MARKET DATA + MERGE
# ============================================================

def load_btc(cfg: Config) -> pd.DataFrame:
    btc = yf.download(cfg.ticker, start=cfg.start_date, end=cfg.end_date, progress=False)

    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)

    btc = btc.reset_index()
    btc["date"] = pd.to_datetime(btc["Date"]).dt.date

    btc["return"] = np.log(btc["Close"] / btc["Close"].shift(1))
    btc["volatility"] = btc["return"].rolling(cfg.vol_window).std()

    btc = btc.dropna(subset=["return", "volatility"]).copy()
    return btc[["date", "Close", "return", "volatility"]]

def build_base_dataframe(cfg: Config, btc: pd.DataFrame, daily_sentiment: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = btc.copy()

    if daily_sentiment is not None:
        df = pd.merge(df, daily_sentiment, on="date", how="left")
        df[["sentiment_mean", "sentiment_std", "tweet_volume"]] = df[
            ["sentiment_mean", "sentiment_std", "tweet_volume"]
        ].fillna(0.0)
    else:
        df["sentiment_mean"] = 0.0
        df["sentiment_std"] = 0.0
        df["tweet_volume"] = 0.0

    df["vol_lag_1"] = df["volatility"].shift(1)
    df["vol_lag_7"] = df["volatility"].shift(7)

    df = df.dropna().reset_index(drop=True)
    return df

def add_target(cfg: Config, df: pd.DataFrame, target_log_transform: bool) -> pd.DataFrame:
    out = df.copy()
    if target_log_transform:
        out["target"] = np.log(out["volatility"] + cfg.eps)
    else:
        out["target"] = out["volatility"]
    return out


# ============================================================
# SPLIT + SCALE (LEAKAGE SAFE)
# ============================================================

def chronological_split(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = int(cfg.train_frac * n)
    n_val = int(cfg.val_frac * n)

    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    return train, val, test

def fit_transform_scalers(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(train[feature_cols].values)
    y_train = scaler_y.fit_transform(train[[target_col]].values)

    X_val = scaler_X.transform(val[feature_cols].values)
    y_val = scaler_y.transform(val[[target_col]].values)

    X_test = scaler_X.transform(test[feature_cols].values)
    y_test = scaler_y.transform(test[[target_col]].values)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y


# ============================================================
# SEQUENCES
# ============================================================

def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.asarray(Xs), np.asarray(ys)

def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, cfg: Config):
    Xtr, ytr = make_sequences(X_train, y_train, cfg.seq_len)
    Xva, yva = make_sequences(X_val, y_val, cfg.seq_len)
    Xte, yte = make_sequences(X_test, y_test, cfg.seq_len)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    yva_t = torch.tensor(yva, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_train,
    )
    return train_loader, (Xva_t, yva_t), (Xte_t, yte_t)


# ============================================================
# MODEL
# ============================================================

class VolatilityLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ============================================================
# TRAINING
# ============================================================

def train_model(cfg: Config, train_loader, val_data, input_dim: int, save_path: str):
    model = VolatilityLSTM(input_dim, cfg.hidden_size, cfg.num_layers, cfg.dropout).to(cfg.device)
    criterion = nn.SmoothL1Loss(beta=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val = float("inf")
    patience_left = cfg.patience

    train_losses, val_losses = [], []
    Xva, yva = val_data

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        batch_losses = []

        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")

        model.eval()
        with torch.no_grad():
            val_preds = model(Xva.to(cfg.device))
            val_loss = float(criterion(val_preds, yva.to(cfg.device)).item())

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        improved = (best_val - val_loss) > cfg.min_delta
        if improved:
            best_val = val_loss
            patience_left = cfg.patience
            torch.save(model.state_dict(), save_path)
        else:
            patience_left -= 1

        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f} | lr {lr_now:.2e} | patience {patience_left}")

        if patience_left <= 0:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(save_path, map_location=cfg.device))
    return model, train_losses, val_losses, float(best_val)


# ============================================================
# METRICS + BASELINES (ALIGNED) + BOOTSTRAP CIs
# ============================================================

def inverse_target(cfg: Config, scaler_y: StandardScaler, y_scaled: np.ndarray, target_log_transform: bool) -> np.ndarray:
    y = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)
    if target_log_transform:
        return np.exp(y) - cfg.eps
    return y

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"mse": float(mse), "rmse": rmse, "mae": mae}

def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int, alpha: float, seed: int) -> Dict[str, Dict[str, float]]:
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 5:
        return {}

    rmses, maes = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        m = compute_metrics(yt, yp)
        rmses.append(m["rmse"])
        maes.append(m["mae"])

    lo = 100 * (alpha / 2)
    hi = 100 * (1 - alpha / 2)
    return {
        "rmse": {"low": float(np.percentile(rmses, lo)), "high": float(np.percentile(rmses, hi))},
        "mae": {"low": float(np.percentile(maes, lo)), "high": float(np.percentile(maes, hi))},
    }

def baseline_preds_persistence(y_true: np.ndarray) -> np.ndarray:
    pred = np.full_like(y_true, np.nan, dtype=float)
    pred[1:] = y_true[:-1]
    return pred

def baseline_preds_rolling_mean(y_true: np.ndarray, k: int) -> np.ndarray:
    pred = np.full_like(y_true, np.nan, dtype=float)
    for t in range(k, len(y_true)):
        pred[t] = float(np.mean(y_true[t-k:t]))
    return pred

def baseline_preds_ewma(y_true: np.ndarray, alpha: float) -> np.ndarray:
    pred = np.full_like(y_true, np.nan, dtype=float)
    ew = float(y_true[0])
    for t in range(1, len(y_true)):
        ew = alpha * ew + (1 - alpha) * float(y_true[t-1])
        pred[t] = ew
    return pred

def aligned_evaluation(cfg: Config, y_true: np.ndarray, y_pred_model: np.ndarray) -> Dict[str, Any]:
    pred_p = baseline_preds_persistence(y_true)
    pred_r = baseline_preds_rolling_mean(y_true, cfg.rolling_k)
    pred_e = baseline_preds_ewma(y_true, cfg.ewma_alpha)

    # Fair comparison mask: all baselines defined
    mask = ~np.isnan(pred_p) & ~np.isnan(pred_r) & ~np.isnan(pred_e)

    y_t = y_true[mask]
    y_m = y_pred_model[mask]
    y_p = pred_p[mask]
    y_r = pred_r[mask]
    y_e = pred_e[mask]

    out = {
        "n_eval": int(mask.sum()),
        "metrics": {
            "model": compute_metrics(y_t, y_m),
            "persistence": compute_metrics(y_t, y_p),
            f"rolling_mean_k={cfg.rolling_k}": compute_metrics(y_t, y_r),
            f"ewma_alpha={cfg.ewma_alpha}": compute_metrics(y_t, y_e),
        },
        "ci": {},
        "aligned_series": {"y_true": y_t, "y_pred_model": y_m},
    }

    if cfg.n_boot > 0:
        out["ci"] = {
            "model": bootstrap_ci(y_t, y_m, cfg.n_boot, cfg.ci_alpha, cfg.seed),
            "persistence": bootstrap_ci(y_t, y_p, cfg.n_boot, cfg.ci_alpha, cfg.seed + 1),
        }

    return out


# ============================================================
# PLOTS (MAIN EXPERIMENT ONLY)
# ============================================================

def make_plots(figs: FigureSaver, train_losses, val_losses, y_true, y_pred):
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.legend()
    figs.save("Training and Validation Loss Curve", "fig1_loss.png")

    plt.figure()
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, linestyle="--", label="Predicted")
    plt.legend()
    figs.save("Actual vs Predicted Volatility", "fig2_actual_vs_pred.png")

    residuals = y_true - y_pred
    plt.figure()
    plt.plot(residuals)
    plt.axhline(0, linestyle="--")
    figs.save("Residual Errors Over Time", "fig3_residuals.png")

    plt.figure()
    plt.hist(residuals, bins=30, edgecolor="black")
    figs.save("Residual Distribution", "fig4_residual_dist.png")

    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn, mx = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    figs.save("Predicted vs Actual Scatter", "fig5_scatter.png")

    rolling_rmse = pd.Series(np.sqrt((y_true - y_pred) ** 2)).rolling(20).mean()
    plt.figure()
    plt.plot(rolling_rmse)
    figs.save("Rolling RMSE (Window=20)", "fig6_rmse.png")


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(
    cfg: Config,
    df_with_target: pd.DataFrame,
    exp_name: str,
    feature_cols: List[str],
    target_log_transform: bool,
    make_main_plots: bool = False,
) -> Dict[str, Any]:

    train_df, val_df, test_df = chronological_split(df_with_target, cfg)

    X_train, y_train, X_val, y_val, X_test, y_test, _, scaler_y = fit_transform_scalers(
        train_df, val_df, test_df, feature_cols, "target"
    )

    train_loader, val_data, test_data = make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, cfg)
    input_dim = train_loader.dataset.tensors[0].shape[-1]

    save_path = os.path.join(cfg.model_dir, f"{cfg.run_name}_{exp_name}_best.pt")

    print(f"\n===== EXPERIMENT: {exp_name} =====")
    print(f"Features: {feature_cols}")
    print(f"Target: {'log(vol)' if target_log_transform else 'vol'} | seq_len={cfg.seq_len}")

    model, train_losses, val_losses, best_val = train_model(cfg, train_loader, val_data, input_dim, save_path)

    # Predict on test
    Xte, yte = test_data
    model.eval()
    with torch.no_grad():
        preds_scaled = model(Xte.to(cfg.device)).cpu().numpy().reshape(-1)

    y_true_scaled = yte.cpu().numpy().reshape(-1)
    y_pred = inverse_target(cfg, scaler_y, preds_scaled, target_log_transform)
    y_true = inverse_target(cfg, scaler_y, y_true_scaled, target_log_transform)

    aligned = aligned_evaluation(cfg, y_true, y_pred)

    if make_main_plots:
        figs = FigureSaver(cfg.fig_dir)
        s = aligned["aligned_series"]
        make_plots(figs, train_losses, val_losses, s["y_true"], s["y_pred_model"])

    return {
        "experiment": exp_name,
        "target_log_transform": target_log_transform,
        "seq_len": cfg.seq_len,
        "features": feature_cols,
        "model_path": save_path,
        "best_val_loss": best_val,
        "n_eval_aligned": aligned["n_eval"],
        "metrics_aligned": aligned["metrics"],
        "ci_aligned": aligned["ci"],
    }


# ============================================================
# MAIN
# ============================================================

def main():
    cfg = Config()

    cwd = os.getcwd()
    cfg.out_dir = os.path.join(cwd, cfg.out_dir)
    cfg.fig_dir = os.path.join(cwd, cfg.fig_dir)
    cfg.model_dir = os.path.join(cwd, cfg.model_dir)

    set_seed(cfg.seed)
    ensure_dirs(cfg)

    print(f"Outputs -> {cfg.out_dir}")
    print(f"Figures  -> {cfg.fig_dir}")
    print(f"Models   -> {cfg.model_dir}")
    print(f"Device: {cfg.device}")

    t0 = time.time()

    with open(os.path.join(cfg.out_dir, f"{cfg.run_name}_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    daily_sentiment = None
    tweets_used = 0

    print("\n[1/3] Loading tweets + sentiment (once)...")
    try:
        tweets = load_and_prepare_tweets(cfg)
        tweets_used = len(tweets)
        print(f"Tweets after sampling: {tweets_used}")
        daily_sentiment = infer_sentiment(cfg, tweets)
        print(f"Daily sentiment rows: {len(daily_sentiment)}")
    except FileNotFoundError:
        print("Tweets CSV not found. Continuing with sentiment features set to zero.")
        daily_sentiment = None

    print("\n[2/3] Loading BTC market data...")
    btc = load_btc(cfg)
    print(f"BTC rows: {len(btc)}")

    base_df = build_base_dataframe(cfg, btc, daily_sentiment)
    print(f"Merged rows (after lags): {len(base_df)}")

    feats_full = ["Close", "return", "sentiment_mean", "sentiment_std", "tweet_volume", "vol_lag_1", "vol_lag_7"]
    feats_market = ["Close", "return", "vol_lag_1", "vol_lag_7"]
    feats_no_lag = ["Close", "return", "sentiment_mean", "sentiment_std", "tweet_volume"]
    feats_price_only = ["Close", "return"]

    experiments: List[Tuple[str, List[str], bool]] = [
        ("multimodal_full", feats_full, True),
        ("market_only", feats_market, True),
        ("no_lag_features", feats_no_lag, True),
        ("price_return_only", feats_price_only, True),
        ("no_log_target_full", feats_full, False),
    ]

    print("\n[3/3] Running experiments...")
    results: List[Dict[str, Any]] = []

    for exp_name, feat_cols, log_tgt in experiments:
        df_t = add_target(cfg, base_df, target_log_transform=log_tgt)
        res = run_experiment(cfg, df_t, exp_name, feat_cols, log_tgt, make_main_plots=(exp_name == "multimodal_full"))
        results.append(res)

    rows = []
    for r in results:
        m = r["metrics_aligned"]["model"]
        bp = r["metrics_aligned"]["persistence"]
        br = r["metrics_aligned"][f"rolling_mean_k={cfg.rolling_k}"]
        be = r["metrics_aligned"][f"ewma_alpha={cfg.ewma_alpha}"]

        ci_model = r.get("ci_aligned", {}).get("model", {})
        ci_pers = r.get("ci_aligned", {}).get("persistence", {})

        row = {
            "experiment": r["experiment"],
            "target": "log(vol)" if r["target_log_transform"] else "vol",
            "seq_len": r["seq_len"],
            "n_eval_aligned": r["n_eval_aligned"],
            "rmse_model": m["rmse"],
            "mae_model": m["mae"],
            "rmse_persistence": bp["rmse"],
            f"rmse_rollmean_k{cfg.rolling_k}": br["rmse"],
            f"rmse_ewma_a{cfg.ewma_alpha}": be["rmse"],
            "best_val_loss": r["best_val_loss"],
            "model_path": r["model_path"],
        }

        if ci_model:
            row["rmse_model_ci_low"] = ci_model["rmse"]["low"]
            row["rmse_model_ci_high"] = ci_model["rmse"]["high"]
            row["mae_model_ci_low"] = ci_model["mae"]["low"]
            row["mae_model_ci_high"] = ci_model["mae"]["high"]

        if ci_pers:
            row["rmse_persist_ci_low"] = ci_pers["rmse"]["low"]
            row["rmse_persist_ci_high"] = ci_pers["rmse"]["high"]
            row["mae_persist_ci_low"] = ci_pers["mae"]["low"]
            row["mae_persist_ci_high"] = ci_pers["mae"]["high"]

        rows.append(row)

    table = pd.DataFrame(rows).sort_values("rmse_model")

    runtime_min = (time.time() - t0) / 60.0
    results_json = os.path.join(cfg.out_dir, f"{cfg.run_name}_all_results.json")
    table_csv = os.path.join(cfg.out_dir, f"{cfg.run_name}_comparison_table.csv")

    with open(results_json, "w") as f:
        json.dump(
            {
                "runtime_min": runtime_min,
                "tweets_used": tweets_used,
                "sentiment_days": int(0 if daily_sentiment is None else len(daily_sentiment)),
                "results": results,
            },
            f,
            indent=2,
        )

    table.to_csv(table_csv, index=False)

    print("\n================ COMPARISON TABLE (ALIGNED) ================")
    print(table.to_string(index=False))
    print("============================================================")
    print(f"\nSaved JSON: {results_json}")
    print(f"Saved CSV : {table_csv}")
    print(f"Runtime (min): {runtime_min:.3f}")

    if cfg.show_plots:
        plt.show()

    if cfg.open_saved_figs:
        try:
            fig_files = [
                "fig1_loss.png", "fig2_actual_vs_pred.png", "fig3_residuals.png",
                "fig4_residual_dist.png", "fig5_scatter.png", "fig6_rmse.png"
            ]
            for ff in fig_files:
                p = os.path.join(cfg.fig_dir, ff)
                if os.path.exists(p):
                    if hasattr(os, "startfile"):
                        os.startfile(p)
                    else:
                        import subprocess
                        opener = "open" if sys.platform == "darwin" else "xdg-open"
                        subprocess.Popen([opener, p])
        except Exception as e:
            print(f"Could not open saved figures automatically: {e}")


if __name__ == "__main__":
    main()
