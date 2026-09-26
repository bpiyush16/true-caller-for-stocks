from __future__ import annotations

import random
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

SEED = 42

START_DATE = "2000-01-01"
HISTORY_END_EXCLUSIVE = "2025-01-01"
TRAIN_END_DATE = "2019-01-01"
VAL_END_DATE = "2022-01-01"

WINDOW = 30
SEQ_LEN = 20
WARMUP_CALENDAR_DAYS = 260

AUTOENCODER_BATCH_SIZE = 256
AUTOENCODER_EPOCHS = 20
AUTOENCODER_LR = 1e-3
AUTOENCODER_HIDDEN_SIZE = 64
AUTOENCODER_NUM_LAYERS = 2
AUTOENCODER_PATIENCE = 5

HMM_COMPONENTS = 3
HMM_PCA_COMPONENTS = 5
HMM_STAY_PROB = 0.90
HMM_RESTARTS = 5

CLASSIFIER_BATCH_SIZE = 64
CLASSIFIER_MAX_EPOCHS = 120
CLASSIFIER_PATIENCE = 12
CLASSIFIER_LR = 5e-4
CLASSIFIER_WEIGHT_DECAY = 1e-4
CLASSIFIER_HIDDEN_SIZE = 128
CLASSIFIER_NUM_LAYERS = 2
CLASSIFIER_DROPOUT = 0.35

HIGH_CONF_THRESHOLD = 0.80

COLUMN_ORDER = ["SP500", "VIX", "BOND", "GOLD", "OIL"]
TICKERS = {
    "SP500": "^GSPC",
    "VIX": "^VIX",
    "BOND": "^TNX",
    "GOLD": "GC=F",
    "OIL": "CL=F",
}

REGIME_NAMES = {
    0: "Calm Market",
    1: "Risk-Off",
    2: "Crisis",
}

VARIANT_CONFIGS = {
    "multi_asset": {
        "display_name": "Multi-Asset",
        "model_path": "supervised_multi_asset_regime_predictor.pth",
        "scaler_path": "supervised_multi_asset_predictor_scaler.pkl",
        "report_path": "supervised_multi_asset_training_report.txt",
        "csv_path": "backtest_predictions_multi_asset.csv",
    },
    "single_asset": {
        "display_name": "Single-Asset (SP500 only)",
        "model_path": "supervised_single_asset_regime_predictor.pth",
        "scaler_path": "supervised_single_asset_predictor_scaler.pkl",
        "report_path": "supervised_single_asset_training_report.txt",
        "csv_path": "backtest_predictions_single_asset.csv",
    },
}

CLASS_COLORS = {
    0: "#2a9d8f",
    1: "#e9c46a",
    2: "#e76f51",
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int = len(COLUMN_ORDER),
        hidden_size: int = AUTOENCODER_HIDDEN_SIZE,
        num_layers: int = AUTOENCODER_NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=input_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (hidden, _) = self.encoder(x)
        latent = hidden[-1]
        latent_seq = latent.unsqueeze(1).repeat(1, x.size(1), 1)
        reconstructed, _ = self.decoder(latent_seq)
        return reconstructed, latent


class RegimeClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = CLASSIFIER_HIDDEN_SIZE,
        num_layers: int = CLASSIFIER_NUM_LAYERS,
        dropout: float = CLASSIFIER_DROPOUT,
        num_classes: int = len(REGIME_NAMES),
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        out = self.dropout(hidden[-1])
        out = self.relu(self.fc1(out))
        return self.fc2(out)


def download_market_prices(start: str, end: str, threads: bool = False) -> pd.DataFrame:
    if hasattr(yf, "config") and hasattr(yf.config, "network"):
        yf.config.network.retries = 1

    data = yf.download(
        tickers=list(TICKERS.values()),
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=threads,
        timeout=20,
    )

    if data.empty:
        raise RuntimeError("Download failed: yfinance returned an empty DataFrame.")
    if not isinstance(data.columns, pd.MultiIndex):
        raise RuntimeError("Expected multi-ticker MultiIndex output from yfinance.")

    raw = {}
    available = set(data.columns.get_level_values(0))
    for name, ticker in TICKERS.items():
        if ticker not in available:
            raise RuntimeError(f"Ticker {ticker} for {name} was not returned by yfinance.")
        if "Close" not in data[ticker].columns:
            raise RuntimeError(f"'Close' column missing for {ticker} / {name}.")

        series = data[ticker]["Close"].dropna()
        if series.empty:
            raise RuntimeError(f"No valid Close prices found for {ticker} / {name}.")
        raw[name] = series

    prices = pd.DataFrame(raw)[COLUMN_ORDER].sort_index().dropna()
    if prices.empty:
        raise RuntimeError("No aligned multi-asset price rows remained after cleaning.")

    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices / prices.shift(1)).dropna()
    if returns.empty:
        raise RuntimeError("Returns are empty after log-return preprocessing.")
    if returns.isna().any().any():
        raise RuntimeError("NaNs found in returns after preprocessing.")
    return returns.astype(np.float32)


def create_sequences(data: np.ndarray, window: int = WINDOW) -> np.ndarray:
    if len(data) < window:
        raise ValueError(f"Not enough rows ({len(data)}) for window size {window}.")
    return np.stack([data[i : i + window] for i in range(len(data) - window + 1)]).astype(
        np.float32
    )


def build_raw_feat(
    raw_returns: np.ndarray,
    vix_prices: np.ndarray,
    window: int = WINDOW,
) -> np.ndarray:
    if len(raw_returns) != len(vix_prices):
        raise ValueError("raw_returns and vix_prices must have the same length.")

    raw_feat = []
    for i in range(len(raw_returns) - window + 1):
        window_returns = raw_returns[i : i + window]
        window_vix = vix_prices[i : i + window]
        raw_feat.append(
            [
                np.mean(window_returns[:, 0]),
                np.mean(window_returns[:, 1]),
                np.std(window_returns[:, 0]),
                np.mean(window_returns[:, 4]),
                np.mean(window_vix),
                np.max(window_vix),
            ]
        )

    return np.asarray(raw_feat, dtype=np.float32)


def build_supervised_features(
    raw_feat: np.ndarray,
    latent_pca: np.ndarray,
) -> Dict[str, np.ndarray]:
    if len(raw_feat) != len(latent_pca):
        raise ValueError("raw_feat and latent_pca must have the same length.")

    momentum_5 = np.asarray(
        [np.mean(raw_feat[max(0, i - 5) : i + 1, 0]) for i in range(len(raw_feat))],
        dtype=np.float32,
    )
    momentum_10 = np.asarray(
        [np.mean(raw_feat[max(0, i - 10) : i + 1, 0]) for i in range(len(raw_feat))],
        dtype=np.float32,
    )
    vix_change_5 = np.asarray(
        [raw_feat[i, 4] - raw_feat[max(0, i - 5), 4] for i in range(len(raw_feat))],
        dtype=np.float32,
    )
    vol_change_5 = np.asarray(
        [raw_feat[i, 2] - raw_feat[max(0, i - 5), 2] for i in range(len(raw_feat))],
        dtype=np.float32,
    )

    multi_asset = np.column_stack(
        [
            latent_pca.astype(np.float32),
            raw_feat.astype(np.float32),
            momentum_5,
            momentum_10,
            vix_change_5,
            vol_change_5,
        ]
    ).astype(np.float32)

    single_asset = np.column_stack(
        [
            raw_feat[:, 0],
            raw_feat[:, 2],
            momentum_5,
            momentum_10,
            vol_change_5,
        ]
    ).astype(np.float32)

    return {
        "multi_asset": multi_asset,
        "single_asset": single_asset,
    }


def build_next_day_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    dates: Iterable[str],
    seq_len: int = SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length.")

    dates = np.asarray(list(dates))
    if len(dates) != len(labels):
        raise ValueError("dates and labels must have the same length.")

    x_all = features[:-1]
    y_all = labels[1:]
    target_dates = dates[1:]

    x_seq = []
    y_seq = []
    current_seq = []
    date_seq = []

    for i in range(len(x_all) - seq_len + 1):
        end = i + seq_len - 1
        x_seq.append(x_all[i : i + seq_len])
        y_seq.append(y_all[end])
        current_seq.append(labels[end])
        date_seq.append(target_dates[end])

    return (
        np.asarray(x_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.int64),
        np.asarray(current_seq, dtype=np.int64),
        np.asarray(date_seq),
    )


def scale_sequence_batch(scaler, batch_3d: np.ndarray) -> np.ndarray:
    flat = batch_3d.reshape(-1, batch_3d.shape[-1])
    scaled = scaler.transform(flat)
    return scaled.reshape(batch_3d.shape).astype(np.float32)


def build_sticky_transmat(n_states: int, stay_prob: float) -> np.ndarray:
    if n_states < 2:
        raise ValueError("n_states must be at least 2.")
    off_diag = (1.0 - stay_prob) / (n_states - 1)
    transmat = np.full((n_states, n_states), off_diag, dtype=np.float64)
    np.fill_diagonal(transmat, stay_prob)
    return transmat


def fit_best_hmm(
    features_train: np.ndarray,
    n_components: int = HMM_COMPONENTS,
    restarts: int = HMM_RESTARTS,
    stay_prob: float = HMM_STAY_PROB,
    seed: int = SEED,
) -> hmm.GaussianHMM:
    best_model = None
    best_score = -np.inf
    best_unique = 0

    for offset in range(restarts):
        current_seed = seed + offset
        kmeans = KMeans(n_clusters=n_components, random_state=current_seed, n_init=20)
        kmeans.fit(features_train)

        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=400,
            tol=1e-3,
            random_state=current_seed,
            params="stmc",
            init_params="sc",
            min_covar=1e-4,
        )
        model.transmat_ = build_sticky_transmat(n_components, stay_prob)
        model.means_ = kmeans.cluster_centers_
        model.fit(features_train)

        train_labels = model.predict(features_train)
        unique_count = len(np.unique(train_labels))
        score = model.score(features_train)

        if unique_count > best_unique or (unique_count == best_unique and score > best_score):
            best_unique = unique_count
            best_score = score
            best_model = model

    if best_model is None:
        raise RuntimeError("Failed to fit any HMM model.")
    if best_unique < n_components:
        raise RuntimeError(
            f"HMM only activated {best_unique}/{n_components} states. "
            "Adjust the feature set or restart settings."
        )

    return best_model


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std == 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return (values - float(np.mean(values))) / std


def compute_state_statistics(
    labels: np.ndarray,
    raw_feat: np.ndarray,
    mask: np.ndarray | None = None,
) -> Dict[int, Dict[str, float]]:
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)

    stats: Dict[int, Dict[str, float]] = {}
    for state in sorted(np.unique(labels)):
        state_mask = (labels == state) & mask
        if not np.any(state_mask):
            raise ValueError(f"State {state} has no rows under the requested mask.")

        stats[int(state)] = {
            "count": int(np.sum(state_mask)),
            "sp_return": float(np.mean(raw_feat[state_mask, 0])),
            "sp_vol": float(np.mean(raw_feat[state_mask, 2])),
            "vix_level": float(np.mean(raw_feat[state_mask, 4])),
            "vix_max": float(np.mean(raw_feat[state_mask, 5])),
        }

    return stats


def canonicalize_hmm_outputs(
    model: hmm.GaussianHMM,
    labels: np.ndarray,
    posteriors: np.ndarray,
    raw_feat: np.ndarray,
    fit_mask: np.ndarray,
) -> Tuple[hmm.GaussianHMM, np.ndarray, np.ndarray, Dict[int, Dict[str, float]]]:
    original_stats = compute_state_statistics(labels, raw_feat, mask=fit_mask)
    states = np.asarray(sorted(original_stats))

    vix_levels = np.asarray([original_stats[int(state)]["vix_level"] for state in states])
    vols = np.asarray([original_stats[int(state)]["sp_vol"] for state in states])
    returns = np.asarray([original_stats[int(state)]["sp_return"] for state in states])

    stress_score = _zscore(vix_levels) + _zscore(vols) - _zscore(returns)
    crisis_old = int(states[np.argmax(stress_score)])
    calm_old = int(states[np.argmin(stress_score)])

    middle = [state for state in states.tolist() if state not in {calm_old, crisis_old}]
    if len(middle) != 1:
        raise RuntimeError("Could not identify a unique middle HMM state.")
    risk_old = int(middle[0])

    order = [calm_old, risk_old, crisis_old]
    old_to_new = {old_state: new_state for new_state, old_state in enumerate(order)}

    remapped_labels = np.asarray([old_to_new[int(label)] for label in labels], dtype=np.int64)
    remapped_posteriors = posteriors[:, order].astype(np.float32)

    model.startprob_ = model.startprob_[order]
    model.transmat_ = model.transmat_[order][:, order]
    model.means_ = model.means_[order]

    if model.covariance_type == "full":
        model.covars_ = model.covars_[order]
    elif model.covariance_type == "diag":
        model.covars_ = model.covars_[order]
    elif model.covariance_type == "spherical":
        model.covars_ = model.covars_[order]
    elif model.covariance_type == "tied":
        model.covars_ = model.covars_

    remapped_stats = {
        0: original_stats[calm_old],
        1: original_stats[risk_old],
        2: original_stats[crisis_old],
    }
    return model, remapped_labels, remapped_posteriors, remapped_stats


def load_autoencoder_model(
    path: str,
    device: torch.device | None = None,
) -> LSTMAutoencoder:
    device = get_device() if device is None else device
    model = LSTMAutoencoder().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_classifier_model(
    path: str,
    input_size: int,
    device: torch.device | None = None,
) -> RegimeClassifier:
    device = get_device() if device is None else device
    model = RegimeClassifier(input_size=input_size).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def extract_latent_vectors(
    scaled_data: np.ndarray,
    model: LSTMAutoencoder,
    window: int = WINDOW,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> np.ndarray:
    device = get_device() if device is None else device
    sequences = create_sequences(scaled_data, window=window)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(sequences)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    latent_list = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            _, latent = model(batch)
            latent_list.append(latent.cpu().numpy())

    return np.concatenate(latent_list, axis=0).astype(np.float32)


def infer_proxy_states_from_prices(
    prices: pd.DataFrame,
    return_scaler,
    autoencoder: LSTMAutoencoder,
    pca,
    hmm_feat_scaler,
    hmm_model: hmm.GaussianHMM,
    window: int = WINDOW,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> Dict[str, np.ndarray]:
    device = get_device() if device is None else device

    returns = compute_log_returns(prices)
    if len(returns) < window:
        raise ValueError(f"Need at least {window} return rows, got {len(returns)}.")

    raw_returns = returns.to_numpy(dtype=np.float32)
    scaled_returns = return_scaler.transform(raw_returns).astype(np.float32)
    vix_prices = prices.loc[returns.index, "VIX"].to_numpy(dtype=np.float32)
    window_end_dates = returns.index[window - 1 :].astype(str).to_numpy()

    latent_vectors = extract_latent_vectors(
        scaled_data=scaled_returns,
        model=autoencoder,
        window=window,
        batch_size=batch_size,
        device=device,
    )
    latent_pca = pca.transform(latent_vectors).astype(np.float32)
    raw_feat = build_raw_feat(raw_returns, vix_prices, window=window)
    hmm_features = np.hstack([latent_pca, raw_feat]).astype(np.float32)
    hmm_features_scaled = hmm_feat_scaler.transform(hmm_features).astype(np.float32)
    regime_posteriors = hmm_model.predict_proba(hmm_features_scaled).astype(np.float32)
    regime_labels = hmm_model.predict(hmm_features_scaled).astype(np.int64)

    if not (
        len(window_end_dates)
        == len(raw_feat)
        == len(latent_vectors)
        == len(latent_pca)
        == len(regime_labels)
        == len(regime_posteriors)
    ):
        raise RuntimeError("Proxy-state inference produced misaligned arrays.")

    return {
        "returns_index": returns.index.astype(str).to_numpy(),
        "window_end_dates": window_end_dates,
        "raw_returns": raw_returns,
        "scaled_returns": scaled_returns,
        "vix_prices": vix_prices,
        "latent_vectors": latent_vectors,
        "latent_pca": latent_pca,
        "raw_feat": raw_feat,
        "hmm_features": hmm_features,
        "hmm_features_scaled": hmm_features_scaled,
        "regime_labels": regime_labels,
        "regime_posteriors": regime_posteriors,
    }


def load_regime_name_mapping(path: str) -> Dict[int, str]:
    values = np.load(path, allow_pickle=True)
    return {int(key): str(value) for key, value in values}


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    current_labels: np.ndarray,
    regime_names: Dict[int, str],
    probabilities: np.ndarray | None = None,
    high_conf_threshold: float = HIGH_CONF_THRESHOLD,
) -> Dict[str, object]:
    label_order = sorted(regime_names)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=label_order, average="macro")
    report = classification_report(
        y_true,
        y_pred,
        labels=label_order,
        target_names=[regime_names[label] for label in label_order],
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_order)

    transition_mask = y_true != current_labels
    transition_total = int(np.sum(transition_mask))
    transition_hits = int(np.sum((y_true == y_pred) & transition_mask))
    transition_hit_rate = (
        transition_hits / transition_total if transition_total else float("nan")
    )

    high_conf_days = 0
    high_conf_accuracy = float("nan")
    confidence = None
    if probabilities is not None:
        confidence = np.max(probabilities, axis=1)
        high_conf_mask = confidence >= high_conf_threshold
        high_conf_days = int(np.sum(high_conf_mask))
        if np.any(high_conf_mask):
            high_conf_accuracy = accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask])

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "report": report,
        "confusion_matrix": cm,
        "transition_total": transition_total,
        "transition_hits": transition_hits,
        "transition_hit_rate": float(transition_hit_rate),
        "high_conf_days": int(high_conf_days),
        "high_conf_accuracy": float(high_conf_accuracy),
        "confidence": confidence,
    }


def build_prediction_frame(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    current_labels: np.ndarray,
    regime_names: Dict[int, str],
    probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    if probabilities is None:
        confidence = np.full(len(y_pred), np.nan, dtype=np.float32)
        prob_columns = {
            f"prob_{regime_names[label].lower().replace(' ', '_')}": np.full(
                len(y_pred), np.nan, dtype=np.float32
            )
            for label in sorted(regime_names)
        }
    else:
        confidence = np.max(probabilities, axis=1).astype(np.float32)
        prob_columns = {
            f"prob_{regime_names[label].lower().replace(' ', '_')}": probabilities[:, label].astype(
                np.float32
            )
            for label in sorted(regime_names)
        }

    frame = pd.DataFrame(
        {
            "date": np.asarray(dates),
            "predicted_label": y_pred.astype(np.int64),
            "predicted_name": [regime_names[int(label)] for label in y_pred],
            "target_label": y_true.astype(np.int64),
            "target_name": [regime_names[int(label)] for label in y_true],
            "current_label": current_labels.astype(np.int64),
            "current_name": [regime_names[int(label)] for label in current_labels],
            "confidence": confidence,
            "correct": (y_pred == y_true),
        }
    )

    for column, values in prob_columns.items():
        frame[column] = values

    return frame


def format_confusion_matrix(cm: np.ndarray, regime_names: Dict[int, str]) -> pd.DataFrame:
    labels = [regime_names[idx] for idx in sorted(regime_names)]
    return pd.DataFrame(cm, index=labels, columns=labels)

