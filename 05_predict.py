import warnings

import joblib
import numpy as np
import pandas as pd
import torch

from pipeline_utils import (
    HIGH_CONF_THRESHOLD,
    REGIME_NAMES,
    SEQ_LEN,
    START_DATE,
    VARIANT_CONFIGS,
    WINDOW,
    WARMUP_CALENDAR_DAYS,
    build_next_day_sequences,
    build_supervised_features,
    download_market_prices,
    evaluate_predictions,
    get_device,
    infer_proxy_states_from_prices,
    load_autoencoder_model,
    load_classifier_model,
    load_regime_name_mapping,
    scale_sequence_batch,
)

warnings.filterwarnings("ignore")

START_EVAL_DATE = "2023-01-01"
END_EVAL_EXCLUSIVE = "2024-01-01"

device = get_device()

print("=" * 60)
print("STAGE 5 - PREDICT SUPERVISED REGIMES")
print("=" * 60)

print("\nStep 1: Loading trained artifacts...")
return_scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca_model.pkl")
hmm_feat_scaler = joblib.load("feat_scaler.pkl")
hmm_model = joblib.load("hmm_model.pkl")
predictor_scaler = joblib.load("supervised_predictor_scaler.pkl")
regime_names = load_regime_name_mapping("supervised_regime_names.npy")

autoencoder = load_autoencoder_model("lstm_model.pth", device=device)
predictor = load_classifier_model(
    path="supervised_regime_predictor.pth",
    input_size=15,
    device=device,
)
print("  Autoencoder, HMM, and supervised predictor loaded.")

print("\nStep 2: Downloading evaluation prices...")
download_start = (
    pd.Timestamp(START_EVAL_DATE) - pd.Timedelta(days=WARMUP_CALENDAR_DAYS)
).strftime("%Y-%m-%d")
prices = download_market_prices(start=download_start, end=END_EVAL_EXCLUSIVE, threads=False)
print(f"  Clean prices shape: {prices.shape}")

print("\nStep 3: Inferring HMM proxy regimes for the downloaded range...")
proxy_bundle = infer_proxy_states_from_prices(
    prices=prices,
    return_scaler=return_scaler,
    autoencoder=autoencoder,
    pca=pca,
    hmm_feat_scaler=hmm_feat_scaler,
    hmm_model=hmm_model,
    window=WINDOW,
    batch_size=256,
    device=device,
)

feature_sets = build_supervised_features(
    raw_feat=proxy_bundle["raw_feat"],
    latent_pca=proxy_bundle["latent_pca"],
)

x_seq, y_seq, current_seq, date_seq = build_next_day_sequences(
    features=feature_sets["multi_asset"],
    labels=proxy_bundle["regime_labels"],
    dates=proxy_bundle["window_end_dates"],
    seq_len=SEQ_LEN,
)
x_seq_scaled = scale_sequence_batch(predictor_scaler, x_seq)

print(f"  Proxy regime windows: {len(proxy_bundle['regime_labels'])}")
print(f"  Sequence shape:       {x_seq.shape}")

print("\nStep 4: Forecasting the next-day HMM proxy regime...")
with torch.no_grad():
    logits = predictor(torch.tensor(x_seq_scaled, dtype=torch.float32).to(device))
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)

mask = (date_seq >= START_EVAL_DATE) & (date_seq < END_EVAL_EXCLUSIVE)
pred_dates = date_seq[mask]
y_true = y_seq[mask]
y_pred = preds[mask]
current_labels = current_seq[mask]
probs = probs[mask]

if len(pred_dates) == 0:
    raise ValueError("No predictions fell inside the requested date range.")

model_metrics = evaluate_predictions(
    y_true=y_true,
    y_pred=y_pred,
    current_labels=current_labels,
    regime_names=regime_names,
    probabilities=probs,
)
baseline_metrics = evaluate_predictions(
    y_true=y_true,
    y_pred=current_labels,
    current_labels=current_labels,
    regime_names=regime_names,
    probabilities=None,
)

print(f"  Agreement vs HMM proxy: {model_metrics['accuracy']:.4f} ({model_metrics['accuracy'] * 100:.1f}%)")
print(f"  Macro-F1:              {model_metrics['macro_f1']:.4f}")
print(
    f"  Persistence baseline:  {baseline_metrics['accuracy']:.4f} "
    f"({baseline_metrics['accuracy'] * 100:.1f}%)"
)
print(
    f"  High-confidence days:  {model_metrics['high_conf_days']} "
    f"(threshold={HIGH_CONF_THRESHOLD:.2f})"
)

print(f"\n{'Date':<15} {'Predicted':<14} {'Proxy Target':<14} {'Current Proxy'}")
print("-" * 62)
for date, pred, target, current in zip(pred_dates, y_pred, y_true, current_labels):
    print(
        f"{date:<15} {regime_names[int(pred)]:<14} "
        f"{regime_names[int(target)]:<14} {regime_names[int(current)]}"
    )

print("\nClassification Report (agreement vs HMM proxy):")
print(model_metrics["report"])

summary_lines = [
    "Summary by predicted regime:",
]
for label in sorted(regime_names):
    count = int(np.sum(y_pred == label))
    summary_lines.append(
        f"  {regime_names[label]:<12} {count:>5} days ({count / len(y_pred) * 100:.1f}%)"
    )
print("\n" + "\n".join(summary_lines))

with open("predictor_results.txt", "w", encoding="utf-8") as report:
    report.write("STAGE 5 - RANGE PREDICTION SUMMARY\n")
    report.write("=" * 60 + "\n")
    report.write(f"Range: {START_EVAL_DATE} to {(pd.Timestamp(END_EVAL_EXCLUSIVE) - pd.Timedelta(days=1)).date()}\n")
    report.write(f"Agreement vs HMM proxy: {model_metrics['accuracy']:.4f}\n")
    report.write(f"Macro-F1: {model_metrics['macro_f1']:.4f}\n")
    report.write(
        f"Persistence baseline: {baseline_metrics['accuracy']:.4f} "
        f"({baseline_metrics['accuracy'] * 100:.1f}%)\n\n"
    )
    report.write(model_metrics["report"])

print("\n" + "=" * 60)
print("STAGE 5 COMPLETE")
print("Saved: predictor_results.txt")
print("Next: Run 06_backtest.py")
print("=" * 60)
