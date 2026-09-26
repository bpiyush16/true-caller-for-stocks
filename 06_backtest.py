import warnings
from datetime import datetime, timedelta

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pipeline_utils import (
    CLASS_COLORS,
    HIGH_CONF_THRESHOLD,
    REGIME_NAMES,
    SEQ_LEN,
    VARIANT_CONFIGS,
    WINDOW,
    WARMUP_CALENDAR_DAYS,
    build_next_day_sequences,
    build_prediction_frame,
    build_supervised_features,
    download_market_prices,
    evaluate_predictions,
    format_confusion_matrix,
    get_device,
    infer_proxy_states_from_prices,
    load_autoencoder_model,
    load_classifier_model,
    load_regime_name_mapping,
    scale_sequence_batch,
)

warnings.filterwarnings("ignore")

BACKTEST_START = "2025-01-01"
BACKTEST_END_EXCLUSIVE = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
BATCH_SIZE = 128

device = get_device()

print("=" * 60)
print("STAGE 6 - BACKTEST SUPERVISED TOMORROW PREDICTOR")
print("=" * 60)
print(
    f"\nBacktest period: {BACKTEST_START} to "
    f"{(pd.Timestamp(BACKTEST_END_EXCLUSIVE) - pd.Timedelta(days=1)).date()}"
)

print("\nStep 1: Loading trained artifacts...")
return_scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca_model.pkl")
hmm_feat_scaler = joblib.load("feat_scaler.pkl")
hmm_model = joblib.load("hmm_model.pkl")
regime_names = load_regime_name_mapping("supervised_regime_names.npy")

autoencoder = load_autoencoder_model("lstm_model.pth", device=device)
variant_models = {}
variant_scalers = {}
variant_input_sizes = {
    "multi_asset": 15,
    "single_asset": 5,
}
for variant_name in ["multi_asset", "single_asset"]:
    variant_models[variant_name] = load_classifier_model(
        path=VARIANT_CONFIGS[variant_name]["model_path"],
        input_size=variant_input_sizes[variant_name],
        device=device,
    )
    variant_scalers[variant_name] = joblib.load(VARIANT_CONFIGS[variant_name]["scaler_path"])
print("  Autoencoder, HMM, and supervised predictors loaded.")

print("\nStep 2: Downloading historical prices for the backtest...")
download_start = (
    pd.Timestamp(BACKTEST_START) - pd.Timedelta(days=WARMUP_CALENDAR_DAYS)
).strftime("%Y-%m-%d")
prices = download_market_prices(start=download_start, end=BACKTEST_END_EXCLUSIVE, threads=False)
print(
    f"  Download range used: {download_start} to "
    f"{(pd.Timestamp(BACKTEST_END_EXCLUSIVE) - pd.Timedelta(days=1)).date()}"
)
print(f"  Clean prices shape:  {prices.shape}")

print("\nStep 3: Building out-of-sample HMM proxy states...")
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

sequence_payload = {}
for variant_name, features in feature_sets.items():
    sequence_payload[variant_name] = build_next_day_sequences(
        features=features,
        labels=proxy_bundle["regime_labels"],
        dates=proxy_bundle["window_end_dates"],
        seq_len=SEQ_LEN,
    )

x_multi, y_seq, current_seq, date_seq = sequence_payload["multi_asset"]
x_single, y_single, current_single, date_single = sequence_payload["single_asset"]
if not (
    np.array_equal(y_seq, y_single)
    and np.array_equal(current_seq, current_single)
    and np.array_equal(date_seq, date_single)
):
    raise RuntimeError("Variant sequence alignment mismatch detected.")

target_dates = pd.to_datetime(date_seq)
mask = (
    (target_dates >= pd.Timestamp(BACKTEST_START))
    & (target_dates < pd.Timestamp(BACKTEST_END_EXCLUSIVE))
)
if not np.any(mask):
    raise ValueError("No backtest targets fell inside the requested date range.")

y_eval = y_seq[mask]
current_eval = current_seq[mask]
date_eval = target_dates[mask]

print(f"  Proxy regime windows: {len(proxy_bundle['regime_labels'])}")
print(f"  Sequence shape:       {x_multi.shape}")
print(f"  Backtest targets:     {len(y_eval)}")


def run_variant_backtest(variant_name: str, x_all: np.ndarray) -> dict:
    config = VARIANT_CONFIGS[variant_name]
    scaler = variant_scalers[variant_name]
    model = variant_models[variant_name]

    x_eval = x_all[mask]
    x_scaled = scale_sequence_batch(scaler, x_eval)

    pred_chunks = []
    prob_chunks = []
    with torch.no_grad():
        for start in range(0, len(x_scaled), BATCH_SIZE):
            batch = torch.tensor(x_scaled[start : start + BATCH_SIZE], dtype=torch.float32).to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            pred_chunks.append(preds)
            prob_chunks.append(probs)

    predicted = np.concatenate(pred_chunks)
    probabilities = np.concatenate(prob_chunks)
    metrics = evaluate_predictions(
        y_true=y_eval,
        y_pred=predicted,
        current_labels=current_eval,
        regime_names=regime_names,
        probabilities=probabilities,
    )
    frame = build_prediction_frame(
        dates=date_eval.strftime("%Y-%m-%d"),
        y_true=y_eval,
        y_pred=predicted,
        current_labels=current_eval,
        regime_names=regime_names,
        probabilities=probabilities,
    )
    frame.to_csv(config["csv_path"], index=False)

    return {
        "display_name": config["display_name"],
        "predicted": predicted,
        "probabilities": probabilities,
        "metrics": metrics,
        "frame": frame,
    }


print("\nStep 4: Running rolling next-day forecasts...")
variant_results = {
    "multi_asset": run_variant_backtest("multi_asset", x_multi),
    "single_asset": run_variant_backtest("single_asset", x_single),
}

for variant_name in ["multi_asset", "single_asset"]:
    result = variant_results[variant_name]
    metrics = result["metrics"]
    print(f"  {result['display_name']}:")
    print(f"    Agreement vs HMM proxy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.1f}%)")
    if metrics["transition_total"]:
        print(
            f"    Transition hit rate: {metrics['transition_hits']}/"
            f"{metrics['transition_total']} ({metrics['transition_hit_rate'] * 100:.1f}%)"
        )
    else:
        print("    Transition hit rate: No transitions in this range.")

baseline_metrics = evaluate_predictions(
    y_true=y_eval,
    y_pred=current_eval,
    current_labels=current_eval,
    regime_names=regime_names,
    probabilities=None,
)
baseline_frame = build_prediction_frame(
    dates=date_eval.strftime("%Y-%m-%d"),
    y_true=y_eval,
    y_pred=current_eval,
    current_labels=current_eval,
    regime_names=regime_names,
    probabilities=None,
)
baseline_frame.to_csv("backtest_predictions_persistence.csv", index=False)

multi_metrics = variant_results["multi_asset"]["metrics"]
single_metrics = variant_results["single_asset"]["metrics"]

comparison_text = (
    "MULTI-ASSET VS SINGLE-ASSET BACKTEST (TARGET = HMM PROXY STATE)\n"
    + "=" * 68
    + "\n"
    + f"Period: {BACKTEST_START} to {(pd.Timestamp(BACKTEST_END_EXCLUSIVE) - pd.Timedelta(days=1)).date()}\n"
    + f"Days evaluated: {len(y_eval)}\n"
    + f"Transitions in range: {multi_metrics['transition_total']}\n\n"
    + f"{'Variant':<28} {'Agreement':<12} {'Macro-F1':<12} {'Hi-Conf':<14} {'Transition Hit'}\n"
    + "-" * 90
    + "\n"
    + f"{variant_results['multi_asset']['display_name']:<28} "
    + f"{multi_metrics['accuracy'] * 100:>7.2f}%    "
    + f"{multi_metrics['macro_f1']:<12.4f} "
    + f"{multi_metrics['high_conf_accuracy'] * 100:>9.2f}%      "
    + f"{multi_metrics['transition_hits']}/{multi_metrics['transition_total']} ({multi_metrics['transition_hit_rate'] * 100:.1f}%)\n"
    + f"{variant_results['single_asset']['display_name']:<28} "
    + f"{single_metrics['accuracy'] * 100:>7.2f}%    "
    + f"{single_metrics['macro_f1']:<12.4f} "
    + f"{single_metrics['high_conf_accuracy'] * 100:>9.2f}%      "
    + f"{single_metrics['transition_hits']}/{single_metrics['transition_total']} ({single_metrics['transition_hit_rate'] * 100:.1f}%)\n"
    + f"{'Persistence baseline':<28} "
    + f"{baseline_metrics['accuracy'] * 100:>7.2f}%    "
    + f"{baseline_metrics['macro_f1']:<12.4f} "
    + f"{'n/a':>9}        "
    + f"{baseline_metrics['transition_hits']}/{baseline_metrics['transition_total']} ({baseline_metrics['transition_hit_rate'] * 100:.1f}%)\n\n"
    + f"Agreement gap (multi - single): {(multi_metrics['accuracy'] - single_metrics['accuracy']) * 100:+.2f} percentage points\n"
    + f"Agreement gap (multi - persistence): {(multi_metrics['accuracy'] - baseline_metrics['accuracy']) * 100:+.2f} percentage points\n"
    + f"Macro-F1 gap (multi - single): {(multi_metrics['macro_f1'] - single_metrics['macro_f1']):+.4f}\n"
)

with open("asset_mode_backtest_comparison.txt", "w", encoding="utf-8") as handle:
    handle.write(comparison_text)

print("\nComparison Summary:")
print(comparison_text)

multi_frame = variant_results["multi_asset"]["frame"]
multi_predicted = variant_results["multi_asset"]["predicted"]
multi_probabilities = variant_results["multi_asset"]["probabilities"]
multi_confidence = np.max(multi_probabilities, axis=1)

multi_frame.to_csv("backtest_predictions.csv", index=False)

results_text = f"""
SUPERVISED NEXT-DAY BACKTEST
============================================================
Target source:            Frozen Stage-3 HMM proxy state
Period:                   {BACKTEST_START} to {(pd.Timestamp(BACKTEST_END_EXCLUSIVE) - pd.Timedelta(days=1)).date()}
Days evaluated:           {len(multi_frame)}
Agreement vs proxy:       {multi_metrics['accuracy']:.4f} ({multi_metrics['accuracy'] * 100:.1f}%)
Macro-F1:                 {multi_metrics['macro_f1']:.4f}
Transitions in period:    {multi_metrics['transition_total']}
Transition hit rate:      {multi_metrics['transition_hits']}/{multi_metrics['transition_total']} ({multi_metrics['transition_hit_rate'] * 100:.1f}%)
High-conf days:           {multi_metrics['high_conf_days']}
High-conf agreement:      {multi_metrics['high_conf_accuracy']:.4f}

Persistence baseline:
  Agreement:              {baseline_metrics['accuracy']:.4f} ({baseline_metrics['accuracy'] * 100:.1f}%)
  Macro-F1:               {baseline_metrics['macro_f1']:.4f}

Multi-Asset Classification Report:
{multi_metrics['report']}

Multi-Asset Confusion Matrix:
{format_confusion_matrix(multi_metrics['confusion_matrix'], regime_names)}
"""

with open("backtest_results.txt", "w", encoding="utf-8") as handle:
    handle.write(results_text.strip() + "\n")

print("\nRecent multi-asset backtest sample:")
print(f"{'Date':<12} {'Predicted':<14} {'Proxy':<14} {'Current':<14} {'Conf'}")
print("-" * 68)
for row in multi_frame.tail(min(20, len(multi_frame))).itertuples(index=False):
    conf_text = "n/a" if np.isnan(row.confidence) else f"{row.confidence * 100:5.1f}%"
    print(
        f"{row.date:<12} {row.predicted_name:<14} {row.target_name:<14} "
        f"{row.current_name:<14} {conf_text}"
    )

print("\nStep 5: Generating diagnostic plot...")
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

axes[0].step(date_eval, y_eval, where="post", color="black", linewidth=1.0, label="HMM proxy target")
axes[0].step(
    date_eval,
    multi_predicted,
    where="post",
    color="#264653",
    linewidth=1.0,
    linestyle="--",
    label="Multi-asset forecast",
)
axes[0].step(
    date_eval,
    current_eval,
    where="post",
    color="#d62828",
    linewidth=0.9,
    linestyle=":",
    label="Persistence baseline",
)
axes[0].set_title("Backtest: HMM Proxy Target vs Next-Day Forecast")
axes[0].set_ylabel("Regime")
axes[0].set_yticks(sorted(regime_names))
axes[0].set_yticklabels([regime_names[idx] for idx in sorted(regime_names)])
axes[0].legend(loc="upper left")

for label in sorted(regime_names):
    axes[1].plot(
        date_eval,
        multi_probabilities[:, label],
        linewidth=0.9,
        label=regime_names[label],
        color=CLASS_COLORS[label],
    )
axes[1].axhline(HIGH_CONF_THRESHOLD, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("Probability")
axes[1].set_title("Predicted Class Probabilities")
axes[1].legend(loc="upper left")

errors = (multi_predicted != y_eval).astype(int)
transition_idx = np.where(y_eval != current_eval)[0]
axes[2].fill_between(
    date_eval,
    0,
    errors,
    step="post",
    color="#d62828",
    alpha=0.45,
    label="Wrong forecast",
)
axes[2].plot(date_eval, multi_confidence, color="#1d3557", linewidth=0.9, label="Confidence")
for idx in transition_idx:
    axes[2].axvline(date_eval[idx], color="black", linewidth=0.8, alpha=0.25)
axes[2].set_ylim(0, 1.05)
axes[2].set_ylabel("Error / Confidence")
axes[2].set_xlabel("Date")
axes[2].set_title("Forecast Errors and Confidence (black lines = HMM proxy transitions)")
axes[2].legend(loc="upper left")

plt.tight_layout()
plt.savefig("backtest_plot.png", dpi=150, bbox_inches="tight")
plt.close()

print("  Saved: backtest_results.txt")
print("  Saved: backtest_predictions.csv")
print("  Saved: backtest_predictions_multi_asset.csv")
print("  Saved: backtest_predictions_single_asset.csv")
print("  Saved: backtest_predictions_persistence.csv")
print("  Saved: asset_mode_backtest_comparison.txt")
print("  Saved: backtest_plot.png")

print("\n" + "=" * 60)
print("STAGE 6 COMPLETE")
print(f"  Multi-asset agreement vs proxy: {multi_metrics['accuracy'] * 100:.1f}%")
print(
    f"  Persistence baseline agreement: {baseline_metrics['accuracy'] * 100:.1f}%"
)
print("  Saved: backtest_results.txt")
print("  Saved: backtest_predictions.csv")
print("  Saved: backtest_predictions_multi_asset.csv")
print("  Saved: backtest_predictions_single_asset.csv")
print("  Saved: backtest_predictions_persistence.csv")
print("  Saved: asset_mode_backtest_comparison.txt")
print("  Saved: backtest_plot.png")
print("\nNext: Run 07_evaluate.py")
print("=" * 60)
