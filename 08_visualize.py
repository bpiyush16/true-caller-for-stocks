import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_utils import load_regime_name_mapping

warnings.filterwarnings("ignore")

regime_names = load_regime_name_mapping("supervised_regime_names.npy")
multi_df = pd.read_csv("backtest_predictions_multi_asset.csv")
single_df = pd.read_csv("backtest_predictions_single_asset.csv")
persistence_df = pd.read_csv("backtest_predictions_persistence.csv")

if multi_df.empty or single_df.empty or persistence_df.empty:
    raise ValueError("At least one backtest prediction file is empty.")

for frame in [multi_df, single_df, persistence_df]:
    frame["date"] = pd.to_datetime(frame["date"])
    frame.sort_values("date", inplace=True)
    frame.reset_index(drop=True, inplace=True)

if not multi_df["date"].equals(single_df["date"]) or not multi_df["date"].equals(persistence_df["date"]):
    raise ValueError("Backtest prediction files do not share identical dates.")

dates = multi_df["date"]
target = multi_df["target_label"].to_numpy(dtype=np.int64)
current = multi_df["current_label"].to_numpy(dtype=np.int64)
pred_multi = multi_df["predicted_label"].to_numpy(dtype=np.int64)
pred_single = single_df["predicted_label"].to_numpy(dtype=np.int64)
pred_persistence = persistence_df["predicted_label"].to_numpy(dtype=np.int64)
conf_multi = multi_df["confidence"].to_numpy(dtype=np.float32)
conf_single = single_df["confidence"].to_numpy(dtype=np.float32)

rolling_window = min(30, len(dates))
roll_multi = pd.Series((pred_multi == target).astype(float)).rolling(rolling_window, min_periods=1).mean()
roll_single = pd.Series((pred_single == target).astype(float)).rolling(rolling_window, min_periods=1).mean()
roll_persistence = pd.Series((pred_persistence == target).astype(float)).rolling(
    rolling_window, min_periods=1
).mean()

transitions = target != current

print("=" * 60)
print("STAGE 8 - VISUALIZE MULTI-ASSET VS SINGLE-ASSET RESULTS")
print("=" * 60)

print("\nStep 1: Creating regime comparison figure...")
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

axes[0].step(dates, target, where="post", color="black", linewidth=1.2, label="HMM proxy target")
axes[0].step(dates, pred_multi, where="post", color="#264653", linewidth=1.0, linestyle="--", label="Multi-Asset")
axes[0].step(dates, pred_single, where="post", color="#d62828", linewidth=0.9, linestyle=":", label="Single-Asset")
axes[0].step(
    dates,
    pred_persistence,
    where="post",
    color="#8d99ae",
    linewidth=0.9,
    linestyle="-.",
    label="Persistence",
)
axes[0].set_title("Proxy Target vs Predicted Next-Day Regimes")
axes[0].set_ylabel("Regime")
axes[0].set_yticks(sorted(regime_names))
axes[0].set_yticklabels([regime_names[idx] for idx in sorted(regime_names)])
axes[0].legend(loc="upper left")

axes[1].plot(dates, roll_multi, color="#264653", linewidth=1.2, label="Multi-Asset")
axes[1].plot(dates, roll_single, color="#d62828", linewidth=1.2, label="Single-Asset")
axes[1].plot(dates, roll_persistence, color="#8d99ae", linewidth=1.2, label="Persistence")
axes[1].set_ylim(0, 1.02)
axes[1].set_ylabel(f"{rolling_window}-day Agreement")
axes[1].set_title(f"Rolling Agreement vs HMM Proxy ({rolling_window}-day window)")
axes[1].legend(loc="lower left")

axes[2].plot(dates, conf_multi, color="#264653", linewidth=1.0, label="Multi-Asset confidence")
axes[2].plot(dates, conf_single, color="#d62828", linewidth=1.0, label="Single-Asset confidence")
axes[2].axhline(0.80, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
for idx in np.where(transitions)[0]:
    axes[2].axvline(dates.iloc[idx], color="black", linewidth=0.8, alpha=0.2)
axes[2].set_ylim(0, 1.02)
axes[2].set_ylabel("Confidence")
axes[2].set_xlabel("Date")
axes[2].set_title("Model Confidence (black lines = HMM proxy transitions)")
axes[2].legend(loc="lower left")

plt.tight_layout()
plt.savefig("regimes_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: regimes_comparison.png")

print("\nStep 2: Creating confidence breakdown figure...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(conf_multi, bins=30, alpha=0.75, color="#264653")
axes[0].set_title("Multi-Asset Confidence Distribution")
axes[0].set_xlabel("Confidence")
axes[0].set_ylabel("Days")

axes[1].hist(conf_single, bins=30, alpha=0.75, color="#d62828")
axes[1].set_title("Single-Asset Confidence Distribution")
axes[1].set_xlabel("Confidence")
axes[1].set_ylabel("Days")

plt.tight_layout()
plt.savefig("confidence_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: confidence_comparison.png")

print("\n" + "=" * 60)
print("STAGE 8 COMPLETE")
print("  Saved: regimes_comparison.png")
print("  Saved: confidence_comparison.png")
print("\nNext: Run 09_predict_tomorrow.py")
print("=" * 60)
