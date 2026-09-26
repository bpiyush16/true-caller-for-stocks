import warnings

import pandas as pd

from pipeline_utils import HIGH_CONF_THRESHOLD, VARIANT_CONFIGS, evaluate_predictions, load_regime_name_mapping

warnings.filterwarnings("ignore")

print("=" * 60)
print("STAGE 7 - EVALUATE MULTI-ASSET VS SINGLE-ASSET RESULTS")
print("=" * 60)

print("\nStep 1: Loading backtest predictions...")
regime_names = load_regime_name_mapping("supervised_regime_names.npy")
variant_frames = {
    "multi_asset": pd.read_csv(VARIANT_CONFIGS["multi_asset"]["csv_path"]),
    "single_asset": pd.read_csv(VARIANT_CONFIGS["single_asset"]["csv_path"]),
    "persistence": pd.read_csv("backtest_predictions_persistence.csv"),
}

for variant_name, frame in variant_frames.items():
    if frame.empty:
        raise ValueError(f"{variant_name} backtest frame is empty.")
    print(f"  {variant_name:<12} {len(frame)} rows")

shared_dates = set(variant_frames["multi_asset"]["date"]) & set(variant_frames["single_asset"]["date"])
if not shared_dates:
    raise ValueError("Multi-asset and single-asset backtests do not share dates.")

print("\nStep 2: Computing evaluation metrics...")
results = {}
for variant_name, frame in variant_frames.items():
    probs = None
    prob_columns = [column for column in frame.columns if column.startswith("prob_")]
    if prob_columns and frame[prob_columns].notna().any().any():
        probs = frame[prob_columns].to_numpy(dtype="float32")

    results[variant_name] = evaluate_predictions(
        y_true=frame["target_label"].to_numpy(dtype="int64"),
        y_pred=frame["predicted_label"].to_numpy(dtype="int64"),
        current_labels=frame["current_label"].to_numpy(dtype="int64"),
        regime_names=regime_names,
        probabilities=probs,
    )

print(f"\n{'Variant':<22} {'Agreement':<12} {'Macro-F1':<12} {'Hi-Conf':<14} {'Transition Hit'}")
print("-" * 86)
for variant_name, label in [
    ("multi_asset", "Multi-Asset"),
    ("single_asset", "Single-Asset"),
    ("persistence", "Persistence"),
]:
    metrics = results[variant_name]
    high_conf_text = (
        f"{metrics['high_conf_accuracy'] * 100:>9.2f}%"
        if metrics["high_conf_days"] > 0
        else "n/a"
    )
    print(
        f"{label:<22} "
        f"{metrics['accuracy'] * 100:>7.2f}%    "
        f"{metrics['macro_f1']:<12.4f} "
        f"{high_conf_text:<14} "
        f"{metrics['transition_hits']}/{metrics['transition_total']} ({metrics['transition_hit_rate'] * 100:.1f}%)"
    )

multi = results["multi_asset"]
single = results["single_asset"]
persistence = results["persistence"]

print("\nImprovement (multi-asset minus comparator):")
print(f"  Vs single-asset agreement gap:   {(multi['accuracy'] - single['accuracy']) * 100:+.2f} percentage points")
print(f"  Vs persistence agreement gap:    {(multi['accuracy'] - persistence['accuracy']) * 100:+.2f} percentage points")
print(f"  Vs single-asset Macro-F1 gap:    {(multi['macro_f1'] - single['macro_f1']):+.4f}")

print("\nStep 3: Saving report...")
results_text = f"""
EVALUATION RESULTS - TARGET = FROZEN HMM PROXY STATE
============================================================
Days evaluated: {len(variant_frames['multi_asset'])}
High-confidence threshold: {HIGH_CONF_THRESHOLD:.2f}

Multi-Asset:
  Agreement:       {multi['accuracy']:.4f} ({multi['accuracy'] * 100:.2f}%)
  Macro-F1:        {multi['macro_f1']:.4f}
  High-conf days:  {multi['high_conf_days']}
  High-conf agree: {multi['high_conf_accuracy']:.4f}
  Transition hit:  {multi['transition_hits']}/{multi['transition_total']} ({multi['transition_hit_rate'] * 100:.1f}%)

Single-Asset:
  Agreement:       {single['accuracy']:.4f} ({single['accuracy'] * 100:.2f}%)
  Macro-F1:        {single['macro_f1']:.4f}
  High-conf days:  {single['high_conf_days']}
  High-conf agree: {single['high_conf_accuracy']:.4f}
  Transition hit:  {single['transition_hits']}/{single['transition_total']} ({single['transition_hit_rate'] * 100:.1f}%)

Persistence baseline:
  Agreement:       {persistence['accuracy']:.4f} ({persistence['accuracy'] * 100:.2f}%)
  Macro-F1:        {persistence['macro_f1']:.4f}
  Transition hit:  {persistence['transition_hits']}/{persistence['transition_total']} ({persistence['transition_hit_rate'] * 100:.1f}%)

Improvement (multi-asset minus single-asset):
  Agreement gap:   {(multi['accuracy'] - single['accuracy']) * 100:+.2f} percentage points
  Macro-F1 gap:    {(multi['macro_f1'] - single['macro_f1']):+.4f}

Improvement (multi-asset minus persistence):
  Agreement gap:   {(multi['accuracy'] - persistence['accuracy']) * 100:+.2f} percentage points
  Macro-F1 gap:    {(multi['macro_f1'] - persistence['macro_f1']):+.4f}

Multi-Asset Classification Report:
{multi['report']}

Single-Asset Classification Report:
{single['report']}
"""

with open("evaluation_results.txt", "w", encoding="utf-8") as handle:
    handle.write(results_text.strip() + "\n")

print("  Saved: evaluation_results.txt")
print("\n" + "=" * 60)
print("STAGE 7 COMPLETE")
print("  Saved: evaluation_results.txt")
print("\nNext: Run 08_visualize.py")
print("=" * 60)
