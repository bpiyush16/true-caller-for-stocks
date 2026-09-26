import warnings

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from pipeline_utils import (
    CLASSIFIER_BATCH_SIZE,
    CLASSIFIER_LR,
    CLASSIFIER_MAX_EPOCHS,
    CLASSIFIER_PATIENCE,
    CLASSIFIER_WEIGHT_DECAY,
    REGIME_NAMES,
    RegimeClassifier,
    TRAIN_END_DATE,
    VAL_END_DATE,
    VARIANT_CONFIGS,
    build_next_day_sequences,
    build_supervised_features,
    evaluate_predictions,
    load_regime_name_mapping,
    scale_sequence_batch,
    set_global_seed,
)

warnings.filterwarnings("ignore")

set_global_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_train_loader(x_train: np.ndarray, y_train: np.ndarray, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=CLASSIFIER_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )


print("=" * 60)
print("STAGE 4 - TRAIN SUPERVISED NEXT-DAY REGIME MODEL")
print("=" * 60)

print("\nStep 1: Loading HMM proxy labels and historical features...")
raw_feat = np.load("raw_feat.npy").astype(np.float32)
latent_pca = np.load("latent_pca.npy").astype(np.float32)
labels = np.load("regime_labels.npy").astype(np.int64)
dates = np.load("hmm_dates.npy", allow_pickle=True)
regime_names = load_regime_name_mapping("regime_names.npy")

if not (len(raw_feat) == len(latent_pca) == len(labels) == len(dates)):
    raise ValueError("raw_feat, latent_pca, labels, and dates must all have the same length.")

feature_sets = build_supervised_features(raw_feat=raw_feat, latent_pca=latent_pca)

np.save("supervised_labels.npy", labels)
np.save("supervised_dates.npy", dates)
np.save("supervised_regime_names.npy", np.array(list(regime_names.items()), dtype=object))

print(f"  Raw feature shape:   {raw_feat.shape}")
print(f"  PCA latent shape:    {latent_pca.shape}")
print(f"  Multi-asset shape:   {feature_sets['multi_asset'].shape}")
print(f"  Single-asset shape:  {feature_sets['single_asset'].shape}")
for label in sorted(regime_names):
    count = int(np.sum(labels == label))
    print(f"  {regime_names[label]:<12} {count:>5} windows ({count / len(labels) * 100:.1f}%)")

print("\nStep 2: Building next-day supervised sequences...")
sequence_payload = {}
for variant_name, features in feature_sets.items():
    sequence_payload[variant_name] = build_next_day_sequences(features, labels, dates)

x_multi, y_seq, current_seq, date_seq = sequence_payload["multi_asset"]
x_single, y_single, current_single, date_single = sequence_payload["single_asset"]
if not (
    np.array_equal(y_seq, y_single)
    and np.array_equal(current_seq, current_single)
    and np.array_equal(date_seq, date_single)
):
    raise RuntimeError("Variant sequence alignment mismatch detected.")

print(f"  Sequence shape (multi):  {x_multi.shape}")
print(f"  Sequence shape (single): {x_single.shape}")

print("\nStep 3: Time-based train/val/test split...")
train_mask = date_seq < TRAIN_END_DATE
val_mask = (date_seq >= TRAIN_END_DATE) & (date_seq < VAL_END_DATE)
test_mask = date_seq >= VAL_END_DATE

if min(int(np.sum(train_mask)), int(np.sum(val_mask)), int(np.sum(test_mask))) == 0:
    raise ValueError("One of the time-based splits is empty.")

y_train = y_seq[train_mask]
y_val = y_seq[val_mask]
y_test = y_seq[test_mask]
current_val = current_seq[val_mask]
current_test = current_seq[test_mask]
test_dates = date_seq[test_mask]

print(f"  Train: {int(np.sum(train_mask))}")
print(f"  Val:   {int(np.sum(val_mask))}")
print(f"  Test:  {int(np.sum(test_mask))}")

class_counts = np.array(
    [int(np.sum(y_train == label)) for label in sorted(regime_names)],
    dtype=np.int64,
)
if np.any(class_counts == 0):
    raise RuntimeError(
        "At least one regime is missing from the training split. "
        "Retrain the HMM or move the split boundary."
    )

class_weights = torch.tensor(
    [len(y_train) / (len(regime_names) * count) for count in class_counts],
    dtype=torch.float32,
).to(device)

for label, weight in enumerate(class_weights.cpu().numpy()):
    print(f"  Class weight - {regime_names[label]}: {weight:.2f}")


def train_variant(variant_name: str, x_all: np.ndarray) -> dict:
    config = VARIANT_CONFIGS[variant_name]
    variant_seed = 42 if variant_name == "multi_asset" else 43
    set_global_seed(variant_seed)

    x_train = x_all[train_mask]
    x_val = x_all[val_mask]
    x_test = x_all[test_mask]

    print(f"\nStep 4-{config['display_name']}: Fitting train-only scaler...")
    scaler = StandardScaler()
    scaler.fit(x_train.reshape(-1, x_train.shape[-1]))
    joblib.dump(scaler, config["scaler_path"])

    x_train_scaled = scale_sequence_batch(scaler, x_train)
    x_val_scaled = scale_sequence_batch(scaler, x_val)
    x_test_scaled = scale_sequence_batch(scaler, x_test)

    print(f"  Feature width: {x_train_scaled.shape[-1]}")
    print(f"  Saved: {config['scaler_path']}")

    print(f"\nStep 5-{config['display_name']}: Training classifier...")
    train_loader = build_train_loader(x_train_scaled, y_train, variant_seed)
    x_val_t = torch.tensor(x_val_scaled, dtype=torch.float32).to(device)
    x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)

    model = RegimeClassifier(input_size=x_train_scaled.shape[-1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CLASSIFIER_LR,
        weight_decay=CLASSIFIER_WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_macro_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(CLASSIFIER_MAX_EPOCHS):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
            val_preds = np.argmax(val_probs, axis=1)

        val_metrics = evaluate_predictions(
            y_true=y_val,
            y_pred=val_preds,
            current_labels=current_val,
            regime_names=regime_names,
            probabilities=val_probs,
        )
        scheduler.step(val_metrics["macro_f1"])

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), config["model_path"])
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            marker = " <-- best" if epoch + 1 == best_epoch else ""
            print(
                f"  Epoch {epoch + 1:>3}/{CLASSIFIER_MAX_EPOCHS} | "
                f"Loss: {avg_loss:.4f} | "
                f"Val Agreement: {val_metrics['accuracy']:.4f} | "
                f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
                f"Patience: {patience_counter}/{CLASSIFIER_PATIENCE}{marker}"
            )

        if patience_counter >= CLASSIFIER_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch + 1}")
            break

    print(f"\n  Best epoch: {best_epoch} | Best val macro-F1: {best_val_macro_f1:.4f}")

    print(f"\nStep 6-{config['display_name']}: Evaluating on held-out test set...")
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.eval()

    with torch.no_grad():
        test_logits = model(x_test_t)
        test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
        test_preds = np.argmax(test_probs, axis=1)

    test_metrics = evaluate_predictions(
        y_true=y_test,
        y_pred=test_preds,
        current_labels=current_test,
        regime_names=regime_names,
        probabilities=test_probs,
    )
    baseline_metrics = evaluate_predictions(
        y_true=y_test,
        y_pred=current_test,
        current_labels=current_test,
        regime_names=regime_names,
        probabilities=None,
    )

    print(f"  Test agreement vs HMM proxy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy'] * 100:.1f}%)")
    print(f"  Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Persistence baseline: {baseline_metrics['accuracy']:.4f} ({baseline_metrics['accuracy'] * 100:.1f}%)")
    print("\n  Classification Report (agreement vs HMM proxy):")
    print(test_metrics["report"])

    if test_metrics["transition_total"]:
        print(
            f"  Transition hit rate: {test_metrics['transition_hits']}/"
            f"{test_metrics['transition_total']} ({test_metrics['transition_hit_rate'] * 100:.1f}%)"
        )
    else:
        print("  No transitions found in the held-out test range.")

    results = f"""
{config['display_name'].upper()} TRAINING REPORT
============================================================
Target source:               Next-day HMM proxy regime from Stage 3
Best epoch:                  {best_epoch}
Best val macro-F1:           {best_val_macro_f1:.4f}
Held-out agreement:          {test_metrics['accuracy']:.4f} ({test_metrics['accuracy'] * 100:.1f}%)
Held-out macro-F1:           {test_metrics['macro_f1']:.4f}
Held-out transition hit:     {test_metrics['transition_hits']}/{test_metrics['transition_total']} ({test_metrics['transition_hit_rate'] * 100:.1f}%)
Held-out high-conf days:     {test_metrics['high_conf_days']}
Held-out high-conf agree.:   {test_metrics['high_conf_accuracy']:.4f}

Persistence baseline agree.: {baseline_metrics['accuracy']:.4f} ({baseline_metrics['accuracy'] * 100:.1f}%)
Persistence baseline macroF1:{baseline_metrics['macro_f1']:.4f}

Test window start:           {test_dates[0]}
Test window end:             {test_dates[-1]}

Classification report:
{test_metrics['report']}
"""

    with open(config["report_path"], "w", encoding="utf-8") as handle:
        handle.write(results.strip() + "\n")

    return {
        "variant_name": variant_name,
        "display_name": config["display_name"],
        "feature_width": int(x_train_scaled.shape[-1]),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro_f1),
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
    }


variant_results = {
    "multi_asset": train_variant("multi_asset", x_multi),
    "single_asset": train_variant("single_asset", x_single),
}

print("\nStep 7: Saving default inference artifacts (multi-asset)...")
torch.save(
    torch.load(VARIANT_CONFIGS["multi_asset"]["model_path"], map_location=device),
    "supervised_regime_predictor.pth",
)
joblib.dump(
    joblib.load(VARIANT_CONFIGS["multi_asset"]["scaler_path"]),
    "supervised_predictor_scaler.pkl",
)
print("  Saved: supervised_regime_predictor.pth")
print("  Saved: supervised_predictor_scaler.pkl")

multi_metrics = variant_results["multi_asset"]["test_metrics"]
single_metrics = variant_results["single_asset"]["test_metrics"]
baseline_metrics = variant_results["multi_asset"]["baseline_metrics"]

comparison_text = (
    "MULTI-ASSET VS SINGLE-ASSET - NEXT-DAY HMM PROXY FORECAST\n"
    + "=" * 60
    + "\n"
    + f"{'Variant':<28} {'Feat':<6} {'Agreement':<12} {'Macro-F1':<12} {'Transition Hit'}\n"
    + "-" * 84
    + "\n"
)

for variant_name in ["multi_asset", "single_asset"]:
    result = variant_results[variant_name]
    metrics = result["test_metrics"]
    comparison_text += (
        f"{result['display_name']:<28} "
        f"{result['feature_width']:<6} "
        f"{metrics['accuracy'] * 100:>7.2f}%    "
        f"{metrics['macro_f1']:<12.4f} "
        f"{metrics['transition_hits']}/{metrics['transition_total']} "
        f"({metrics['transition_hit_rate'] * 100:.1f}%)\n"
    )

comparison_text += (
    f"{'Persistence baseline':<28} "
    f"{'-':<6} "
    f"{baseline_metrics['accuracy'] * 100:>7.2f}%    "
    f"{baseline_metrics['macro_f1']:<12.4f} "
    f"{baseline_metrics['transition_hits']}/{baseline_metrics['transition_total']} "
    f"({baseline_metrics['transition_hit_rate'] * 100:.1f}%)\n"
)

comparison_text += "\n"
comparison_text += (
    f"Agreement gap (multi - single): {(multi_metrics['accuracy'] - single_metrics['accuracy']) * 100:+.2f} percentage points\n"
)
comparison_text += (
    f"Macro-F1 gap (multi - single): {(multi_metrics['macro_f1'] - single_metrics['macro_f1']):+.4f}\n"
)
comparison_text += (
    f"Agreement gap (multi - persistence): {(multi_metrics['accuracy'] - baseline_metrics['accuracy']) * 100:+.2f} percentage points\n"
)

with open("asset_mode_comparison.txt", "w", encoding="utf-8") as handle:
    handle.write(comparison_text)
with open("supervised_training_report.txt", "w", encoding="utf-8") as handle:
    handle.write(comparison_text + "\n\n" + variant_results["multi_asset"]["test_metrics"]["report"])

print("\n" + "=" * 60)
print("STAGE 4 COMPLETE")
print("  Saved: supervised_regime_predictor.pth")
print("  Saved: supervised_predictor_scaler.pkl")
print("  Saved: supervised_regime_names.npy")
print("  Saved: supervised_labels.npy")
print("  Saved: supervised_dates.npy")
print("  Saved: supervised_training_report.txt")
print("  Saved: asset_mode_comparison.txt")
print("\nNext: Run 05_predict.py")
print("=" * 60)
