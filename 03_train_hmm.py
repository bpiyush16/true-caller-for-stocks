import warnings

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pipeline_utils import (
    HMM_COMPONENTS,
    HMM_PCA_COMPONENTS,
    REGIME_NAMES,
    TRAIN_END_DATE,
    WINDOW,
    build_raw_feat,
    canonicalize_hmm_outputs,
    compute_state_statistics,
    fit_best_hmm,
    format_confusion_matrix,
)

warnings.filterwarnings("ignore")

print("=" * 50)
print("STAGE 3 - TRAIN HMM")
print("=" * 50)

print("\nStep 1: Loading latent history and aligned returns...")
latent_vectors = np.load("latent_vectors.npy")
raw_returns = np.load("raw_returns.npy")
vix_prices = np.load("vix_prices.npy")
sequence_end_dates = np.load("sequence_end_dates.npy", allow_pickle=True)

print(f"  Latent vectors shape: {latent_vectors.shape}")
print(f"  Raw returns shape:    {raw_returns.shape}")
print(f"  VIX prices shape:     {vix_prices.shape}")
print(f"  Sequence dates shape: {sequence_end_dates.shape}")

if latent_vectors.ndim != 2:
    raise ValueError(f"latent_vectors must be 2D, got shape {latent_vectors.shape}")
if raw_returns.ndim != 2 or raw_returns.shape[1] != 5:
    raise ValueError(f"raw_returns must have shape (N, 5), got {raw_returns.shape}")
if vix_prices.ndim != 1:
    raise ValueError(f"vix_prices must be 1D, got shape {vix_prices.shape}")

expected_sequences = len(raw_returns) - WINDOW + 1
if len(latent_vectors) != expected_sequences:
    raise ValueError(
        f"latent_vectors length mismatch: got {len(latent_vectors)}, expected {expected_sequences}"
    )
if len(sequence_end_dates) != expected_sequences:
    raise ValueError(
        f"sequence_end_dates length mismatch: got {len(sequence_end_dates)}, expected {expected_sequences}"
    )

train_mask = sequence_end_dates < TRAIN_END_DATE
if int(np.sum(train_mask)) <= HMM_COMPONENTS:
    raise RuntimeError("Not enough pre-2019 windows to fit a 3-state HMM.")

print(f"  HMM training windows: {int(np.sum(train_mask))}")

print("\nStep 2: Building HMM features...")
raw_feat = build_raw_feat(raw_returns, vix_prices, window=WINDOW)
if len(raw_feat) != len(latent_vectors):
    raise ValueError("raw_feat and latent_vectors must have the same length.")

pca = PCA(n_components=HMM_PCA_COMPONENTS, random_state=42)
latent_pca_train = pca.fit_transform(latent_vectors[train_mask])
latent_pca = pca.transform(latent_vectors).astype(np.float32)

hmm_features = np.hstack([latent_pca, raw_feat]).astype(np.float32)

feat_scaler = StandardScaler()
feat_scaler.fit(hmm_features[train_mask])
hmm_features_scaled = feat_scaler.transform(hmm_features).astype(np.float32)

print(f"  Raw feature shape:        {raw_feat.shape}")
print(f"  PCA latent shape:         {latent_pca.shape}")
print(f"  Combined HMM shape:       {hmm_features.shape}")
print(f"  PCA train variance ratio: {np.round(pca.explained_variance_ratio_, 4)}")

print("\nStep 3: Fitting 3-state Gaussian HMM on the training period...")
hmm_model = fit_best_hmm(hmm_features_scaled[train_mask], n_components=HMM_COMPONENTS)

raw_labels = hmm_model.predict(hmm_features_scaled)
raw_posteriors = hmm_model.predict_proba(hmm_features_scaled)
hmm_model, regime_labels, regime_posteriors, train_state_stats = canonicalize_hmm_outputs(
    model=hmm_model,
    labels=raw_labels,
    posteriors=raw_posteriors,
    raw_feat=raw_feat,
    fit_mask=train_mask,
)

full_state_stats = compute_state_statistics(regime_labels, raw_feat)

print("  Final transition matrix:")
for row in hmm_model.transmat_:
    print(f"    {np.round(row, 3)}")

print("\nStep 4: State summary after canonical relabeling...")
print(f"  {'State':<6} {'Name':<14} {'Rows':<8} {'VIX Level':<12} {'SP Return':<12} {'SP Vol'}")
print("  " + "-" * 68)
for state in sorted(REGIME_NAMES):
    stats = full_state_stats[state]
    print(
        f"  {state:<6} {REGIME_NAMES[state]:<14} {stats['count']:<8} "
        f"{stats['vix_level']:<12.2f} {stats['sp_return']:<12.6f} {stats['sp_vol']:.6f}"
    )

print("\nStep 5: Saving HMM artifacts...")
np.save("regime_labels.npy", regime_labels)
np.save("regime_names.npy", np.array(list(REGIME_NAMES.items()), dtype=object))
np.save("raw_feat.npy", raw_feat)
np.save("hmm_dates.npy", sequence_end_dates)
np.save("latent_pca.npy", latent_pca)
np.save("hmm_posteriors.npy", regime_posteriors)
np.save("hmm_train_mask.npy", train_mask)

joblib.dump(hmm_model, "hmm_model.pkl")
joblib.dump(feat_scaler, "feat_scaler.pkl")
joblib.dump(pca, "pca_model.pkl")
joblib.dump(hmm_features_scaled, "hmm_features.pkl")

with open("stage3_hmm_report.txt", "w", encoding="utf-8") as report:
    report.write("STAGE 3 - HMM TRAINING REPORT\n")
    report.write("=" * 60 + "\n")
    report.write(f"HMM training windows: {int(np.sum(train_mask))}\n")
    report.write("PCA explained variance ratio: ")
    report.write(", ".join(f"{value:.4f}" for value in pca.explained_variance_ratio_))
    report.write("\n\nTransition matrix:\n")
    report.write(str(np.round(hmm_model.transmat_, 4)))
    report.write("\n\nTraining-period state summary:\n")
    for state in sorted(REGIME_NAMES):
        stats = train_state_stats[state]
        report.write(
            f"  {REGIME_NAMES[state]} | rows={stats['count']} | "
            f"vix={stats['vix_level']:.2f} | "
            f"ret={stats['sp_return']:.6f} | vol={stats['sp_vol']:.6f}\n"
        )
    report.write("\nFull-history state summary:\n")
    for state in sorted(REGIME_NAMES):
        stats = full_state_stats[state]
        report.write(
            f"  {REGIME_NAMES[state]} | rows={stats['count']} | "
            f"vix={stats['vix_level']:.2f} | "
            f"ret={stats['sp_return']:.6f} | vol={stats['sp_vol']:.6f}\n"
        )
    report.write("\nState confusion vs persistence baseline is measured later in supervised stages.\n")
    report.write("\nFull-history state-by-state counts:\n")
    counts = np.bincount(regime_labels, minlength=len(REGIME_NAMES))
    report.write(str(dict(zip(REGIME_NAMES.values(), counts.tolist()))))
    report.write("\n")

print("\n" + "=" * 50)
print("STAGE 3 COMPLETE")
print("  Saved: hmm_model.pkl")
print("  Saved: regime_labels.npy")
print("  Saved: regime_names.npy")
print("  Saved: pca_model.pkl")
print("  Saved: feat_scaler.pkl")
print("  Saved: raw_feat.npy")
print("  Saved: latent_pca.npy")
print("  Saved: hmm_posteriors.npy")
print("  Saved: hmm_features.pkl")
print("  Saved: hmm_dates.npy")
print("  Saved: hmm_train_mask.npy")
print("  Saved: stage3_hmm_report.txt")
print("\nNext: Run 04_train_regim_pre.py")
print("=" * 50)
