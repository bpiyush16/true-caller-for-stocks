import warnings
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import torch

from pipeline_utils import (
    HIGH_CONF_THRESHOLD,
    REGIME_NAMES,
    SEQ_LEN,
    VARIANT_CONFIGS,
    WINDOW,
    WARMUP_CALENDAR_DAYS,
    build_supervised_features,
    download_market_prices,
    get_device,
    infer_proxy_states_from_prices,
    load_autoencoder_model,
    load_classifier_model,
    load_regime_name_mapping,
)

warnings.filterwarnings("ignore")

device = get_device()

print("=" * 60)
print("STAGE 9 - TOMORROW REGIME FORECAST")
print("=" * 60)

print("\nStep 1: Loading trained artifacts...")
return_scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca_model.pkl")
hmm_feat_scaler = joblib.load("feat_scaler.pkl")
hmm_model = joblib.load("hmm_model.pkl")
multi_scaler = joblib.load("supervised_predictor_scaler.pkl")
single_scaler = joblib.load(VARIANT_CONFIGS["single_asset"]["scaler_path"])
regime_names = load_regime_name_mapping("supervised_regime_names.npy")

autoencoder = load_autoencoder_model("lstm_model.pth", device=device)
multi_model = load_classifier_model("supervised_regime_predictor.pth", input_size=15, device=device)
single_model = load_classifier_model(
    VARIANT_CONFIGS["single_asset"]["model_path"],
    input_size=5,
    device=device,
)
print("  Artifacts loaded.")

print("\nStep 2: Downloading recent prices...")
end_exclusive = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=WARMUP_CALENDAR_DAYS)).strftime("%Y-%m-%d")
prices = download_market_prices(start=start_date, end=end_exclusive, threads=False)
print(f"  Clean prices shape: {prices.shape}")

print("\nStep 3: Inferring the current HMM proxy regime...")
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

if len(feature_sets["multi_asset"]) < SEQ_LEN:
    raise ValueError(f"Need at least {SEQ_LEN} feature rows, got {len(feature_sets['multi_asset'])}.")

current_date = pd.Timestamp(proxy_bundle["window_end_dates"][-1])
current_label = int(proxy_bundle["regime_labels"][-1])
current_name = regime_names[current_label]

multi_seq = multi_scaler.transform(feature_sets["multi_asset"])[-SEQ_LEN:]
single_seq = single_scaler.transform(feature_sets["single_asset"])[-SEQ_LEN:]

print(f"  Current proxy date:   {current_date.date()}")
print(f"  Current proxy regime: {current_name}")

print("\nStep 4: Forecasting the next proxy regime...")
with torch.no_grad():
    multi_logits = multi_model(torch.tensor(multi_seq, dtype=torch.float32).unsqueeze(0).to(device))
    single_logits = single_model(torch.tensor(single_seq, dtype=torch.float32).unsqueeze(0).to(device))
    multi_probs = torch.softmax(multi_logits, dim=1).cpu().numpy()[0]
    single_probs = torch.softmax(single_logits, dim=1).cpu().numpy()[0]

multi_label = int(np.argmax(multi_probs))
single_label = int(np.argmax(single_probs))
tomorrow = (current_date + pd.offsets.BDay(1)).date()

multi_conf = float(np.max(multi_probs))
if multi_conf >= HIGH_CONF_THRESHOLD:
    conf_label = "HIGH confidence"
elif multi_conf >= 0.65:
    conf_label = "MODERATE confidence"
else:
    conf_label = "LOW confidence"

print("\n" + "=" * 60)
print("STAGE 9 - TOMORROW REGIME FORECAST")
print("=" * 60)
print(f"\n  Today    ({current_date.date()}): {current_name} [HMM proxy]")
print(f"  Baseline ({tomorrow}): {current_name} [persistence]")
print(f"  Multi    ({tomorrow}): {regime_names[multi_label]}")
print(f"  Single   ({tomorrow}): {regime_names[single_label]}")

print("\n  Multi-Asset Probabilities:")
for label in sorted(regime_names):
    print(f"    {regime_names[label]:<12} {multi_probs[label] * 100:6.1f}%")

print("\n  Single-Asset Probabilities:")
for label in sorted(regime_names):
    print(f"    {regime_names[label]:<12} {single_probs[label] * 100:6.1f}%")

print(f"\n  Multi-Asset confidence: {multi_conf * 100:.1f}% ({conf_label})")
if multi_label != current_label:
    print("  Multi-Asset regime-change signal detected.")
else:
    print("  Multi-Asset continuation expected.")

with open("predictor_results.txt", "a", encoding="utf-8") as handle:
    handle.write("\n\nSTAGE 9 - TOMORROW FORECAST\n")
    handle.write("=" * 60 + "\n")
    handle.write(f"Current proxy date: {current_date.date()}\n")
    handle.write(f"Current proxy regime: {current_name}\n")
    handle.write(f"Tomorrow (baseline): {current_name}\n")
    handle.write(f"Tomorrow (multi): {regime_names[multi_label]}\n")
    handle.write(f"Tomorrow (single): {regime_names[single_label]}\n")
    handle.write(
        "Multi probabilities: "
        + ", ".join(f"{regime_names[idx]}={multi_probs[idx]:.4f}" for idx in sorted(regime_names))
        + "\n"
    )

print("\n" + "=" * 60)
print("STAGE 9 COMPLETE")
print("NOTE: The target regime is the frozen HMM proxy state from Stage 3.")
print("NOTE: This is still a probabilistic forecast, not a guarantee.")
print("This is the final stage in the pipeline.")
print("=" * 60)
