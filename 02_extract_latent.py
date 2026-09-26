import warnings

import numpy as np

from pipeline_utils import (
    WINDOW,
    extract_latent_vectors,
    get_device,
    load_autoencoder_model,
)

warnings.filterwarnings("ignore")

device = get_device()

print("=" * 50)
print("STAGE 2 - EXTRACT LATENT VECTORS")
print("=" * 50)

print("\nStep 1: Loading trained autoencoder...")
model = load_autoencoder_model("lstm_model.pth", device=device)
print(f"  Model loaded on {device}")

print("\nStep 2: Loading scaled historical data...")
scaled = np.load("scaled_data.npy")
sequence_end_dates = np.load("sequence_end_dates.npy", allow_pickle=True)
print(f"  Scaled data shape: {scaled.shape}")

print("\nStep 3: Extracting latent vectors for all windows...")
latent_vectors = extract_latent_vectors(
    scaled_data=scaled,
    model=model,
    window=WINDOW,
    batch_size=256,
    device=device,
)

print(f"  Latent vectors shape: {latent_vectors.shape}")
if len(latent_vectors) != len(sequence_end_dates):
    raise ValueError(
        f"Mismatch: latent_vectors={len(latent_vectors)} vs sequence_end_dates={len(sequence_end_dates)}"
    )

np.save("latent_vectors.npy", latent_vectors)

print("\n" + "=" * 50)
print("STAGE 2 COMPLETE")
print(f"  Latent vectors shape: {latent_vectors.shape}")
print("  Saved: latent_vectors.npy")
print("\nNext: Run 03_train_hmm.py")
print("=" * 50)
