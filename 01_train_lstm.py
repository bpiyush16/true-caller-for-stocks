import time
import warnings

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from pipeline_utils import (
    AUTOENCODER_BATCH_SIZE,
    AUTOENCODER_EPOCHS,
    AUTOENCODER_HIDDEN_SIZE,
    AUTOENCODER_LR,
    AUTOENCODER_NUM_LAYERS,
    AUTOENCODER_PATIENCE,
    COLUMN_ORDER,
    HISTORY_END_EXCLUSIVE,
    LSTMAutoencoder,
    START_DATE,
    TRAIN_END_DATE,
    VAL_END_DATE,
    WINDOW,
    compute_log_returns,
    create_sequences,
    download_market_prices,
    get_device,
    set_global_seed,
)

warnings.filterwarnings("ignore")

set_global_seed()
device = get_device()


print("=" * 60)
print("STAGE 1 - TRAIN LSTM AUTOENCODER")
print("=" * 60)
if torch.cuda.is_available():
    print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    print("GPU Not Available - Using CPU")
print(f"Device: {device}")
print(f"Historical range: {START_DATE} to 2024-12-31")
print(f"Training cut-off: < {TRAIN_END_DATE}")
print(f"Validation range: {TRAIN_END_DATE} to {VAL_END_DATE}")
print()

print("Step 1: Downloading and aligning prices...")
t0 = time.time()
prices = download_market_prices(start=START_DATE, end=HISTORY_END_EXCLUSIVE, threads=False)
print(f"  Aligned prices shape: {prices.shape}")
print(f"  Download time: {time.time() - t0:.2f} sec")
print()

print("Step 2: Building log returns and train-only scaler...")
t1 = time.time()
returns = compute_log_returns(prices)
raw_returns = returns.to_numpy(dtype=np.float32)
vix_prices = prices.loc[returns.index, "VIX"].to_numpy(dtype=np.float32)
returns_dates = returns.index.astype(str).to_numpy()
sequence_end_dates = returns.index[WINDOW - 1 :].astype(str).to_numpy()

train_return_mask = returns_dates < TRAIN_END_DATE
if int(np.sum(train_return_mask)) <= WINDOW:
    raise RuntimeError("Not enough training return rows to fit the scaler and autoencoder.")

scaler = MinMaxScaler()
scaler.fit(raw_returns[train_return_mask])
scaled = scaler.transform(raw_returns).astype(np.float32)

np.save("raw_returns.npy", raw_returns)
np.save("vix_prices.npy", vix_prices)
np.save("scaled_data.npy", scaled)
np.save("returns_dates.npy", returns_dates)
np.save("sequence_end_dates.npy", sequence_end_dates)
np.save("column_order.npy", np.asarray(COLUMN_ORDER))
joblib.dump(scaler, "scaler.pkl")

print(f"  Returns shape: {returns.shape}")
print(f"  Train return rows: {int(np.sum(train_return_mask))}")
print(f"  Scaled shape: {scaled.shape}")
print(f"  Prep time: {time.time() - t1:.2f} sec")
print("  Saved: raw_returns.npy")
print("  Saved: vix_prices.npy")
print("  Saved: scaled_data.npy")
print("  Saved: returns_dates.npy")
print("  Saved: sequence_end_dates.npy")
print("  Saved: column_order.npy")
print("  Saved: scaler.pkl")
print()

print("Step 3: Creating sliding windows...")
t2 = time.time()
x_all = create_sequences(scaled, window=WINDOW)
train_window_mask = sequence_end_dates < TRAIN_END_DATE
val_window_mask = (sequence_end_dates >= TRAIN_END_DATE) & (sequence_end_dates < VAL_END_DATE)

if not np.any(train_window_mask):
    raise RuntimeError("No training windows were created.")
if not np.any(val_window_mask):
    raise RuntimeError("No validation windows were created.")

x_train = x_all[train_window_mask]
x_val = x_all[val_window_mask]

np.save("sequence_train_mask.npy", train_window_mask)
np.save("sequence_val_mask.npy", val_window_mask)

print(f"  Full sequence shape: {x_all.shape}")
print(f"  Train windows: {len(x_train)}")
print(f"  Val windows:   {len(x_val)}")
print(f"  Window prep time: {time.time() - t2:.2f} sec")
print("  Saved: sequence_train_mask.npy")
print("  Saved: sequence_val_mask.npy")
print()

print("Step 4: Setting up PyTorch loaders...")
pin_memory = device.type == "cuda"
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(x_train)),
    batch_size=AUTOENCODER_BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=pin_memory,
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(x_val)),
    batch_size=AUTOENCODER_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory,
)
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches:   {len(val_loader)}")
print()

print(f"Step 5: Training LSTM autoencoder ({AUTOENCODER_EPOCHS} epochs max)...")
model = LSTMAutoencoder(
    input_size=len(COLUMN_ORDER),
    hidden_size=AUTOENCODER_HIDDEN_SIZE,
    num_layers=AUTOENCODER_NUM_LAYERS,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=AUTOENCODER_LR)
criterion = nn.MSELoss()

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0
history = []
train_start = time.time()

for epoch in range(AUTOENCODER_EPOCHS):
    model.train()
    train_loss = 0.0

    for (batch,) in train_loader:
        batch = batch.to(device, non_blocking=pin_memory)
        optimizer.zero_grad()
        reconstructed, _ = model(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (batch,) in val_loader:
            batch = batch.to(device, non_blocking=pin_memory)
            reconstructed, _ = model(batch)
            val_loss += criterion(reconstructed, batch).item()
    val_loss /= len(val_loader)

    history.append((epoch + 1, train_loss, val_loss))
    print(
        f"Epoch {epoch + 1:02d}/{AUTOENCODER_EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "lstm_model.pth")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= AUTOENCODER_PATIENCE:
            print(f"Early stopping triggered after epoch {epoch + 1}.")
            break

elapsed = time.time() - train_start

with open("stage1_training_report.txt", "w", encoding="utf-8") as report:
    report.write("STAGE 1 - AUTOENCODER TRAINING REPORT\n")
    report.write("=" * 60 + "\n")
    report.write(f"Train windows: {len(x_train)}\n")
    report.write(f"Val windows:   {len(x_val)}\n")
    report.write(f"Best epoch:    {best_epoch}\n")
    report.write(f"Best val loss: {best_val_loss:.6f}\n")
    report.write(f"Training time: {elapsed:.2f} sec\n\n")
    report.write("Epoch\tTrainLoss\tValLoss\n")
    for epoch, train_loss, val_loss in history:
        report.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\n")

print()
print("=" * 60)
print("STAGE 1 COMPLETE")
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Training time: {elapsed:.2f} sec")
print("Saved: lstm_model.pth")
print("Saved: scaler.pkl")
print("Saved: scaled_data.npy")
print("Saved: raw_returns.npy")
print("Saved: vix_prices.npy")
print("Saved: returns_dates.npy")
print("Saved: sequence_end_dates.npy")
print("Saved: sequence_train_mask.npy")
print("Saved: sequence_val_mask.npy")
print("Saved: stage1_training_report.txt")
print("Next: Run 02_extract_latent.py")
print("=" * 60)
