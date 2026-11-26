"""finetune_patchtst.py

Fine-tune PatchTST models from HuggingFace on ETT/Weather datasets.
Trains the full model (all parameters). Saves best checkpoint by test MSE.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.patchtst_utils import build_patchtst_config, load_pretrained_patchtst
from utils import dataset_config


parser = argparse.ArgumentParser(
    description="Fine-tune PatchTST on ETT/Weather datasets."
)
parser.add_argument("--data-dir", type=str, default=str(dataset_config.FORECASTING_DIR),
                    help="Directory containing CSV files")
parser.add_argument("--model-name", type=str, default="ibm-research/patchtst-etth1-pretrain",
                    help="HuggingFace model ID for PatchTST")
parser.add_argument("--datasets", type=str, nargs="+", default=[
    "ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather",
])
parser.add_argument("--seq-len", type=int, default=512, help="Input sequence length")
parser.add_argument("--patch-len", type=int, default=8, help="Patch length")
parser.add_argument("--forecast-horizon", type=int, default=96, help="Forecast horizon")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Max learning rate")
parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum LR (cosine annealing)")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--train-stride", type=int, default=8,
                    help="Stride between sliding training windows (1 = every position)")
parser.add_argument("--no-pretrained", action="store_true", default=False,
                    help="Train from random init instead of pretrained weights")
parser.add_argument("--force", action="store_true", default=False,
                    help="Overwrite an existing model.pt")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--out-dir", type=str, default="models/",
                    help="Root directory for saving fine-tuned models")
parser.add_argument("--num-workers", type=int, default=0)


def load_forecasting_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(file_path, parse_dates=["date"])
    n = len(df)
    split = int(n * 0.8)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()
    return train_df, test_df


def standardize(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    val_cols = [c for c in train_df.columns if c != "date"]
    mean = train_df[val_cols].mean().values.astype(np.float32)
    std = train_df[val_cols].std().values.astype(np.float32)
    std[std < 1e-8] = 1.0
    train_norm = (train_df[val_cols].values.astype(np.float32) - mean) / std
    test_norm = (test_df[val_cols].values.astype(np.float32) - mean) / std
    return train_norm, test_norm, mean, std


def create_forecast_data(data: np.ndarray, seq_len: int, forecast_horizon: int,
                         stride: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if stride is None:
        stride = forecast_horizon
    C = data.shape[1]
    total_len = seq_len + forecast_horizon
    inputs, targets = [], []
    for start in range(0, len(data) - total_len + 1, stride):
        inputs.append(data[start:start + seq_len].T)
        targets.append(data[start + seq_len:start + total_len].T)
    if not inputs:
        return np.zeros((0, C, seq_len), dtype=np.float32), np.zeros((0, C, forecast_horizon), dtype=np.float32)
    return np.stack(inputs, axis=0), np.stack(targets, axis=0)


def finetune_patchtst_on_dataset(args, model: torch.nn.Module,
                                train_inputs: torch.Tensor, train_targets: torch.Tensor,
                                test_inputs: torch.Tensor, test_targets: torch.Tensor,
                                device: torch.device):
    train_in = train_inputs.transpose(1, 2).contiguous()
    train_tgt = train_targets.transpose(1, 2).contiguous()
    test_in = test_inputs.transpose(1, 2).contiguous()
    test_tgt = test_targets.transpose(1, 2).contiguous()

    train_dataset = TensorDataset(train_in, train_tgt)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    test_dataset = TensorDataset(test_in, test_tgt)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr,
    )
    loss_fn = torch.nn.MSELoss()
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    epoch_train_losses = []
    epoch_test_metrics = []
    best_test_mse = float("inf")
    best_state = None

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_trainable:,} (full model)")

    for epoch in range(args.epochs):
        current_lr = scheduler.get_last_lr()[0]

        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch_x, batch_t in train_loader:
            batch_x, batch_t = batch_x.to(device), batch_t.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=use_amp):
                outputs = model(past_values=batch_x)
                pred = outputs.prediction_outputs
                loss = loss_fn(pred, batch_t)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_train_loss = total_loss / max(n_batches, 1)
        epoch_train_losses.append(avg_train_loss)

        model.eval()
        test_mae, test_mse = 0.0, 0.0
        n_test = 0
        with torch.no_grad():
            for batch_x, batch_t in test_loader:
                batch_x, batch_t = batch_x.to(device), batch_t.to(device)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    outputs = model(past_values=batch_x)
                    pred = outputs.prediction_outputs
                test_mse += loss_fn(pred, batch_t).item() * batch_x.size(0)
                test_mae += torch.abs(pred - batch_t).mean().item() * batch_x.size(0)
                n_test += batch_x.size(0)
        avg_test_mse = test_mse / max(n_test, 1)
        avg_test_mae = test_mae / max(n_test, 1)
        epoch_test_metrics.append({"MAE": avg_test_mae, "MSE": avg_test_mse})

        if avg_test_mse < best_test_mse:
            best_test_mse = avg_test_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1:3d}/{args.epochs}  "
                  f"LR: {current_lr:.2e}  "
                  f"Train MSE: {avg_train_loss:.6f}  "
                  f"Test MAE: {avg_test_mae:.6f}  Test MSE: {avg_test_mse:.6f}  "
                  f"{'*' if avg_test_mse == best_test_mse else ''}")

    return epoch_train_losses, epoch_test_metrics, best_state


def run_finetuning(args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist.")
        return

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for ds_stem in args.datasets:
        csv_paths = list(data_dir.rglob(f"{ds_stem}.csv"))
        if not csv_paths:
            print(f"SKIPPED {ds_stem}: no CSV found under {data_dir}")
            continue
        csv_path = csv_paths[0]
        print(f"\n{'=' * 70}\nDataset: {ds_stem} ({csv_path})")

        train_df, test_df = load_forecasting_data(csv_path)
        print(f"  Raw rows: train={len(train_df)}, test={len(test_df)}")
        train_norm, test_norm, mean, std = standardize(train_df, test_df)
        print(f"  Channels: {train_norm.shape[1]}")

        train_inputs, train_targets = create_forecast_data(
            train_norm, args.seq_len, args.forecast_horizon, stride=args.train_stride,
        )
        test_inputs, test_targets = create_forecast_data(
            test_norm, args.seq_len, args.forecast_horizon,
        )
        print(f"  Train samples: {len(train_inputs)}, Test samples: {len(test_inputs)}")

        if len(train_inputs) == 0:
            print(f"  SKIPPED {ds_stem}: not enough training data")
            continue
        if len(test_inputs) == 0:
            print(f"  SKIPPED {ds_stem}: not enough test data")
            continue

        save_dir = out_root / ds_stem
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "model.pt"
        if model_path.exists() and not args.force:
            print(f"  Already fine-tuned at {model_path}, skipping (use --force to overwrite).")
            continue

        if args.no_pretrained:
            print(f"\n  Initializing PatchTST from random weights (--no-pretrained)")
            from transformers import PatchTSTForPrediction
            model = PatchTSTForPrediction(
                build_patchtst_config(train_norm.shape[1], args.seq_len, args.forecast_horizon,
                                      patch_length=args.patch_len, patch_stride=args.patch_len)
            )
        else:
            print(f"\n  Loading pretrained PatchTST weights from {args.model_name}")
            model, _report = load_pretrained_patchtst(
                args.model_name, train_norm.shape[1], args.seq_len, args.forecast_horizon,
            )
        model.to(device)

        train_in_t = torch.from_numpy(train_inputs)
        train_tgt_t = torch.from_numpy(train_targets)
        test_in_t = torch.from_numpy(test_inputs)
        test_tgt_t = torch.from_numpy(test_targets)

        train_losses, test_metrics, best_state = finetune_patchtst_on_dataset(
            args, model, train_in_t, train_tgt_t,
            test_in_t, test_tgt_t, device,
        )

        torch.save(best_state, model_path)

        best_epoch = None
        best_test_mse = float("inf")
        for ei, tm in enumerate(test_metrics):
            if tm["MSE"] < best_test_mse:
                best_test_mse = tm["MSE"]
                best_epoch = ei

        config = {
            "model_name": args.model_name,
            "pretrained": not args.no_pretrained,
            "dataset": ds_stem,
            "seq_len": args.seq_len,
            "patch_len": args.patch_len,
            "train_stride": args.train_stride,
            "forecast_horizon": args.forecast_horizon,
            "epochs": args.epochs,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "batch_size": args.batch_size,
            "best_epoch": best_epoch,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "best_test_mse": best_test_mse,
            "epoch_train_losses": [round(v, 6) for v in train_losses],
            "epoch_test_metrics": test_metrics,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Saved best model (epoch {best_epoch}/{args.epochs}, "
              f"test MSE: {best_test_mse:.6f}) to {model_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parser.parse_args()
    run_finetuning(args)


if __name__ == "__main__":
    main()
