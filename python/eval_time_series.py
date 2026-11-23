"""eval_time_series.py

Evaluate pre-softmax saturation/approximation on forecasting datasets (ETT, Weather).
Model: PatchTSTForPrediction (from HuggingFace). Metrics: MAE, MSE on forecast horizon.
Uses fine-tuned model.pt from finetune_patchtst.py.

Outputs a CSV with columns:
  model, ceiling, softmax_type, in_bw, out_bw, k_bits, s_bits, MAE, MSE
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from utils.attention_hooks import progress_bar, install_patchtst_saturation_hook, install_patchtst_lightmax_hook
from utils.patchtst_utils import build_patchtst_config
from utils import dataset_config

# ---------------- Argument parser ----------------
parser = argparse.ArgumentParser(
    description="Evaluate attention saturation on forecasting datasets (PatchTST)."
)
parser.add_argument("--dataset", type=str, default="/media/lubin/Datasets/Forecasting",
                    help="Directory containing forecasting CSV files")
parser.add_argument("--models", type=str, nargs="+", default=[
    "ibm-research/patchtst-etth1-pretrain",
], help="HuggingFace model IDs (used for config reference)")
parser.add_argument("--saturation", type=float, nargs="+", default=[8, 16, 24, 64],
                    help="Saturation T values to test. Baseline always included.")
parser.add_argument("--use-lightmax", action="store_true", default=False,
                    help="Replace softmax with LightMax approximation")
parser.add_argument("--no-baseline", action="store_true", default=False,
                    help="Skip baseline evaluation")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--num-samples", type=int, default=0, help="0 = all windows")
parser.add_argument("--output-dir", type=str, default="panda",
                    help="Directory for CSV output")
parser.add_argument("--csv-name", type=str, default="",
                    help="CSV filename suffix (default: '' -> _results)")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cache-dir", type=str, default=str(dataset_config.HF_HUB_CACHE),
                    help="HF model cache dir (default: HF_HUB_CACHE from dataset_config)")
parser.add_argument("--in-bw", type=int, default=8)
parser.add_argument("--out-bw", type=int, default=8)
parser.add_argument("--k-bits", type=int, default=4)
parser.add_argument("--s-bits", type=int, default=6)
parser.add_argument("--seq-len", type=int, default=512, help="Input sequence length")
parser.add_argument("--patch-len", type=int, default=12, help="Patch length (pretrained arch)")
parser.add_argument("--eval-stride", type=int, default=1,
                    help="Stride between test windows (1 = sliding, standard protocol)")
parser.add_argument("--forecast-horizon", type=int, default=96, help="Forecast horizon")
parser.add_argument("--finetune-dir", type=str, default="models",
                    help="Directory with fine-tuned model.pt (from finetune_patchtst.py)")

DATASETS = [("ETTh1", "ETTh1"), ("ETTh2", "ETTh2"),
            ("ETTm1", "ETTm1"), ("ETTm2", "ETTm2"), ("weather", "Weather")]


def load_forecasting_data(file_path: str):
    """Load a CSV, split 80/20 train/test by row."""
    df = pd.read_csv(file_path, parse_dates=["date"])
    n = len(df)
    split = int(n * 0.8)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()
    return train_df, test_df


def standardize(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Standardize with train-split statistics per channel."""
    val_cols = [c for c in train_df.columns if c != "date"]
    mean = train_df[val_cols].mean().values.astype(np.float32)
    std = train_df[val_cols].std().values.astype(np.float32)
    std[std < 1e-8] = 1.0
    train_norm = (train_df[val_cols].values.astype(np.float32) - mean) / std
    test_norm = (test_df[val_cols].values.astype(np.float32) - mean) / std
    return train_norm, test_norm, mean, std


def create_forecast_data(data: np.ndarray, seq_len: int, forecast_horizon: int,
                         stride: int | None = None):
    """Create (input, target) pairs for forecasting with the given window stride."""
    if stride is None:
        stride = seq_len
    inputs, targets = [], []
    total = seq_len + forecast_horizon
    for start in range(0, len(data) - total + 1, stride):
        inputs.append(data[start:start + seq_len].T)  # [C, seq_len]
        targets.append(data[start + seq_len:start + total].T)  # [C, horizon]
    return np.stack(inputs), np.stack(targets)


def load_datasets(data_dir: str, seq_len: int, forecast_horizon: int, rng_seed: int,
                  eval_stride: int = 1):
    """Load all forecasting datasets, returning dict of {stem: {label, inputs, targets}}."""
    data_dir = Path(data_dir)
    dataset_info = {}
    for short, label in DATASETS:
        csv_path = data_dir / f"{short}.csv"
        if not csv_path.exists():
            print(f"  SKIPPED {label}: {csv_path} not found")
            continue
        train_df, test_df = load_forecasting_data(str(csv_path))
        _, test_norm, _, _ = standardize(train_df, test_df)
        inputs, targets = create_forecast_data(test_norm, seq_len, forecast_horizon,
                                               stride=eval_stride)
        if len(inputs) == 0:
            print(f"  SKIPPED {label}: no windows of length {seq_len + forecast_horizon}")
            continue
        dataset_info[short] = {"label": label, "inputs": inputs, "targets": targets}
        print(f"  {label}: {len(inputs)} test windows, C={test_norm.shape[1]} channels")
    return dataset_info


@torch.no_grad()
def evaluate(model, inputs, targets, batch_size, device):
    """Compute MAE and MSE over forecast horizon."""
    mae_total = 0.0
    mse_total = 0.0
    n = 0
    for i in progress_bar(range(0, len(inputs), batch_size)):
        batch_x = torch.from_numpy(inputs[i:i + batch_size]).to(device).half()
        batch_y = torch.from_numpy(targets[i:i + batch_size]).to(device)
        batch_x = batch_x.transpose(1, 2)  # [B, C, seq_len] -> [B, seq_len, C] for PatchTST
        outputs = model(past_values=batch_x).prediction_outputs
        outputs = outputs.transpose(1, 2)  # [B, horizon, C] -> [B, C, horizon] to match targets
        mae_total += (outputs - batch_y).abs().sum().item()
        mse_total += (outputs - batch_y).pow(2).sum().item()
        n += batch_y.numel()
    return {"MAE": mae_total / n, "MSE": mse_total / n}


def run_task(args):
    """Orchestrate forecasting evaluation and save CSV."""
    import os
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from transformers.utils.logging import disable_progress_bar
    disable_progress_bar()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.dataset)
    if not data_dir.exists():
        print(f"Dataset directory {data_dir} not found.")
        return

    dataset_info = load_datasets(str(data_dir), args.seq_len, args.forecast_horizon, args.seed,
                                 eval_stride=args.eval_stride)

    ft_root = Path(args.finetune_dir)
    if not ft_root.exists():
        print(f"Fine-tuned directory {ft_root} not found, skipping.")
        return

    from transformers import PatchTSTForPrediction

    rows = []

    for model_name in args.models:
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")

        for ds_stem, info in dataset_info.items():
            ds_label = info["label"]
            model_path = ft_root / ds_stem / "model.pt"
            if not model_path.exists():
                print(f"  SKIPPED {ds_label}: no model.pt at {model_path}")
                continue

            C = info["inputs"].shape[1]
            model_config = build_patchtst_config(
                C, args.seq_len, args.forecast_horizon,
                patch_length=args.patch_len, patch_stride=args.patch_len,
            )

            sat_values = args.saturation if args.saturation else [32.0]
            has_lightmax_flag = args.use_lightmax

            # --- 1. Baseline: pure model, no hooks ---
            if not args.no_baseline:
                model = PatchTSTForPrediction(config=model_config)
                model.load_state_dict(
                    torch.load(model_path, map_location="cpu", weights_only=True)
                )
                model.to(device).half()
                model.eval()

                inputs = info["inputs"]
                targets = info["targets"]
                n = len(inputs)
                if args.num_samples > 0:
                    inputs = inputs[:args.num_samples]
                    targets = targets[:args.num_samples]

                r = evaluate(model, inputs, targets, args.batch_size, device)
                mae, mse = r["MAE"], r["MSE"]
                rows.append({
                    "model": f"{model_name}/{ds_label}",
                    "ceiling": "",
                    "softmax_type": "original",
                    "in_bw": "",
                    "out_bw": "",
                    "k_bits": "",
                    "s_bits": "",
                    "MAE": f"{mae:.6f}",
                    "MSE": f"{mse:.6f}",
                })
                print(f"  {ds_label:>8} {'original':>20} (T=None):  MAE={mae:.6f}  MSE={mse:.6f}")

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # --- 2. Per saturation: LightMax / plain softmax ---
            for sat in sat_values:
                # --- 2a. LightMax ---
                if args.use_lightmax:
                    model = PatchTSTForPrediction(config=model_config)
                    model.load_state_dict(
                        torch.load(model_path, map_location="cpu", weights_only=True)
                    )
                    model.to(device).half()
                    model.eval()

                    from lightmax import LightMax
                    lightmax_module = LightMax(
                        ceiling=sat,
                        in_bw=args.in_bw, out_bw=args.out_bw,
                        k_bits=args.k_bits, s_bits=args.s_bits,
                    ).to(device)
                    install_patchtst_lightmax_hook(model, lightmax_module, thr=sat)

                    inputs = info["inputs"]
                    targets = info["targets"]
                    n = len(inputs)
                    if args.num_samples > 0:
                        inputs = inputs[:args.num_samples]
                        targets = targets[:args.num_samples]

                    r = evaluate(model, inputs, targets, args.batch_size, device)
                    mae, mse = r["MAE"], r["MSE"]
                    rows.append({
                        "model": f"{model_name}/{ds_label}",
                        "ceiling": sat,
                        "softmax_type": "lightmax",
                        "in_bw": str(args.in_bw),
                        "out_bw": str(args.out_bw),
                        "k_bits": str(args.k_bits),
                        "s_bits": str(args.s_bits),
                        "MAE": f"{mae:.6f}",
                        "MSE": f"{mse:.6f}",
                    })
                    print(f"  {ds_label:>8} {'lightmax':>20} (T={sat}):  MAE={mae:.6f}  MSE={mse:.6f}")

                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # --- 2b. No LightMax flag set -> plain softmax with saturation hooks ---
                if not has_lightmax_flag:
                    model = PatchTSTForPrediction(config=model_config)
                    model.load_state_dict(
                        torch.load(model_path, map_location="cpu", weights_only=True)
                    )
                    model.to(device).half()
                    model.eval()

                    install_patchtst_saturation_hook(model, sat)

                    inputs = info["inputs"]
                    targets = info["targets"]
                    n = len(inputs)
                    if args.num_samples > 0:
                        inputs = inputs[:args.num_samples]
                        targets = targets[:args.num_samples]

                    r = evaluate(model, inputs, targets, args.batch_size, device)
                    mae, mse = r["MAE"], r["MSE"]
                    rows.append({
                        "model": f"{model_name}/{ds_label}",
                        "ceiling": sat,
                        "softmax_type": "original",
                        "in_bw": "",
                        "out_bw": "",
                        "k_bits": "",
                        "s_bits": "",
                        "MAE": f"{mae:.6f}",
                        "MSE": f"{mse:.6f}",
                    })
                    print(f"  {ds_label:>8} {'original':>20} (T={sat}):  MAE={mae:.6f}  MSE={mse:.6f}")

                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    if rows:
        df = pd.DataFrame(rows, columns=[
            "model", "ceiling", "softmax_type", "in_bw", "out_bw", "k_bits", "s_bits",
            "MAE", "MSE",
        ])
        csv_suffix = args.csv_name if args.csv_name else "_results"
        out_path = output_dir / f"eval_time_series{csv_suffix}.csv"
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")


# ----------------- Main function -----------------
def main():
    args = parser.parse_args()
    run_task(args)


if __name__ == "__main__":
    main()
