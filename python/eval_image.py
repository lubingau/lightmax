"""eval_image.py

Evaluate pre-softmax saturation/approximation on ImageNet classification.
Models: vit_small, vit_base, swin_small, swin_base, HF swin-large.
Metrics: top-1, top-5 accuracy.

Outputs a CSV with columns:
  model, ceiling, softmax_type, in_bw, out_bw, k_bits, s_bits, top1, top5
"""

import argparse
from pathlib import Path
from typing import List, Optional

import torch
import timm
import pandas as pd
from timm.data import create_transform, resolve_data_config
from PIL import Image
from utils.attention_hooks import (
    progress_bar, install_swin_saturation_hook, install_timm_saturation_hook,
    install_timm_lightmax_hook, install_swin_lightmax_hook,
)
from utils import dataset_config


# ---------------- Argument parser ----------------
parser = argparse.ArgumentParser(
    description="Evaluate attention saturation on ImageNet classification."
)
parser.add_argument("--dataset", type=str, default="/media/lubin/Datasets/Imagenet/val",
                    help="Path to ImageNet validation directory")
parser.add_argument("--models", type=str, nargs="+", default=[
    "vit_small_patch16_224",
    "vit_base_patch16_224",
    "vit_large_patch16_224",
    "swin_small_patch4_window7_224",
    "swin_base_patch4_window7_224",
    "swin_large_patch4_window7_224",
])
parser.add_argument("--saturation", type=float, nargs="+", default=[8, 16, 24, 64],
                    help="Saturation T values to test. Baseline always included.")
parser.add_argument("--use-lightmax", action="store_true", default=False,
                    help="Replace softmax with LightMax approximation")
parser.add_argument("--no-baseline", action="store_true", default=False,
                    help="Skip baseline evaluation")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-samples", type=int, default=0, help="0 = all")
parser.add_argument("--output-dir", type=str, default="panda",
                    help="Directory for CSV output")
parser.add_argument("--csv-name", type=str, default="",
                    help="CSV filename suffix (default: '' -> _results)")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cache-dir", type=str, default=str(dataset_config.HF_HUB_CACHE),
                    help="HF model cache dir (default: HF_HUB_CACHE from dataset_config)")
parser.add_argument("--in-bw", type=int, default=10)
parser.add_argument("--out-bw", type=int, default=10)
parser.add_argument("--k-bits", type=int, default=4)
parser.add_argument("--s-bits", type=int, default=6)


class _HFImageTransform:
    """Pickable transform wrapping a HuggingFace image processor for single images."""

    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        self.model_name = model_name
        from transformers import AutoImageProcessor
        self.processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)

    def __call__(self, img):
        return self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)


class ImageNetDataset:
    """ImageNet validation folder with per-model transform support."""

    def __init__(self, root: str):
        root = Path(root)
        self.samples: List[tuple] = []
        self.transform = None
        cls_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        cls_to_idx = {d.name: i for i, d in enumerate(cls_dirs)}
        for cls_dir in cls_dirs:
            for img_file in sorted(cls_dir.iterdir()):
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((img_file, cls_to_idx[cls_dir.name]))

    def __len__(self) -> int:
        return len(self.samples)

    def set_transform(self, transform) -> None:
        self.transform = transform

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _is_hf_model(name: str) -> bool:
    """Check if model name looks like a HuggingFace hub path."""
    return "/" in name


@torch.no_grad()
def evaluate(model, loader, device, is_hf=False):
    """Compute top-1 and top-5 accuracy."""
    top1_correct = 0
    top5_correct = 0
    total = 0
    for images, labels in progress_bar(loader):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if is_hf:
            logits = logits.logits
        _, pred = logits.topk(5, dim=-1)
        top1_correct += (pred[:, 0] == labels).sum().item()
        top5_correct += pred.eq(labels.unsqueeze(-1)).any(dim=-1).sum().item()
        total += labels.size(0)
    return {"top1": 100.0 * top1_correct / total, "top5": 100.0 * top5_correct / total}


def run_task(args):
    """Orchestrate classification evaluation and save CSV."""
    import os
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from transformers.utils.logging import disable_progress_bar
    disable_progress_bar()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ImageNetDataset(args.dataset)
    n = len(dataset) if args.num_samples == 0 else min(args.num_samples, len(dataset))
    dataset.samples = dataset.samples[:n]
    print(f"Samples: {n}")

    rows = []

    cache_kwargs = {"cache_dir": args.cache_dir} if args.cache_dir else {}

    for model_name in args.models:
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")

        is_hf = _is_hf_model(model_name)

        if is_hf:
            from transformers import AutoImageProcessor
            dataset.set_transform(_HFImageTransform(model_name, cache_dir=args.cache_dir))
        else:
            transform = create_transform(
                **resolve_data_config({}, model=model_name),
                is_training=False,
            )
            dataset.set_transform(transform)

        sat_values = args.saturation if args.saturation else [32.0]
        has_lightmax_flag = args.use_lightmax

        # --- 1. Baseline: pure model, no hooks ---
        if not args.no_baseline:
            if is_hf:
                from transformers import AutoModelForImageClassification
                model = AutoModelForImageClassification.from_pretrained(model_name, **cache_kwargs).to(device)
            else:
                model = timm.create_model(model_name, pretrained=True).to(device)
            model.eval()

            loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=4, pin_memory=True, drop_last=False,
            )
            r = evaluate(model, loader, device, is_hf=is_hf)
            top1 = r["top1"]
            top5 = r["top5"]
            rows.append({
                "model": model_name,
                "ceiling": "",
                "softmax_type": "original",
                "in_bw": "",
                "out_bw": "",
                "k_bits": "",
                "s_bits": "",
                "top1": f"{top1:.2f}",
                "top5": f"{top5:.2f}",
            })
            print(f"  {'original':>20} (T=None):  top-1={top1:.2f}%  top-5={top5:.2f}%")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- 2. Per saturation: LightMax / plain softmax ---
        for sat in sat_values:
            # --- 2a. LightMax ---
            if args.use_lightmax:
                if is_hf:
                    from transformers import AutoModelForImageClassification
                    model = AutoModelForImageClassification.from_pretrained(model_name, **cache_kwargs).to(device)
                else:
                    model = timm.create_model(model_name, pretrained=True).to(device)
                model.eval()

                from lightmax import LightMax
                lightmax_module = LightMax(
                    ceiling=sat,
                    in_bw=args.in_bw, out_bw=args.out_bw,
                    k_bits=args.k_bits, s_bits=args.s_bits,
                ).to(device)
                if is_hf:
                    install_swin_lightmax_hook(model, lightmax_module, thr=sat)
                else:
                    install_timm_lightmax_hook(model, lightmax_module, thr=sat)

                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True, drop_last=False,
                )
                r = evaluate(model, loader, device, is_hf=is_hf)
                top1 = r["top1"]
                top5 = r["top5"]
                rows.append({
                    "model": model_name,
                    "ceiling": sat,
                    "softmax_type": "lightmax",
                    "in_bw": str(args.in_bw),
                    "out_bw": str(args.out_bw),
                    "k_bits": str(args.k_bits),
                    "s_bits": str(args.s_bits),
                    "top1": f"{top1:.2f}",
                    "top5": f"{top5:.2f}",
                })
                print(f"  {'lightmax':>20} (T={sat}):  top-1={top1:.2f}%  top-5={top5:.2f}%")

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # --- 2b. No LightMax flag set -> plain softmax with saturation hooks ---
            if not has_lightmax_flag:
                if is_hf:
                    from transformers import AutoModelForImageClassification
                    model = AutoModelForImageClassification.from_pretrained(model_name, **cache_kwargs).to(device)
                else:
                    model = timm.create_model(model_name, pretrained=True).to(device)
                model.eval()

                if is_hf:
                    install_swin_saturation_hook(model, sat)
                else:
                    install_timm_saturation_hook(model, sat)

                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True, drop_last=False,
                )
                r = evaluate(model, loader, device, is_hf=is_hf)
                top1 = r["top1"]
                top5 = r["top5"]
                rows.append({
                    "model": model_name,
                    "ceiling": sat,
                    "softmax_type": "original",
                    "in_bw": "",
                    "out_bw": "",
                    "k_bits": "",
                    "s_bits": "",
                    "top1": f"{top1:.2f}",
                    "top5": f"{top5:.2f}",
                })
                print(f"  {'original':>20} (T={sat}):  top-1={top1:.2f}%  top-5={top5:.2f}%")

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    df = pd.DataFrame(rows, columns=[
        "model", "ceiling", "softmax_type", "in_bw", "out_bw", "k_bits", "s_bits",
        "top1", "top5",
    ])
    csv_suffix = args.csv_name if args.csv_name else "_results"
    out_path = output_dir / f"eval_image{csv_suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


# ----------------- Main function -----------------
def main():
    args = parser.parse_args()
    data_dir = Path(args.dataset)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    run_task(args)


if __name__ == "__main__":
    main()
