"""measure_lightmax_mse.py

Measure MSE of LightMax approximation vs torch.softmax on real attention
logits from ViT Base on ImageNet-1k validation set.

Memory-efficient: accumulates squared errors on-the-fly per batch, never
stores logit tensors across batches.

Also provides capture_and_export_logits() to export real quantized logits
for the VHDL hardware simulation pipeline.
"""

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import timm
from timm.data import create_transform, resolve_data_config
from PIL import Image
from utils import dataset_config


parser = argparse.ArgumentParser(
    description="Measure LightMax MSE on real attention logits (ImageNet val)."
)
parser.add_argument("--model", type=str, default="vit_base_patch16_224",
                    help="ViT model name from timm")
parser.add_argument("--data-dir", type=str, default=str(dataset_config.IMAGENET_DIR),
                    help="Path to ImageNet validation directory")
parser.add_argument("--ceiling", type=float, default=24.0,
                    help="Ceiling for LightMax saturation")
parser.add_argument("--in-bw", type=int, default=8)
parser.add_argument("--out-bw", type=int, default=8)
parser.add_argument("--k-bits", type=int, default=4)
parser.add_argument("--s-bits", type=int, default=6)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-samples", type=int, default=10000,
                    help="0 = all samples")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--layer-idx", type=int, default=0,
                    help="Attention layer index to capture logits from")
parser.add_argument("--query-idx", type=int, default=0,
                    help="Query row index within the attention matrix (0 = CLS)")
parser.add_argument("--head-idx", type=int, default=0,
                    help="Attention head index to capture from")


class ImageNetDataset:
    """ImageNet validation folder with configurable transform."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.samples: List[tuple] = []
        self.transform: Optional[Callable] = None
        cls_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        cls_to_idx = {d.name: i for i, d in enumerate(cls_dirs)}
        for cls_dir in cls_dirs:
            for img_file in sorted(cls_dir.iterdir()):
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((img_file, cls_to_idx[cls_dir.name]))

    def __len__(self) -> int:
        return len(self.samples)

    def set_transform(self, transform: Callable) -> None:
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


@torch.no_grad()
def run_and_measure(
    args: argparse.Namespace,
    lightmax_module: nn.Module,
    variant_name: str,
) -> Dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_layers = 12

    se_accum = [0.0] * num_layers
    count_accum = [0] * num_layers

    model = timm.create_model(args.model, pretrained=True).to(device)
    model.eval()

    layer_idx = [0]

    for module in model.modules():
        if isinstance(module, timm.models.vision_transformer.Attention):
            module.fused_attn = False
            li = layer_idx[0]
            layer_idx[0] += 1

            def _make_hooked(attn_mod, layer_index):
                def _hooked(x, attn_mask=None):
                    B, N, C = x.shape
                    qkv = attn_mod.qkv(x).reshape(
                        B, N, 3, attn_mod.num_heads, attn_mod.head_dim
                    ).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)
                    q = q * attn_mod.scale
                    attn = q @ k.transpose(-2, -1)

                    ref = torch.softmax(attn, dim=-1).float()
                    approx = lightmax_module(attn).float()
                    se_accum[layer_index] += (approx - ref).pow(2).sum().item()
                    count_accum[layer_index] += ref.numel()

                    if attn_mask is not None:
                        attn = attn + attn_mask
                    attn = attn.softmax(dim=-1)
                    attn = attn_mod.attn_drop(attn)
                    x = attn @ v
                    x = x.transpose(1, 2).reshape(B, N, attn_mod.attn_dim)
                    x = attn_mod.norm(x)
                    x = attn_mod.proj(x)
                    x = attn_mod.proj_drop(x)
                    return x
                return _hooked

            module.forward = _make_hooked(module, li)

    transform = create_transform(
        **resolve_data_config({}, model=args.model),
        is_training=False,
    )
    dataset = ImageNetDataset(args.data_dir)
    dataset.set_transform(transform)
    n = len(dataset) if args.num_samples == 0 else min(args.num_samples, len(dataset))
    if n < len(dataset):
        dataset.samples = dataset.samples[:n]

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False,
    )

    num_batches = 0
    for images, _ in loader:
        images = images.to(device)
        model(images)
        num_batches += 1
        if num_batches % 50 == 0:
            print(f"  [{variant_name}] Processed {num_batches * args.batch_size} images...", flush=True)

    per_layer_mse = [se / max(c, 1) for se, c in zip(se_accum, count_accum)]
    avg_mse = sum(per_layer_mse) / len(per_layer_mse)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "per_layer_mse": per_layer_mse,
        "avg_mse": avg_mse,
        "num_forward_passes": num_batches,
        "num_samples": n,
        "num_layers": num_layers,
    }


def run_measurement(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {args.device}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Samples: {args.num_samples if args.num_samples > 0 else 'all (50K)'}", flush=True)
    print(f"Config: ceiling={args.ceiling}, in_bw={args.in_bw}, out_bw={args.out_bw}, "
          f"k_bits={args.k_bits}, s_bits={args.s_bits}", flush=True)
    print(flush=True)

    from lightmax import LightMax
    lightmax_module = LightMax(
        ceiling=args.ceiling,
        in_bw=args.in_bw, out_bw=args.out_bw,
        k_bits=args.k_bits, s_bits=args.s_bits,
    ).to(args.device)
    lightmax_module.eval()

    res = run_and_measure(args, lightmax_module, "LightMax")
    for li, mse in enumerate(res["per_layer_mse"]):
        print(f"  Layer {li:2d}:  MSE = {mse:.8e}", flush=True)
    print(f"  Average: MSE = {res['avg_mse']:.8e}", flush=True)
    print(flush=True)

    print("=" * 60, flush=True)
    print(f"Summary (ceiling={args.ceiling}, in_bw={args.in_bw}, "
          f"out_bw={args.out_bw}, k_bits={args.k_bits}, s_bits={args.s_bits})", flush=True)
    print(f"Samples: {res['num_samples']}, Batches: {res['num_forward_passes']}", flush=True)
    print(f"Per-layer MSE: {[f'{m:.4e}' for m in res['per_layer_mse']]}", flush=True)
    print(f"Average MSE: {res['avg_mse']:.8e}", flush=True)

    return res


@torch.no_grad()
def capture_and_export_logits(
    args: argparse.Namespace,
    output_path: str,
    layer_idx: int = 0,
    query_idx: int = 0,
    head_idx: int = 0,
    num_queries: int = 1,
) -> int:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = timm.create_model(args.model, pretrained=True).to(device)
    model.eval()

    captured = [None]
    original_forwards = {}

    for module in model.modules():
        if isinstance(module, timm.models.vision_transformer.Attention):
            module.fused_attn = False

    target_layer = 0
    for module in model.modules():
        if isinstance(module, timm.models.vision_transformer.Attention):
            if target_layer == layer_idx:
                original_forwards[module] = module.forward

                def _make_capture_hook(attn_mod, q_idx, h_idx, n_q, orig_fn):
                    def _hooked(x, attn_mask=None):
                        if captured[0] is not None:
                            return orig_fn(x, attn_mask)

                        B, N, C = x.shape
                        qkv = attn_mod.qkv(x).reshape(
                            B, N, 3, attn_mod.num_heads, attn_mod.head_dim
                        ).permute(2, 0, 3, 1, 4)
                        q, k, v = qkv.unbind(0)
                        q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)
                        q = q * attn_mod.scale
                        attn = q @ k.transpose(-2, -1)

                        h_idx_safe = min(h_idx, attn.shape[1] - 1)
                        max_q = min(q_idx + n_q, attn.shape[2])
                        rows = attn[0, h_idx_safe, q_idx:max_q, :].float().cpu()

                        ceiling = getattr(args, 'ceiling', 24.0)
                        in_bw = getattr(args, 'in_bw', 8)
                        input_scale = ceiling / (2 ** (in_bw - 1) - 1)

                        rows = rows.clamp(-ceiling, ceiling)
                        quantized = torch.round(rows / input_scale).clamp(
                            -(2 ** (in_bw - 1)), (2 ** (in_bw - 1)) - 1
                        ).to(torch.int32)

                        captured[0] = quantized

                        if attn_mask is not None:
                            attn = attn + attn_mask
                        attn = attn.softmax(dim=-1)
                        attn = attn_mod.attn_drop(attn)
                        x = attn @ v
                        x = x.transpose(1, 2).reshape(B, N, attn_mod.attn_dim)
                        x = attn_mod.norm(x)
                        x = attn_mod.proj(x)
                        x = attn_mod.proj_drop(x)
                        return x
                    return _hooked

                module.forward = _make_capture_hook(module, query_idx, head_idx, num_queries, original_forwards[module])
                break
            target_layer += 1

    if target_layer < layer_idx:
        raise ValueError(f"Layer index {layer_idx} out of range (model has {target_layer} layers)")

    transform = create_transform(
        **resolve_data_config({}, model=args.model),
        is_training=False,
    )
    dataset = ImageNetDataset(args.data_dir)
    dataset.set_transform(transform)
    dataset.samples = dataset.samples[:1]

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=0, drop_last=False,
    )

    for images, _ in loader:
        images = images.to(device)
        model(images)
        break

    if captured[0] is None:
        raise RuntimeError("Hook never triggered -- no attention layer matched")

    rows = captured[0]

    with open(output_path, "w") as f:
        if num_queries == 1:
            f.write("index  value\n")
            for i, val in enumerate(rows[0].tolist()):
                f.write(f"{i}  {val}\n")
        else:
            header = "index  " + "  ".join(f"unit{i}" for i in range(num_queries))
            f.write(header + "\n")
            n_keys = rows.shape[1]
            for j in range(n_keys):
                vals = "  ".join(str(rows[i, j].item()) for i in range(num_queries))
                f.write(f"{j}  {vals}\n")

    del model, captured[0]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows.shape[1]


def main() -> None:
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    run_measurement(args)


if __name__ == "__main__":
    main()
