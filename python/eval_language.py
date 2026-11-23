"""eval_language.py

Evaluate pre-softmax saturation/approximation on ModernBERT MLM (WikiText-103).
Metrics: CE, PPL, Top-1, KL divergence vs exact softmax.

Outputs a CSV with columns:
  model, ceiling, softmax_type, in_bw, out_bw, k_bits, s_bits, CE, PPL, Top1, KL
"""

import math
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from utils.attention_hooks import progress_bar, install_modernbert_saturation_hook, install_modernbert_lightmax_hook
from utils import dataset_config

# ---------------- Argument parser ----------------
parser = argparse.ArgumentParser(
    description="Evaluate attention+output softmax on MLM (ModernBERT)."
)
parser.add_argument("--dataset", type=str, default="wikitext-103",
                    help="Dataset name (wiki text)")
parser.add_argument("--models", type=str, nargs="+", default=[
    "answerdotai/ModernBERT-base",
    "answerdotai/ModernBERT-large",
])
parser.add_argument("--saturation", type=float, nargs="+", default=[8, 16, 24, 64],
                    help="Saturation T values to test (positive floats). Baseline always included.")
parser.add_argument("--use-lightmax", action="store_true", default=False,
                    help="Replace softmax with LightMax approximation")
parser.add_argument("--no-baseline", action="store_true", default=False,
                    help="Skip baseline evaluation")
parser.add_argument("--seq-len", type=int, nargs="+", default=[512],
                    help="Sequence lengths to evaluate (default: 512 4096)")
parser.add_argument("--batch-size", type=int, default=8,
                    help="Batch size (all seq_len except the longest)")
parser.add_argument("--batch-size-4096", type=int, default=1,
                    help="Batch size for the longest seq_len value (typically 4096)")
parser.add_argument("--num-samples", type=int, default=0,
                    help="0 = all chunks, >0 limits the number of chunks evaluated")
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

SPECIAL_IDS = {"cls": 50281, "sep": 50282, "mask": 50284}
MASK_RATE = 0.30


def prepare_dataset(tokenizer, seq_len: int, max_tokens: int = 2_000_000):
    """Load WikiText-103 raw test, concat, tokenize, chunk non-overlapping."""
    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="test",
        cache_dir=str(dataset_config.HF_DATASETS_CACHE),
    )
    full_text = "\n".join(ds["text"])
    tokens = tokenizer(
        full_text, return_tensors="pt", truncation=True,
        max_length=max_tokens,
    ).input_ids[0]
    n = len(tokens)
    n_chunks = n // seq_len
    tokens = tokens[:n_chunks * seq_len]
    input_ids = tokens.view(n_chunks, seq_len)
    return input_ids


def generate_masks(input_ids: torch.Tensor, seed: int):
    """Generate boolean masks: never on [CLS] (pos 0) or [SEP] (pos -1)."""
    seq_len = input_ids.size(1)
    n_mask = round(MASK_RATE * (seq_len - 2))
    rng = torch.Generator().manual_seed(seed)
    masks = torch.zeros_like(input_ids, dtype=torch.bool)
    for i in range(input_ids.size(0)):
        idx = torch.randperm(seq_len - 2, generator=rng)[:n_mask]
        masks[i, 1 + idx] = True
    masked_ids = input_ids.clone()
    masked_ids[masks] = SPECIAL_IDS["mask"]
    return masks, masked_ids


@torch.no_grad()
def evaluate(model, tokenizer, input_ids, seq_len, device, batch_size,
             saturation=None, lightmax_module=None, num_seeds=1, cache_dir=None):
    """Run per-batch exact + approximate forward pass.

    For each batch, store exact log_probs, then run approximate in the same
    batch to avoid storing all reference log_probs.
    """
    from transformers import AutoModelForMaskedLM
    from torch.nn.functional import kl_div, log_softmax
    import transformers.models.modernbert.modeling_modernbert as modernbert_module

    # Save original eager_attention_forward BEFORE any patching so model_ref
    # (the exact reference) always uses the clean softmax, while model uses
    # the saturated/approximate version.
    original_eager_forward = modernbert_module.eager_attention_forward

    model_ref = AutoModelForMaskedLM.from_pretrained(
        model.config.name_or_path, attn_implementation="eager",
        cache_dir=cache_dir,
    ).to(device).half()
    model_ref.eval()

    # Install hooks (patches the module-level function for `model` only)
    if lightmax_module is not None:
        install_modernbert_lightmax_hook(model, lightmax_module, thr=saturation)
    elif saturation is not None:
        install_modernbert_saturation_hook(model, saturation)

    # The patched version that `model` will use
    patched_eager_forward = modernbert_module.eager_attention_forward

    # Collect results per seed
    all_results = []
    for seed in range(num_seeds):
        masks, masked_ids = generate_masks(input_ids, seed)
        exact_results = {"CE": [], "Top1": []}
        approx_results = {"CE": [], "Top1": []}
        kl_total = 0.0

        for i in progress_bar(range(0, len(input_ids), batch_size)):
            batch_ids = masked_ids[i:i + batch_size].to(device)
            batch_masks = masks[i:i + batch_size].to(device)
            batch_targets = input_ids[i:i + batch_size].to(device)

            # Exact reference — temporarily restore original eager_attention_forward
            modernbert_module.eager_attention_forward = original_eager_forward
            logits_exact = model_ref(batch_ids).logits.float()

            # Approximate pass — re-apply the patched version
            modernbert_module.eager_attention_forward = patched_eager_forward
            logits_approx = model(batch_ids).logits.float()

            log_probs_exact = log_softmax(logits_exact, dim=-1)
            log_probs_approx = log_softmax(logits_approx, dim=-1)

            masked_indices = batch_masks.nonzero(as_tuple=True)

            # Exact metrics
            ref_vals = log_probs_exact[masked_indices[0], masked_indices[1],
                                       batch_targets[batch_masks]]
            exact_loss = -ref_vals.mean().item()
            exact_top1 = (logits_exact.argmax(dim=-1)[masked_indices] ==
                          batch_targets[batch_masks]).float().mean().item()
            exact_results["CE"].append(exact_loss)
            exact_results["Top1"].append(exact_top1)

            # Approximate metrics
            approx_vals = log_probs_approx[masked_indices[0], masked_indices[1],
                                            batch_targets[batch_masks]]
            approx_loss = -approx_vals.mean().item()
            approx_top1 = (logits_approx.argmax(dim=-1)[masked_indices] ==
                           batch_targets[batch_masks]).float().mean().item()
            approx_results["CE"].append(approx_loss)
            approx_results["Top1"].append(approx_top1)

            # KL divergence in the same batch loop
            kl_total += kl_div(
                log_probs_exact, log_probs_approx,
                log_target=True, reduction="sum"
            ).item()

        kl_total /= len(input_ids) * input_ids.size(1)

        all_results.append({
            "exact": {"CE": float(np.mean(exact_results["CE"])),
                      "PPL": float(np.exp(np.mean(exact_results["CE"]))),
                      "Top1": float(100 * np.mean(exact_results["Top1"])),
                      "KL": 0.0},
            "approx": {"CE": float(np.mean(approx_results["CE"])),
                       "PPL": float(np.exp(np.mean(approx_results["CE"]))),
                       "Top1": float(100 * np.mean(approx_results["Top1"])),
                       "KL": float(kl_total)},
        })

    # Restore original module-level function
    modernbert_module.eager_attention_forward = original_eager_forward

    del model_ref
    return all_results


def run_task(args):
    """Orchestrate LLM MLM evaluation and save CSV."""
    import os
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from transformers.utils.logging import disable_progress_bar
    disable_progress_bar()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForMaskedLM

    rows = []

    cache_kwargs = {"cache_dir": args.cache_dir} if args.cache_dir else {}

    for model_name in args.models:
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, **cache_kwargs)

        sat_values = args.saturation if args.saturation else [32.0]

        for seq_len in args.seq_len:
            print(f"\n  --- Sequence length: {seq_len} ---")
            batch_size = args.batch_size_4096 if seq_len == max(args.seq_len) else args.batch_size
            has_lightmax_flag = args.use_lightmax

            # Prepare dataset once per seq_len
            input_ids = prepare_dataset(tokenizer, seq_len)
            n_chunks = input_ids.size(0)
            print(f"  Chunks: {n_chunks}, seq_len: {seq_len}")
            if args.num_samples > 0 and args.num_samples < n_chunks:
                input_ids = input_ids[:args.num_samples]
                print(f"  Limited to {args.num_samples} chunks")

            # --- 1. Baseline: pure model, no hooks ---
            if not args.no_baseline:
                model = AutoModelForMaskedLM.from_pretrained(
                    model_name, attn_implementation="eager", **cache_kwargs,
                ).to(device).half()
                model.eval()

                all_results = evaluate(
                    model, tokenizer, input_ids, seq_len,
                    device=device, batch_size=batch_size,
                    saturation=None, lightmax_module=None,
                    cache_dir=args.cache_dir,
                )

                # Aggregate
                agg = {}
                for phase in ("exact", "approx"):
                    if phase not in all_results[0]:
                        continue
                    vals = {k: [] for k in all_results[0][phase]}
                    for r in all_results:
                        for k, v in r[phase].items():
                            vals[k].append(v)
                    agg[phase] = {}
                    for k, vlist in vals.items():
                        agg[phase][k] = float(np.mean(vlist))

                ce = agg.get("approx", {}).get("CE", 0.0)
                ppl = agg.get("approx", {}).get("PPL", 0.0)
                top1 = agg.get("approx", {}).get("Top1", 0.0)
                kl = agg.get("approx", {}).get("KL", 0.0)

                row = {
                    "model": model_name,
                    "ceiling": "",
                    "softmax_type": "original",
                    "in_bw": "",
                    "out_bw": "",
                    "k_bits": "",
                    "s_bits": "",
                    "CE": f"{ce:.6f}",
                    "PPL": f"{ppl:.4f}",
                    "Top1": f"{top1:.2f}",
                    "KL": f"{kl:.6f}",
                }
                rows.append(row)
                print(f"  {'original':>20} (T=None):  CE={ce:.6f}  PPL={ppl:.4f}  Top-1={top1:.2f}%  KL={kl:.6f}")

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # --- 2. Per saturation: LightMax / plain softmax ---
            for sat in sat_values:
                # --- 2a. LightMax ---
                if args.use_lightmax:
                    model = AutoModelForMaskedLM.from_pretrained(
                        model_name, attn_implementation="eager", **cache_kwargs,
                    ).to(device).half()
                    model.eval()

                    from lightmax import LightMax
                    lightmax_module = LightMax(
                        ceiling=sat,
                        in_bw=args.in_bw, out_bw=args.out_bw,
                        k_bits=args.k_bits, s_bits=args.s_bits,
                    ).to(device)

                    all_results = evaluate(
                        model, tokenizer, input_ids, seq_len,
                        device=device, batch_size=batch_size,
                        saturation=sat, lightmax_module=lightmax_module,
                        cache_dir=args.cache_dir,
                    )

                    # Aggregate
                    agg = {}
                    for phase in ("exact", "approx"):
                        if phase not in all_results[0]:
                            continue
                        vals = {k: [] for k in all_results[0][phase]}
                        for r in all_results:
                            for k, v in r[phase].items():
                                vals[k].append(v)
                    agg[phase] = {}
                    for k, vlist in vals.items():
                        agg[phase][k] = float(np.mean(vlist))

                    ce = agg.get("approx", {}).get("CE", 0.0)
                    ppl = agg.get("approx", {}).get("PPL", 0.0)
                    top1 = agg.get("approx", {}).get("Top1", 0.0)
                    kl = agg.get("approx", {}).get("KL", 0.0)

                    row = {
                        "model": model_name,
                        "ceiling": sat,
                        "softmax_type": "lightmax",
                        "in_bw": str(args.in_bw),
                        "out_bw": str(args.out_bw),
                        "k_bits": str(args.k_bits),
                        "s_bits": str(args.s_bits),
                        "CE": f"{ce:.6f}",
                        "PPL": f"{ppl:.4f}",
                        "Top1": f"{top1:.2f}",
                        "KL": f"{kl:.6f}",
                    }
                    rows.append(row)
                    print(f"  {'lightmax':>20} (T={sat}):  CE={ce:.6f}  PPL={ppl:.4f}  Top-1={top1:.2f}%  KL={kl:.6f}")

                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # --- 2b. No LightMax flag set -> plain softmax with saturation hooks ---
                if not has_lightmax_flag:
                    model = AutoModelForMaskedLM.from_pretrained(
                        model_name, attn_implementation="eager", **cache_kwargs,
                    ).to(device).half()
                    model.eval()

                    all_results = evaluate(
                        model, tokenizer, input_ids, seq_len,
                        device=device, batch_size=batch_size,
                        saturation=sat, lightmax_module=None,
                        cache_dir=args.cache_dir,
                    )

                    # Aggregate
                    agg = {}
                    for phase in ("exact", "approx"):
                        if phase not in all_results[0]:
                            continue
                        vals = {k: [] for k in all_results[0][phase]}
                        for r in all_results:
                            for k, v in r[phase].items():
                                vals[k].append(v)
                    agg[phase] = {}
                    for k, vlist in vals.items():
                        agg[phase][k] = float(np.mean(vlist))

                    ce = agg.get("approx", {}).get("CE", 0.0)
                    ppl = agg.get("approx", {}).get("PPL", 0.0)
                    top1 = agg.get("approx", {}).get("Top1", 0.0)
                    kl = agg.get("approx", {}).get("KL", 0.0)

                    row = {
                        "model": model_name,
                        "ceiling": sat,
                        "softmax_type": "original",
                        "in_bw": "",
                        "out_bw": "",
                        "k_bits": "",
                        "s_bits": "",
                        "CE": f"{ce:.6f}",
                        "PPL": f"{ppl:.4f}",
                        "Top1": f"{top1:.2f}",
                        "KL": f"{kl:.6f}",
                    }
                    rows.append(row)
                    print(f"  {'original':>20} (T={sat}):  CE={ce:.6f}  PPL={ppl:.4f}  Top-1={top1:.2f}%  KL={kl:.6f}")

                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # Save
    df = pd.DataFrame(rows, columns=[
        "model", "ceiling", "softmax_type", "in_bw", "out_bw", "k_bits", "s_bits",
        "CE", "PPL", "Top1", "KL",
    ])
    csv_suffix = args.csv_name if args.csv_name else "_results"
    out_path = output_dir / f"eval_language{csv_suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


# ----------------- Main function -----------------
def main():
    args = parser.parse_args()
    run_task(args)


if __name__ == "__main__":
    main()
