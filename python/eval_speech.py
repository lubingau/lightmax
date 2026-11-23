"""eval_speech.py

Evaluate pre-softmax saturation/approximation on LibriSpeech (wav2vec2).
Models: wav2vec2-base-960h, wav2vec2-large-960h.  Metric: WER.

Outputs a CSV with columns:
  model, ceiling, softmax_type, in_bw, out_bw, k_bits, s_bits, WER
"""

import argparse
from pathlib import Path

import torch
import pandas as pd
from datasets import load_dataset
from utils.attention_hooks import progress_bar, install_wav2vec2_saturation_hook, install_wav2vec2_lightmax_hook
from utils import dataset_config

# ---------------- Argument parser ----------------
parser = argparse.ArgumentParser(
    description="Evaluate attention saturation on LibriSpeech (wav2vec2)."
)
parser.add_argument("--dataset", type=str, default="openslr/librispeech_asr",
                    help="Dataset name")
parser.add_argument("--config", type=str, default="clean",
                    help="Dataset config (use 'clean' for test.clean)")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--models", type=str, nargs="+", default=[
    "facebook/wav2vec2-base-960h",
    "facebook/wav2vec2-large-960h",
])
parser.add_argument("--saturation", type=float, nargs="+", default=[8, 16, 24, 64],
                    help="Saturation T values to test. Baseline always included.")
parser.add_argument("--use-lightmax", action="store_true", default=False,
                    help="Replace softmax with LightMax approximation")
parser.add_argument("--no-baseline", action="store_true", default=False,
                    help="Skip baseline evaluation")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--num-samples", type=int, default=0, help="0 = all")
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


@torch.no_grad()
def compute_wer(model, processor, loader, device):
    """Compute WER for all samples in loader."""
    all_preds, all_refs = [], []
    for batch in progress_bar(loader):
        input_values = batch["input_values"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        logits = model(input_values, attention_mask=attention_mask).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_texts = processor.batch_decode(pred_ids)
        ref_texts = batch["text"]
        all_preds.extend(pred_texts)
        all_refs.extend(ref_texts)

    total_wer = 0.0
    for p, r in zip(all_preds, all_refs):
        p_words = p.strip().split()
        r_words = r.strip().split()
        if len(r_words) == 0:
            total_wer += 0 if len(p_words) == 0 else 1.0
        else:
            d = [[0] * (len(r_words) + 1) for _ in range(len(p_words) + 1)]
            for i in range(len(p_words) + 1):
                d[i][0] = i
            for j in range(len(r_words) + 1):
                d[0][j] = j
            for i in range(1, len(p_words) + 1):
                for j in range(1, len(r_words) + 1):
                    cost = 0 if p_words[i - 1] == r_words[j - 1] else 1
                    d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            total_wer += d[len(p_words)][len(r_words)] / len(r_words)
    return {"WER": 100.0 * total_wer / len(all_refs)}


def collate_fn(batch, processor):
    """Collate audio batches for wav2vec2."""
    input_values = processor(
        [b["audio"]["array"] for b in batch],
        sampling_rate=16000, return_tensors="pt", padding=True,
    ).input_values
    texts = [b["text"] for b in batch]
    return {"input_values": input_values, "text": texts}


def run_task(args):
    """Orchestrate audio evaluation and save CSV."""
    import os
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from transformers.utils.logging import disable_progress_bar
    disable_progress_bar()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    print("Loading dataset...")
    dataset = load_dataset(
        args.dataset, args.config,
        split=args.split,
        cache_dir=str(dataset_config.HF_DATASETS_CACHE),
    )
    n = len(dataset) if args.num_samples == 0 else min(args.num_samples, len(dataset))
    dataset = dataset.select(range(n))
    print(f"Samples: {n}")

    rows = []

    cache_kwargs = {"cache_dir": args.cache_dir} if args.cache_dir else {}

    for model_name in args.models:
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")

        processor = Wav2Vec2Processor.from_pretrained(model_name, **cache_kwargs)

        sat_values = args.saturation if args.saturation else [32.0]
        has_lightmax_flag = args.use_lightmax

        # --- 1. Baseline: pure model, no hooks ---
        if not args.no_baseline:
            model = Wav2Vec2ForCTC.from_pretrained(model_name, **cache_kwargs).to(device)
            model.eval()

            loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                collate_fn=lambda b: collate_fn(b, processor),
                num_workers=0,
            )
            r = compute_wer(model, processor, loader, device)
            wer = r["WER"]
            rows.append({
                "model": model_name,
                "ceiling": "",
                "softmax_type": "original",
                "in_bw": "",
                "out_bw": "",
                "k_bits": "",
                "s_bits": "",
                "WER": f"{wer:.2f}",
            })
            print(f"  {'original':>20} (T=None):  WER={wer:.2f}%")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- 2. Per saturation: LightMax / plain softmax ---
        for sat in sat_values:
            # --- 2a. LightMax ---
            if args.use_lightmax:
                model = Wav2Vec2ForCTC.from_pretrained(model_name, **cache_kwargs).to(device)
                model.eval()

                from lightmax import LightMax
                lightmax_module = LightMax(
                    ceiling=sat,
                    in_bw=args.in_bw, out_bw=args.out_bw,
                    k_bits=args.k_bits, s_bits=args.s_bits,
                ).to(device)
                install_wav2vec2_lightmax_hook(model, lightmax_module, thr=sat)

                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=False,
                    collate_fn=lambda b: collate_fn(b, processor),
                    num_workers=0,
                )
                r = compute_wer(model, processor, loader, device)
                wer = r["WER"]
                rows.append({
                    "model": model_name,
                    "ceiling": sat,
                    "softmax_type": "lightmax",
                    "in_bw": str(args.in_bw),
                    "out_bw": str(args.out_bw),
                    "k_bits": str(args.k_bits),
                    "s_bits": str(args.s_bits),
                    "WER": f"{wer:.2f}",
                })
                print(f"  {'lightmax':>20} (T={sat}):  WER={wer:.2f}%")

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # --- 2b. No LightMax flag set -> plain softmax with saturation hooks ---
            if not has_lightmax_flag:
                model = Wav2Vec2ForCTC.from_pretrained(model_name, **cache_kwargs).to(device)
                model.eval()

                install_wav2vec2_saturation_hook(model, sat)

                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=False,
                    collate_fn=lambda b: collate_fn(b, processor),
                    num_workers=0,
                )
                r = compute_wer(model, processor, loader, device)
                wer = r["WER"]
                rows.append({
                    "model": model_name,
                    "ceiling": sat,
                    "softmax_type": "original",
                    "in_bw": "",
                    "out_bw": "",
                    "k_bits": "",
                    "s_bits": "",
                    "WER": f"{wer:.2f}",
                })
                print(f"  {'original':>20} (T={sat}):  WER={wer:.2f}%")

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    df = pd.DataFrame(rows, columns=[
        "model", "ceiling", "softmax_type", "in_bw", "out_bw", "k_bits", "s_bits",
        "WER",
    ])
    csv_suffix = args.csv_name if args.csv_name else "_results"
    out_path = output_dir / f"eval_speech{csv_suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


# ----------------- Main function -----------------
def main():
    args = parser.parse_args()
    run_task(args)


if __name__ == "__main__":
    main()
