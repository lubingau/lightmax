# LightMax - Evaluation Pipeline

## Scripts

**Saturation sweep**: runs all tasks across saturation thresholds 4-64 with baseline (no LightMax):
```bash
./run_all_saturations.sh
```

**LightMax evaluation**: runs all tasks with LightMax (default: c=24, 8-bit, k=4, s=6):
```bash
./run_lightmax.sh
```

Both write one CSV per task into `panda/`.

**MSE measurement**: measures LightMax MSE vs torch.softmax on real attention logits from ViT on ImageNet:
```bash
python measure_lightmax_mse.py --num-samples 10000
```

## Models & Datasets

| Task | Model | Dataset | Data parameters |
|------|-------|---------|----------------|
| Image | ViT-S/B/L, Swin-S/B/L | ImageNet-1k val | 224×224 images, ViT patch 16, Swin patch 4 window 7 |
| Language | ModernBERT-B/L | WikiText-103 | 512-token chunks |
| Speech | wav2vec2-B/L | LibriSpeech test-clean | Raw audio (variable length) |
| Time series | PatchTST | ETTh1/h2, ETTm1/m2, Weather | Sequence 512, horizon 96 |

Weights download automatically on first use (HF hub / timm).

## PatchTST Fine-Tuning

PatchTST is the only fine-tuned model:

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 1e-4 → 1e-6 (cosine annealing) |
| Epochs | 50 |
| Batch size | 64 |
| Loss | MSE |
| Precision | fp16 AMP |
| Train/val split | 80/20 chronological |
| Normalization | Per-channel z-score (train stats) |
| Sliding windows | seq 512 → horizon 96, stride 8 |

```bash
python finetune/finetune_patchtst.py
```

## Figures

Render the saturation figure (1×4 subplot) from the CSVs:
```bash
python figures/plot_saturation.py
```

Render the baseline-vs-LightMax comparison table (LaTeX):
```bash
python figures/plot_lightmax.py
```

Individual eval scripts (`eval_image.py`, `eval_language.py`, etc.) support more options (see `--help`).
