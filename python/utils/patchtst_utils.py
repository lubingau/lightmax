"""patchtst_utils.py

Shared helpers for PatchTST forecasting in this repo.

The HuggingFace pretrained checkpoints (e.g. ibm-research/patchtst-etth1-pretrain)
were saved with transformers 4.33, whose PatchTSTConfig/module names differ from
modern transformers (>=5.x): old kwargs (patch_size/stride/norm/positional_encoding/
encoder_layers/encoder_attention_heads) are silently ignored by new configs, and the
checkpoint tensor names do not match the new module tree.  This module builds configs
with the correct modern kwarg names and remaps old checkpoint keys so pretrained
weights actually load.
"""

import os
from pathlib import Path

import torch

# Architecture of the released pretrained checkpoints (transformers 4.33 config):
# patch_length=12, stride=12, d_model=128, 6 encoder layers, 16 heads, ffn 512.
PRETRAINED_ARCH = dict(
    patch_length=12,
    patch_stride=12,
    d_model=128,
    num_hidden_layers=6,
    num_attention_heads=16,
    ffn_dim=512,
)


def build_patchtst_config(num_channels: int, seq_len: int, forecast_horizon: int,
                          patch_length: int = 12, patch_stride: int = 12):
    """Build a PatchTSTConfig with modern (transformers >=5.x) kwarg names.

    Defaults match the released pretrained checkpoints so their weights load.
    """
    from transformers import PatchTSTConfig

    return PatchTSTConfig(
        num_input_channels=num_channels,
        context_length=seq_len,
        prediction_length=forecast_horizon,
        patch_length=patch_length,
        patch_stride=patch_stride,
        d_model=PRETRAINED_ARCH["d_model"],
        num_hidden_layers=PRETRAINED_ARCH["num_hidden_layers"],
        num_attention_heads=PRETRAINED_ARCH["num_attention_heads"],
        ffn_dim=PRETRAINED_ARCH["ffn_dim"],
        norm_type="batchnorm",
        positional_encoding_type="sincos",
        scaling="mean",
        loss="mse",
        pre_norm=False,
        use_cls_token=False,
        head_dropout=0.2,
    )


def remap_old_checkpoint(sd: dict) -> dict:
    """Map transformers-4.33 PatchTST tensor names to modern module names."""
    out = {}
    for k, v in sd.items():
        nk = k
        nk = nk.replace("model.encoder.encoder.layers.", "model.encoder.layers.")
        nk = nk.replace("norm_sublayer1.1.", "norm_sublayer1.batchnorm.")
        nk = nk.replace("norm_sublayer3.1.", "norm_sublayer3.batchnorm.")
        nk = nk.replace("model.encoder.w_p.", "model.encoder.embedder.input_embedding.")
        nk = nk.replace("model.encoder.w_pos", "model.encoder.positional_encoder.position_enc")
        nk = nk.replace("head.linear.", "head.projection.")
        out[nk] = v
    return out


def load_pretrained_patchtst(model_name: str, num_channels: int,
                             seq_len: int, forecast_horizon: int,
                             cache_dir: str | None = None):
    """Load a pretrained PatchTST checkpoint into a modern PatchTSTForPrediction.

    Downloads the checkpoint, remaps its 4.33-era tensor names, and transfers every
    tensor whose shape matches (the forecasting head is task-specific and stays
    randomly initialized).  Returns (model, report_dict).
    """
    from huggingface_hub import snapshot_download
    from transformers import PatchTSTForPrediction

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    snap = snapshot_download(model_name, cache_dir=cache_dir)

    bin_path = Path(snap) / "pytorch_model.bin"
    st_path = Path(snap) / "model.safetensors"
    if bin_path.exists():
        old_sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    elif st_path.exists():
        from safetensors.torch import load_file
        old_sd = load_file(st_path)
    else:
        raise FileNotFoundError(f"No weights file in {snap}")

    config = build_patchtst_config(num_channels, seq_len, forecast_horizon)
    model = PatchTSTForPrediction(config)

    new_sd = remap_old_checkpoint(old_sd)
    own = model.state_dict()
    compat = {k: v for k, v in new_sd.items() if k in own and own[k].shape == v.shape}
    missing = [k for k in own if k not in compat]
    own.update(compat)
    model.load_state_dict(own)

    report = {
        "model_name": model_name,
        "transferred": len(compat),
        "total": len(own),
        "random_init": missing,
    }
    print(f"  Pretrained load: {len(compat)}/{len(own)} tensors transferred from {model_name}")
    if missing:
        print(f"  Randomly initialized (task-specific): {missing}")
    return model, report
