"""plot_saturation.py

Read CSV files from panda/ and produce a single 1x4 subplot figure
showing performance vs saturation across all tasks.
Saturation on the x-axis (None/original plotted at "∞" right of the max).
Color encodes model type (hue) and model size (shade).
"""

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- Argument parser ----------------
parser = argparse.ArgumentParser(
    description="Plot performance vs saturation across all tasks in a 1x4 subplot."
)
parser.add_argument("--input-dir", type=str, default="panda",
                    help="Directory containing CSV files")
parser.add_argument("--output-dir", type=str, default="figures",
                    help="Directory for output plots")
parser.add_argument("--csv-name", type=str, default="_saturations",
                    help="CSV filename suffix (default: _saturations)")
parser.add_argument("--dpi", type=int, default=300)
parser.add_argument("--saturations", type=int, nargs="*", default=[4, 8, 12, 16, 24, 32, 64],
                    help="Saturation thresholds to include in the plot (default: 4 8 16 32 64)")
parser.add_argument("--saturations-label-mask", type=int, nargs="*", default=[12],
                    help="Saturation values whose x tick labels should be masked / hidden (default: 12 16)")
parser.add_argument("--highlight-ceiling", type=int, default=24,
                    help="Saturation ceiling value to highlight with a vertical line (default: 24)")
parser.add_argument("--no-highlight", action="store_true",
                    help="Disable the highlight vertical line")

caption = """\
Figure 1: Performance vs. saturation threshold across four domains. \
Each subplot shows model accuracy (or error) as a function of the softmax \
saturation ceiling, with the original unmodified softmax plotted at ∞. \
(a) Image Classification - top-1 accuracy on ImageNet-1k validation. \
(b) NLP - next-token prediction accuracy on WikiText-103. \
(c) Speech Recognition - word error rate on LibriSpeech test-clean. \
(d) Time Series Forecasting - MSE on the ETT benchmark (ETTh1, ETTh2, ETTm1, ETTm2). \
Colors encode model families (ModernBERT, ViT, Swin, PatchTST, Wav2Vec2) \
and line opacity encodes model size within each family."""

mpl.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["DejaVu Serif"],
    "font.size":       11,
})

_FAMILY_PALETTE = {
    "ModernBERT":  "#463480",
    "ViT":         "#007DC0",
    "Swin":        "#4dc26b",
    "PatchTST":    "#f2844b",
    "Wav2Vec2":    "#fcce25",
}
_PATCHTST_COLORS = {
    "PatchTST-Weather": "#fdd0a2",
    "PatchTST-ETTh1":    "#fdae6b",
    "PatchTST-ETTh2":    "#f2844b",
    "PatchTST-ETTm1":    "#d94801",
    "PatchTST-ETTm2":    "#a63603",
}


_SIZE_ALPHA = {"tiny": 0.1, "small": 0.25, "base": 0.5, "large": 1.00}

_STYLE_MAP = {
    "original":       ("-",  1.0, "o"),
}

_NAME_MAP = {
    "vit_base_patch16_224":                     "ViT-B",
    "vit_small_patch16_224":                    "ViT-S",
    "vit_large_patch16_224":                    "ViT-L",
    "swin_base_patch4_window7_224":             "Swin-B",
    "swin_small_patch4_window7_224":            "Swin-S",
    "swin_large_patch4_window7_224":            "Swin-L",
    "answerdotai/ModernBERT-base":             "ModernBERT-B",
    "answerdotai/ModernBERT-large":            "ModernBERT-L",
    "facebook/wav2vec2-base-960h":             "Wav2Vec2-B",
    "facebook/wav2vec2-large-960h":            "Wav2Vec2-L",
    "ibm-research/patchtst-etth1-pretrain/ETTh1": "PatchTST-ETTh1",
    "ibm-research/patchtst-etth1-pretrain/ETTh2": "PatchTST-ETTh2",
    "ibm-research/patchtst-etth1-pretrain/ETTm1": "PatchTST-ETTm1",
    "ibm-research/patchtst-etth1-pretrain/ETTm2": "PatchTST-ETTm2",
    "ibm-research/patchtst-etth1-pretrain/Weather": "PatchTST-Weather",
}

_YLABEL_MAP = {
    "Top1": "Top-1 (%)",
    "top1": "Top-1 (%)",
    "MSE":  "MSE",
    "WER":  "WER (%)",
}

_LEGEND_LOC_MAP = {
    "eval_image":       "lower right",
    "eval_language":    "lower right",
    "eval_speech":      "lower right",
    "eval_time_series": "upper right",
}

_FONT_SIZE_LABEL = 20
_FONT_SIZE_TICK = 16
_FONT_SIZE_LEGEND = 14
_FONT_SIZE_TITLE = 20

_SIZE_SORT_ORDER = {"large": 0, "base": 1, "small": 2, "tiny": 3}


def alpha_blend_to_hex(fg_hex, alpha, bg_hex="#ffffff"):
    """Blend foreground *fg_hex* over *bg_hex* at *alpha*, return a solid hex string."""
    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    def rgb_to_hex(r, g, b):
        return f"#{int(round(r*255)):02x}{int(round(g*255)):02x}{int(round(b*255)):02x}"
    fg = hex_to_rgb(fg_hex)
    bg = hex_to_rgb(bg_hex)
    blended = tuple(f * alpha + b * (1 - alpha) for f, b in zip(fg, bg))
    return rgb_to_hex(*blended)


def _legend_sort_key(model_name: str):
    """Return a sort key so legend entries order by family then size (L → B → S)."""
    base, size = _parse_model_type_size(model_name)
    return (base, _SIZE_SORT_ORDER.get(size, 99), model_name)


def _parse_model_type_size(model_name: str):
    """Return (type_label, size_label) for a model name."""
    name = model_name.lower()
    if "modernbert" in name:
        base = "ModernBERT"
    elif name.startswith("vit_"):
        base = "ViT"
    elif name.startswith("swin_"):
        base = "Swin"
    elif "patchtst" in name:
        base = "PatchTST"
    elif "wav2vec2" in name:
        base = "Wav2Vec2"
    else:
        base = model_name.split("/")[0].split("_")[0] if "_" in model_name else model_name[:20]
    size = next((s for s in _SIZE_ALPHA if s in name), "")
    return base, size


def _model_color(base: str, model_name: str) -> str:
    """Return the base hex colour for a model family."""
    if base == "PatchTST":
        short = _short_model(model_name)
        if short in _PATCHTST_COLORS:
            return _PATCHTST_COLORS[short]
    return _FAMILY_PALETTE.get(base, "#808080")


def _short_model(model_name: str) -> str:
    """Shorten long model names for legend labels."""
    if model_name in _NAME_MAP:
        return _NAME_MAP[model_name]
    # Fallback: strip known prefixes and -pretrain suffix
    s = model_name
    for prefix in ["ibm-research/", "answerdotai/", "facebook/"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.replace("-pretrain", "")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_task(ax, df, metric_col: str, title: str, saturations=None, ylim=None, label_mask_values=None, legend_loc="best", highlight_ceiling=24, no_highlight=False):
    """Draw one subplot on *ax*."""
    df = df.copy()
    df["ceiling_num"] = pd.to_numeric(df["ceiling"], errors="coerce")

    if saturations is not None:
        df = df[df["ceiling_num"].isna() | df["ceiling_num"].isin(saturations)]

    max_sat = df["ceiling_num"].max()
    if pd.isna(max_sat):
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="gray")
        ax.set_title(title, fontweight="bold")
        return

    inf_x = max_sat * 1.25
    models = sorted(df["model"].unique(), key=_legend_sort_key)
    stypes = df["softmax_type"].unique()

    for model in models:
        base_type, size = _parse_model_type_size(model)
        color = _model_color(base_type, model)
        label = _short_model(model)

        for stype in stypes:
            sdf = df[(df["model"] == model) & (df["softmax_type"] == stype)].copy()
            if sdf.empty:
                continue
            sdf["x"] = sdf["ceiling_num"].fillna(inf_x)
            sdf = sdf.sort_values("x")
            y = pd.to_numeric(sdf[metric_col], errors="coerce")
            base_alpha = _SIZE_ALPHA.get(size, 0.85)
            ls, style_alpha, marker = _STYLE_MAP.get(stype, ("-", 0.85, "o"))

            ax.plot(sdf["x"], y, marker=marker, label=label,
                    color=alpha_blend_to_hex(color, base_alpha * style_alpha), linestyle=ls, markersize=4, markeredgewidth=0)

    # ── highlight ────────────────────────────────────────────────────
    if not no_highlight and highlight_ceiling is not None:
        ax.axvline(x=highlight_ceiling, color="red", linestyle="--", linewidth=1.5, alpha=1, zorder=0)

    # ── x-axis ticks ─────────────────────────────────────────────────
    ticks = sorted(df["ceiling_num"].dropna().unique())
    mask_set = set(label_mask_values) if label_mask_values else set()
    tick_labels = []
    for t in ticks:
        tick_labels.append(f"{int(t)}" if int(t) not in mask_set else "")
    tick_labels.append("∞")
    ax.set_xticks(list(ticks) + [inf_x])
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(0, inf_x * 1.05)
    ax.tick_params(axis='x', labelsize=_FONT_SIZE_TICK)
    ax.tick_params(axis='y', labelsize=_FONT_SIZE_TICK)

    ax.set_xlabel("Saturation ceiling c", fontsize=_FONT_SIZE_LABEL)
    ax.set_ylabel(_YLABEL_MAP.get(metric_col, metric_col), fontsize=_FONT_SIZE_LABEL)
    ax.set_title(title, fontweight="normal", fontsize=_FONT_SIZE_TITLE, pad=10)
    ax.legend(fontsize=_FONT_SIZE_LEGEND, loc=legend_loc, handlelength=1.8)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_suffix = args.csv_name if args.csv_name else "_results"
    # tasks = [
    #     (f"eval_classification{csv_suffix}.csv",  "top1", "(a) Image Classification\n(ImageNet-1k val)", (60, None)),
    #     (f"eval_llm{csv_suffix}.csv",            "Top1", "(b) Natural Language Processing\n(WikiText-103)"),
    #     (f"eval_audio{csv_suffix}.csv",          "WER",  "(c) Speech Recognition\n(LibriSpeech test-clean)", (0, 6)),
    #     (f"eval_forecasting{csv_suffix}.csv",    "MSE",  "(d) Time Series Forecasting\n(ETTh1/ETTm1/ETTh2/ETTm2)"),
    # ]
    tasks = [
        (f"eval_image{csv_suffix}.csv",       "top1", "(a) ImageNet-1k val", (60, None)),
        (f"eval_language{csv_suffix}.csv",    "Top1", "(b) WikiText-103", (0, None)),
        (f"eval_speech{csv_suffix}.csv",      "WER",  "(c) LibriSpeech test-clean", (0, 6)),
        (f"eval_time_series{csv_suffix}.csv", "MSE",  "(d) ETT/Weather", (0, 1.05))
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for ax, (fname, metric, title, *ylim_args) in zip(axes, tasks):
        csv_path = input_dir / fname
        # look up legend location by matching csv filename prefix
        legend_loc = next((legend_loc for key, legend_loc in _LEGEND_LOC_MAP.items() if fname.startswith(key)), "best")
        if csv_path.exists():
            _plot_task(ax, pd.read_csv(csv_path), metric, title,
                       saturations=args.saturations, ylim=ylim_args[0] if ylim_args else None,
                       label_mask_values=args.saturations_label_mask,
                       legend_loc=legend_loc,
                       highlight_ceiling=args.highlight_ceiling, no_highlight=args.no_highlight)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=_FONT_SIZE_LABEL, color="gray")
            ax.set_title(title, fontweight="bold", fontsize=_FONT_SIZE_LABEL)
    
    fig.tight_layout(w_pad=1)
    # fig.suptitle(caption, fontsize=12, y=-0.2, x=0.01, ha='left', va='bottom', wrap=True)
    out_path = output_dir / "performance_vs_saturation.pdf"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
