"""plot_lightmax.py

Read multiple CSV files from panda/ and generate LaTeX comparison table(s)
showing baseline vs LightMax across all tasks and models.
Baseline values are read from files matching ``--sat-pattern`` (default:
``*_saturations.csv``); LightMax values from files matching
``--lightmax-pattern`` (default: ``*_lightmax*.csv``), filtered by the requested
configuration parameters (ceiling, in-bw, out-bw, k-bits, s-bits).
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ---------------- Argument parser ----------------
parser = argparse.ArgumentParser(
    description="Create LaTeX comparison table(s) for LightMax methods across all tasks."
)
parser.add_argument("--input-dir", type=str, default="panda",
                    help="Directory containing CSV files")
parser.add_argument("--output-dir", type=str, default="figures",
                    help="Directory for output files")
parser.add_argument("--output", type=str, default="comparison_all.tex",
                    help="Output LaTeX filename (default: comparison_all.tex)")
parser.add_argument("--ceiling", type=float, default=24.0,
                    help="Saturation ceiling filter (default: 24.0)")
parser.add_argument("--in-bw", type=int, default=8,
                    help="Input bandwidth filter (default: 8)")
parser.add_argument("--out-bw", type=int, default=8,
                    help="Output bandwidth filter (default: 8)")
parser.add_argument("--k-bits", type=int, default=4,
                    help="K bits filter (default: 4)")
parser.add_argument("--s-bits", type=int, default=6,
                    help="S bits filter (default: 6)")
parser.add_argument("--small-tables", action="store_true", default=False,
                    help="Also generate per-domain tables (default: only big table)")
parser.add_argument("--decimals-classification", type=int, default=2,
                    help="Decimal places for classification top1 (default: 2)")
parser.add_argument("--decimals-forecasting", type=int, default=4,
                    help="Decimal places for forecasting MSE (default: 4)")
parser.add_argument("--decimals-llm", type=int, default=2,
                    help="Decimal places for LLM Top1 (default: 2)")
parser.add_argument("--decimals-audio", type=int, default=2,
                    help="Decimal places for audio WER (default: 2)")
parser.add_argument("--pdf", action="store_true", default=False,
                    help="Compile PDF via pdflatex (default: False)")
parser.add_argument("--lightmax-pattern", type=str, default="*_lightmax*.csv",
                    help="Glob pattern for LightMax CSV files (default: *_lightmax*.csv)")
parser.add_argument("--sat-pattern", type=str, default="*_saturations.csv",
                    help="Glob pattern for saturation/baseline CSV files (default: *_saturations.csv)")


_NAME_MAP = {
    "vit_base_patch16_224":                        "ViT-B",
    "vit_small_patch16_224":                       "ViT-S",
    "vit_large_patch16_224":                       "ViT-L",
    "swin_base_patch4_window7_224":                "Swin-B",
    "swin_small_patch4_window7_224":               "Swin-S",
    "swin_large_patch4_window7_224":               "Swin-L",
    "answerdotai/ModernBERT-base":                "ModernBERT-B",
    "answerdotai/ModernBERT-large":               "ModernBERT-L",
    "facebook/wav2vec2-base-960h":                 "Wav2Vec2-B",
    "facebook/wav2vec2-large-960h":                "Wav2Vec2-L",
    "ibm-research/patchtst-etth1-pretrain/ETTh1":  "PatchTST-ETTh1",
    "ibm-research/patchtst-etth1-pretrain/ETTh2":  "PatchTST-ETTh2",
    "ibm-research/patchtst-etth1-pretrain/ETTm1":  "PatchTST-ETTm1",
    "ibm-research/patchtst-etth1-pretrain/ETTm2":  "PatchTST-ETTm2",
    "ibm-research/patchtst-etth1-pretrain/Weather": "PatchTST-Weather",
}

_TASK_CONFIG = {
    "eval_image":       ("Classification", "top1"),
    "eval_time_series": ("Forecasting",    "MSE"),
    "eval_language":    ("LLM",            "Top1"),
    "eval_speech":      ("Audio",          "WER"),
}


def _short_model(model_name: str) -> str:
    """Shorten model name via _NAME_MAP or fallback stripping."""
    if model_name in _NAME_MAP:
        return _NAME_MAP[model_name]
    s = model_name
    for prefix in ["ibm-research/", "answerdotai/", "facebook/"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.replace("-pretrain", "")


def _parse_task_from_filename(fname: str) -> str:
    """Extract task prefix from a filename like ``eval_image_lightmax_sat16_bw8_ks6.csv``."""
    for prefix in _TASK_CONFIG:
        if fname.startswith(prefix):
            return prefix
    m = re.match(r'^([^_]+)', fname)
    return m.group(1) if m else fname


def _build_table_df(input_dir: Path, ceiling: float, in_bw: int, out_bw: int,
                   k_bits: int, s_bits: int,
                   lightmax_pattern: str = "*_lightmax*.csv",
                   sat_pattern: str = "*_saturations.csv") -> pd.DataFrame:
    """Read baseline from saturations files and LightMax from lightmax files,
    merge into a single table with columns:
    ``Task``, ``Model``, ``Baseline``, ``LightMax``, ``Diff``.
    """
    # ------------------------------------------------------------------
    # 1. Baseline from pattern-matched saturations CSV files
    # ------------------------------------------------------------------
    sat_files = sorted(input_dir.glob(sat_pattern))
    baseline_rows = []
    for fpath in sat_files:
        task_prefix = _parse_task_from_filename(fpath.name)
        task_name, metric_col = _TASK_CONFIG.get(task_prefix, (task_prefix, "top1"))
        df = pd.read_csv(fpath)
        df["task"] = task_name
        df["_metric_col"] = metric_col
        # Keep only the "true" baseline: original softmax with NaN ceiling
        bl = df[df["softmax_type"] == "original"].copy()
        bl["ceiling"] = pd.to_numeric(bl["ceiling"], errors="coerce")
        bl = bl[bl["ceiling"].isna()]
        for _, row in bl.iterrows():
            baseline_rows.append({
                "task": task_name,
                "model": row["model"],
                "value": row[metric_col],
            })

    baseline_df = pd.DataFrame(baseline_rows).drop_duplicates(
        subset=["task", "model"], keep="first"
    )

    # ------------------------------------------------------------------
    # 2. LightMax from pattern-matched LightMax CSV files
    # ------------------------------------------------------------------
    lightmax_files = sorted(input_dir.glob(lightmax_pattern))
    lightmax_rows = []
    for fpath in lightmax_files:
        task_prefix = _parse_task_from_filename(fpath.name)
        task_name, metric_col = _TASK_CONFIG.get(task_prefix, (task_prefix, "top1"))
        df = pd.read_csv(fpath)
        df["task"] = task_name
        df["_metric_col"] = metric_col
        for c in ["ceiling", "in_bw", "out_bw", "k_bits", "s_bits"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Filter for lightmax matching the requested config
        mask = (
            (df["softmax_type"] == "lightmax") &
            (df["ceiling"].round(6) == float(ceiling)) &
            (df["in_bw"] == float(in_bw)) &
            (df["out_bw"] == float(out_bw)) &
            (df["k_bits"] == float(k_bits)) &
            (df["s_bits"] == float(s_bits))
        )
        subset = df[mask]
        for _, row in subset.iterrows():
            lightmax_rows.append({
                "task": task_name,
                "model": row["model"],
                "value": row[metric_col],
            })

    lightmax_df = pd.DataFrame(lightmax_rows).drop_duplicates(
        subset=["task", "model"], keep="first"
    )

    if baseline_df.empty or lightmax_df.empty:
        print("Warning: no baseline or LightMax data found.")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 3. Merge and compute diff
    # ------------------------------------------------------------------
    merged = baseline_df.merge(
        lightmax_df, on=["task", "model"],
        suffixes=("_baseline", "_lightmax"),
        how="inner",
    )

    # Build final table with formatted values
    table_rows = []
    for _, row in merged.iterrows():
        task = row["task"]
        bl_val = float(row["value_baseline"])
        lightmax_val = float(row["value_lightmax"])
        diff = bl_val - lightmax_val
        table_rows.append({
            "Task": task,
            "Model": _short_model(row["model"]),
            "Baseline": bl_val,
            "LightMax": lightmax_val,
            "Diff": diff,
        })

    result = pd.DataFrame(table_rows)
    # Sort by task then by model (within task using size order)
    task_order = [t for t, _ in _TASK_CONFIG.values()]
    result["_task_order"] = result["Task"].apply(
        lambda t: task_order.index(t) if t in task_order else 99
    )
    result = result.sort_values(["_task_order", "Model"]).drop(columns=["_task_order"])
    return result.reset_index(drop=True)


def _generate_latex_table(df: pd.DataFrame, caption: str, label: str,
                          precision_map: dict, include_task: bool = False) -> str:
    """Generate a LaTeX table string from DataFrame using pandas ``to_latex``.

    Parameters
    ----------
    df : DataFrame with columns Model, Baseline, LightMax, Diff
        (plus Task if include_task=True)
    """
    if df.empty:
        return "% (empty table -- no data)"

    display = df.copy()
    is_combined = include_task  # combined table has a Task column

    # Identify the value columns
    val_cols = ["Baseline", "LightMax"]

    # Format numeric columns as plain strings (no LaTeX escaping issues)
    for col in val_cols:
        display[col] = display[col].apply(
            lambda x: f"{float(x):.{precision_map.get(col, 4)}f}" if pd.notna(x) else "--"
        )

    # Format Diff with LaTeX: {\color{black}$\downarrow$ XX.XX} (blank when 0)
    def _fmt_diff(x):
        if pd.isna(x):
            return r"{\color{black}$\downarrow$ --}"
        val = float(x)
        if abs(val) == 0.0:
            return ""
        dec = precision_map.get("_diff_default", 4)
        return rf"{{\color{{black}}$\downarrow$ {val:.{dec}f}}}"

    # Determine decimal places for Diff from whichever metric is relevant
    # We'll store precision in task name for Diff or use a general approach
    display["Diff"] = display["Diff"].apply(
        lambda x: rf"{{\color{{black}}$\downarrow$ {float(x):.{precision_map.get('_diff_default', 4)}f}}}"
        if pd.notna(x) and abs(float(x)) != 0.0 else ("" if pd.notna(x) else r"{\color{black}$\downarrow$ --}")
    )

    # Build column format
    if is_combined:
        col_order = ["Task", "Model", "Baseline", "LightMax", "Diff"]
        col_fmt = "@{}llrrr@{}"
    else:
        col_order = ["Model", "Baseline", "LightMax", "Diff"]
        col_fmt = "@{}lrrr@{}"

    display = display[col_order]

    # Manually build the LaTeX to have full control over Diff column formatting
    # Since pandas to_latex with escape=True would escape our LaTeX in Diff,
    # we build the tabular content manually.

    lines = [
        r"\begin{table}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
    ]

    # Header row
    header = " & ".join(col_order) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Data rows
    for _, row in display.iterrows():
        cells = []
        for col in col_order:
            cells.append(str(row[col]))
        lines.append("  " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


def _generate_full_document(tables: list) -> str:
    """Wrap one or more LaTeX tables in a complete ``article`` document."""
    parts = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage[a4paper,margin=2cm]{geometry}",
        r"\usepackage{caption}",
        "",
        r"\begin{document}",
        "",
    ]
    for table_code in tables:
        parts.append(table_code)
        parts.append("")
    parts.append(r"\end{document}")
    return "\n".join(parts)


def _compile_pdf(tex_path: str) -> None:
    """Run ``pdflatex`` on the given .tex file."""
    out_dir = os.path.dirname(os.path.abspath(tex_path)) or "."
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", out_dir, tex_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            pdf_path = tex_path.replace(".tex", ".pdf")
            print(f"PDF written to: {pdf_path}")
        else:
            print(f"pdflatex exited with code {result.returncode}.", file=sys.stderr)
            for line in result.stderr.splitlines()[-10:]:
                print(f"  {line}", file=sys.stderr)
    except FileNotFoundError:
        print("pdflatex not found -- install TeX distribution or compile manually.",
              file=sys.stderr)


# ----------------- Main function -----------------
def main():
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build precision map: task_name -> decimal places
    precision_map = {
        "Classification": args.decimals_classification,
        "Forecasting":    args.decimals_forecasting,
        "LLM":            args.decimals_llm,
        "Audio":          args.decimals_audio,
    }

    # Build the combined table data
    combined_df = _build_table_df(
        input_dir, args.ceiling, args.in_bw, args.out_bw,
        args.k_bits, args.s_bits,
        lightmax_pattern=args.lightmax_pattern,
        sat_pattern=args.sat_pattern,
    )
    if combined_df.empty:
        print("No data found.")
        return

    # --- 1. Combined table (all tasks together) ---
    caption_main = (
        r"\textbf{LightMax Comparison} --- "
        f"ceiling={args.ceiling}, in-bw={args.in_bw}, out-bw={args.out_bw}, "
        f"k-bits={args.k_bits}, s-bits={args.s_bits}"
    )

    # We adjust precision_map to have per-task Diff precision
    # The Diff column uses the precision of the task's metric.
    # We propagate task info via a helper column that we drop before display.
    # Instead, we'll handle formatting in _generate_latex_table by passing
    # the per-task precision info through.

    # For the combined table, each row has a Task column. We'll pre-process
    # Diff per-row using per-task precision.

    def _apply_precision(row):
        task = row["Task"]
        dec = precision_map.get(task, 4)
        diff_val = row["Diff"]
        if pd.notna(diff_val):
            diff_f = float(diff_val)
            if abs(diff_f) == 0.0:
                row["Diff"] = ""
            else:
                arrow = r"$\downarrow$" if diff_f >= 0 else r"$\uparrow$"
                row["Diff"] = rf"{{\color{{black}}{arrow} {diff_f:.{dec}f}}}"
        else:
            row["Diff"] = r"{\color{black}$\downarrow$ --}"
        # Also format Baseline and LightMax
        for col in ["Baseline", "LightMax"]:
            v = row[col]
            if pd.notna(v):
                row[col] = f"{float(v):.{dec}f}"
            else:
                row[col] = "--"
        return row

    combined_display = combined_df.apply(_apply_precision, axis=1)
    combined_cols = ["Task", "Model", "Baseline", "LightMax", "Diff"]
    combined_display = combined_display[combined_cols]

    # Build combined table LaTeX
    col_fmt_combined = "@{}llccc@{}"
    lines = [
        r"\begin{table}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption_main}}}",
        r"\label{tab:lightmax_comparison}",
        rf"\begin{{tabular}}{{{col_fmt_combined}}}",
        r"\toprule",
        " & ".join(combined_cols) + r" \\",
        r"\midrule",
    ]
    for _, row in combined_display.iterrows():
        lines.append("  " + " & ".join(str(row[c]) for c in combined_cols) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    main_table = "\n".join(lines) + "\n"

    # --- 2. Per-task tables (only if --small-tables) ---
    per_task_tables = []
    if args.small_tables:
        for task_display in [t for t, _ in _TASK_CONFIG.values()]:
            task_df = combined_df[combined_df["Task"] == task_display].copy()
            if task_df.empty:
                continue
            # Format with task-specific precision
            dec = precision_map.get(task_display, 4)
            task_display_cols = ["Model", "Baseline", "LightMax", "Diff"]
            task_rows = []
            for _, row in task_df.iterrows():
                r = {"Model": row["Model"]}
                for col in ["Baseline", "LightMax"]:
                    v = row[col]
                    r[col] = f"{float(v):.{dec}f}" if pd.notna(v) else "--"
                diff_v = row["Diff"]
                if pd.notna(diff_v):
                    diff_f = float(diff_v)
                    if abs(diff_f) == 0.0:
                        r["Diff"] = ""
                    else:
                        arrow = r"$\downarrow$" if diff_f >= 0 else r"$\uparrow$"
                        r["Diff"] = rf"{{\color{{black}}{arrow} {diff_f:.{dec}f}}}"
                else:
                    r["Diff"] = r"{\color{black}$\downarrow$ --}"
                task_rows.append(r)
            task_display_df = pd.DataFrame(task_rows)

            caption_task = (
                f"\\textbf{{{task_display}}} --- LightMax Comparison "
                f"(ceiling={args.ceiling}, in-bw={args.in_bw}, out-bw={args.out_bw}, "
                f"k-bits={args.k_bits}, s-bits={args.s_bits})"
            )
            label_task = f"tab:lightmax_{task_display.lower().replace(' ', '_')}"
            col_fmt_task = "@{}lclr@{}"

            lines = [
                r"\begin{table}",
                r"\centering",
                r"\small",
                r"\setlength{\tabcolsep}{4pt}",
                rf"\caption{{{caption_task}}}",
                rf"\label{{{label_task}}}",
                rf"\begin{{tabular}}{{{col_fmt_task}}}",
                r"\toprule",
                " & ".join(task_display_cols) + r" \\",
                r"\midrule",
            ]
            for _, row in task_display_df.iterrows():
                lines.append(
                    "  " + " & ".join(str(row[c]) for c in task_display_cols) + r" \\"
                )
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")
            per_task_tables.append("\n".join(lines) + "\n")

    # --- 3. Write output ---
    all_tables = [main_table] + per_task_tables
    full_doc = _generate_full_document(all_tables)

    tex_path = output_dir / args.output
    with open(tex_path, "w") as f:
        f.write(full_doc)
    print(f"Saved: {tex_path}")

    if args.pdf:
        _compile_pdf(str(tex_path))


if __name__ == "__main__":
    main()
