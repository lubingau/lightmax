"""run_sim_real.py

Python driver that captures real attention logits from a ViT model running
on ImageNet, feeds them into the GHDL LightMax hardware simulation, then
calls analyze_results.py to compare HW output vs Python reference.

Usage:
    python run_sim_real.py --data-dir /path/to/imagenet/val  [options]

All --ceiling/--in-bw/--out-bw/--k-bits/--s-bits options are forwarded to
both the logit capture and the analysis reference model.

Default behavior (synthetic data via ./run_sim) is unchanged.
"""

import argparse
import os
import pathlib
import random
import subprocess
import sys
import time

BASE_DIR = pathlib.Path(__file__).resolve().parent
SIM_DIR = BASE_DIR / "sim"

sys.path.insert(0, str(BASE_DIR.parent / "python"))
from measure_lightmax_mse import capture_and_export_logits


def main():
    parser = argparse.ArgumentParser(
        description="Run VHDL simulation with real captured attention logits."
    )
    # Capture args
    parser.add_argument("--data-dir", type=str, default="/media/lubin/Datasets/Imagenet/val", help="Path to ImageNet validation directory")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224", help="ViT model name from timm")
    parser.add_argument("--layer-idx", type=int, default=0, help="Attention layer index to capture logits from")
    parser.add_argument("--query-idx", type=int, default=0, help="Query row index (0 = CLS)")
    parser.add_argument("--head-idx", type=int, default=0, help="Attention head index")
    # LightMax config (must match VHDL generics)
    parser.add_argument("--ceiling", type=float, default=24.0)
    parser.add_argument("--in-bw", type=int, default=8)
    parser.add_argument("--out-bw", type=int, default=8)
    parser.add_argument("--k-bits", type=int, default=4)
    parser.add_argument("--s-bits", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")
    # Simulation
    parser.add_argument("--multi", type=int, default=1, help="Number of parallel units (even, >=2 for multi-unit)")
    parser.add_argument("--random", action="store_true", default=False, help="Pick random layer/query/head indices each run")
    parser.add_argument("--clean", action="store_true", default=False, help="Recompile all VHDL files")

    args = parser.parse_args()

    os.makedirs(SIM_DIR, exist_ok=True)
    os.chdir(BASE_DIR)

    # ---- Step 1: Capture real logits ----
    print("=" * 60, flush=True)
    print("Step 1: Capturing real attention logits...", flush=True)
    print(f"  model={args.model}, layer={args.layer_idx}, "
          f"query={args.query_idx}, head={args.head_idx}", flush=True)
    print(f"  ceiling={args.ceiling}, in_bw={args.in_bw}", flush=True)

    if args.random:
        args.layer_idx = random.randrange(12)
        args.query_idx = random.randrange(197)
        args.head_idx = random.randrange(12)
        print(f"  (random) layer={args.layer_idx}, query={args.query_idx}, "
              f"head={args.head_idx}", flush=True)

    is_multi = args.multi >= 2
    input_path = SIM_DIR / (f"sim_input_multi.txt" if is_multi else "sim_input.txt")

    n_values = capture_and_export_logits(
        args,
        str(input_path),
        layer_idx=args.layer_idx,
        query_idx=args.query_idx,
        head_idx=args.head_idx,
        num_queries=args.multi,
    )
    print(f"  Captured {n_values} logit values → {input_path}", flush=True)

    # ---- Step 2: Compile VHDL ----
    print(flush=True)
    print("Step 2: Compiling VHDL...", flush=True)

    rtl_dir = BASE_DIR / "rtl"
    vhdl_sources = [
        rtl_dir / "exp_lut.vhd",
        rtl_dir / "k_lut.vhd",
        rtl_dir / "s_lut.vhd",
        rtl_dir / "dual_exp_lut.vhd",
        rtl_dir / "input_buffer.vhd",
        rtl_dir / "lightmax_core.vhd",
        rtl_dir / "dual_lightmax.vhd",
        rtl_dir / "lightmax.vhd",
        rtl_dir / "lightmax_wrapper.vhd",
        rtl_dir / "multi_lightmax_wrapper.vhd",
    ]
    if is_multi:
        vhdl_sources.append(rtl_dir / "axis_wrapper_multi.vhd")
        tb_file = BASE_DIR / "tb" / "tb_lightmax_multi_file.vhd"
    else:
        vhdl_sources.append(rtl_dir / "axis_wrapper.vhd")
        tb_file = BASE_DIR / "tb" / "tb_lightmax_file.vhd"
    vhdl_sources.append(tb_file)

    if args.clean:
        for f in BASE_DIR.glob("work-obj93.cf"):
            f.unlink()
        for f in BASE_DIR.glob("*.o"):
            f.unlink()

    compile_cmd = ["ghdl", "-a", "--std=02", "-frelaxed"] + [str(s) for s in vhdl_sources]
    print(f"  {' '.join(compile_cmd)}", flush=True)
    subprocess.run(compile_cmd, check=True)

    # ---- Step 3: Elaborate ----
    print(flush=True)
    print("Step 3: Elaborating...", flush=True)
    if is_multi:
        elab_cmd = ["ghdl", "-e", "--std=02", "-frelaxed", "Tb_Lightmax_Multi_File"]
    else:
        elab_cmd = ["ghdl", "-e", "--std=02", "-frelaxed", "Tb_Lightmax_File"]
    print(f"  {' '.join(elab_cmd)}", flush=True)
    subprocess.run(elab_cmd, check=True)

    # ---- Step 4: Run simulation ----
    print(flush=True)
    print("Step 4: Running GHDL simulation...", flush=True)
    stop_time = f"{max(10_000, n_values * 200)}ns"
    if is_multi:
        entity_name = "Tb_Lightmax_Multi_File"
    else:
        entity_name = "Tb_Lightmax_File"
    run_cmd = [
        "ghdl", "-r", "--std=02", "-frelaxed",
        entity_name,
        f"-gNUM_VALUES={n_values}",
        f"-gEXP_LUT_BITS=7",
        f"-gK_LUT_IN_BITS={args.k_bits}",
        f"-gS_LUT_IN_BITS={args.s_bits}",
        f"-gK_LUT_OUT_BITS=6",
        f"-gS_LUT_OUT_BITS=3",
        f"--vcd={SIM_DIR / 'output.vcd'}",
        f"--stop-time={stop_time}",
    ]
    if is_multi:
        run_cmd.insert(9, f"-gPARALLEL_COUNT={args.multi}")

    print(f"  {' '.join(run_cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(run_cmd, check=True)
    elapsed = time.time() - t0
    print(f"  Simulation finished in {elapsed:.1f}s", flush=True)

    # ---- Step 5: Analyze results (forward LightMax config) ----
    print(flush=True)
    print("Step 5: Analyzing results...", flush=True)
    sys.argv = [
        "analyze_results.py",
        f"--ceiling={args.ceiling}",
        f"--in-bw={args.in_bw}",
        f"--out-bw={args.out_bw}",
        f"--k-bits={args.k_bits}",
        f"--s-bits={args.s_bits}",
    ]
    if is_multi:
        sys.argv.append(f"--units={args.multi}")
    import analyze_results
    analyze_results.main()


if __name__ == "__main__":
    main()
