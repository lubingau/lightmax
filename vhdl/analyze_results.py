"""analyze_results.py"""

import argparse
import os
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "python"))
from lightmax import LightMax


# ---------------- Argument parser ----------------
parser = argparse.ArgumentParser(description="Analyze LightMax simulation output")
parser.add_argument("--units", type=int, default=1, help="Number of parallel units (1, 2, 4, 8, 16, 32)")
parser.add_argument("--ceiling", type=float, default=24.0)
parser.add_argument("--in-bw", type=int, default=8)
parser.add_argument("--out-bw", type=int, default=8)
parser.add_argument("--k-bits", type=int, default=4)
parser.add_argument("--s-bits", type=int, default=6)

# LightMax config (set by main() from args)
LM_CEILING: float = 32.0
LM_IN_BW: int = 8
LM_OUT_BW: int = 8
LM_K_BITS: int = 4
LM_S_BITS: int = 6


def read_hw_output(path: str) -> np.ndarray:
    """Read single-unit hardware output file (one value per line)."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                data.append(int(parts[-1]))
    return np.array(data, dtype=np.int32)


def read_hw_output_dual(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read dual-unit hardware output file (two values per line).

    File format from testbench: (core_1_val, core_0_val).  Returns (core_0, core_1).
    """
    core0, core1 = [], []
    pat = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                core1.append(int(m.group(1)))
                core0.append(int(m.group(2)))
    return np.array(core0, dtype=np.int32), np.array(core1, dtype=np.int32)


def read_single_column(path: str, skip_header: bool = False) -> np.ndarray:
    """Read a single-column integer file (last column per line)."""
    data = []
    with open(path) as f:
        if skip_header:
            next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                data.append(int(parts[-1]))
    return np.array(data, dtype=np.int32)


def read_dual_input(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read dual-unit input file with columns: index val0 val1."""
    unit0, unit1 = [], []
    with open(path) as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                unit0.append(int(parts[1]))
                unit1.append(int(parts[2]))
    return np.array(unit0, dtype=np.int32), np.array(unit1, dtype=np.int32)


def compute_reference(inputs: np.ndarray) -> np.ndarray:
    """Compute LightMax softmax reference output for the given quantized inputs."""
    input_scale = LM_CEILING / (2 ** (LM_IN_BW - 1) - 1)
    model = LightMax(
        ceiling=LM_CEILING, in_bw=LM_IN_BW, out_bw=LM_OUT_BW,
        k_bits=LM_K_BITS, s_bits=LM_S_BITS,
    )
    model.eval()
    x = torch.tensor([v * input_scale for v in inputs], dtype=torch.float32)
    with torch.no_grad():
        probs = model(x)
    return (probs / model.output_scale).round().int().numpy()


def print_comparison(inputs: np.ndarray, hw_out: np.ndarray, ref_out: np.ndarray) -> None:
    """Print input, hardware output, reference, and difference table."""
    n = min(len(inputs), len(hw_out), len(ref_out))
    print(f"Read {n} values from simulation")
    print(f"{'idx':>4}  {'input':>6}  {'hw_out':>6}  {'ref_out':>6}  {'diff':>6}")
    print("-" * 40)
    for i in range(n):
        diff = int(hw_out[i]) - int(ref_out[i])
        print(f"{i:4d}  {inputs[i]:6d}  {hw_out[i]:6d}  {ref_out[i]:6d}  {diff:6d}")
    mae = np.mean(np.abs(hw_out[:n] - ref_out[:n]))
    print(f"\nMean Absolute Error: {mae:.3f}")


def plot_results(inputs: np.ndarray, hw_out: np.ndarray, ref_out: np.ndarray, save_path: str) -> None:
    """Plot input values and compare hardware output to reference."""
    n = min(len(inputs), len(hw_out), len(ref_out))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.stem(range(n), inputs, basefmt=" ")
    ax1.set_ylabel("Input (signed 8-bit)")
    ax1.set_title("LightMax — Input Values")
    ax1.grid(True, alpha=0.3)

    ax2.stem(range(n), hw_out, linefmt="C1-", markerfmt="C1o", basefmt=" ", label="HW output")
    ax2.stem(range(n), ref_out, linefmt="C2--", markerfmt="C2x", basefmt=" ", label="Reference")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Output (unsigned 8-bit)")
    ax2.set_title("LightMax — Hardware Output vs LightMax")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")


def analyze_single(sim_dir: str) -> None:
    """Analyze single-unit simulation results."""
    inp_path = os.path.join(sim_dir, "sim_input.txt")
    out_path = os.path.join(sim_dir, "sim_output.txt")

    inputs = read_single_column(inp_path, skip_header=True)
    outputs = read_hw_output(out_path)

    if len(outputs) == 0:
        print("ERROR: no data read — run the simulation first:")
        sys.exit(1)

    ref_out = compute_reference(inputs[:len(outputs)])
    print_comparison(inputs[:len(outputs)], outputs, ref_out)
    # plot_results(inputs[:len(outputs)], outputs, ref_out,
    #              os.path.join(sim_dir, "softmax_plot.png"))


def generate_unit_inputs(num_values: int, unit_count: int) -> list[np.ndarray]:
    """Generate test inputs matching tb_multi_lightmax formula: (i+1)*12 - j*(8-i)."""
    return [np.array([(i + 1) * 12 - j * (8 - i) for j in range(num_values)], dtype=np.int32)
            for i in range(unit_count)]


def read_output_file(path: str, unit_count: int) -> list[np.ndarray]:
    """Read a P-column output file (time then P values per line)."""
    data = [[] for _ in range(unit_count)]
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 + unit_count and parts[1] == 'ns':
                for i in range(unit_count):
                    data[i].append(int(parts[2 + i]))
    return [np.array(d, dtype=np.int32) for d in data]


def read_multi_input(path: str, unit_count: int) -> list[np.ndarray]:
    """Read multi-column input file (header + data)."""
    data = [[] for _ in range(unit_count)]
    with open(path) as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 1 + unit_count:
                for i in range(unit_count):
                    data[i].append(int(parts[1 + i]))
    return [np.array(d, dtype=np.int32) for d in data]


def analyze_units(sim_dir: str, unit_count: int) -> None:
    """Analyze parallel simulation results for the given parallel count.

    Tries sim_input_multi.txt / sim_output_multi.txt first (real data path).
    Falls back to sim_output_P{unit_count}.txt with synthetic inputs.
    """
    out_path_multi = os.path.join(sim_dir, "sim_output_multi.txt")
    out_path_named = os.path.join(sim_dir, f"sim_output_P{unit_count}.txt")

    out_path = out_path_multi if os.path.exists(out_path_multi) else out_path_named
    outputs = read_output_file(out_path, unit_count)

    if len(outputs[0]) == 0:
        print(f"ERROR: no data read from {out_path}")
        sys.exit(1)

    num_values = len(outputs[0])

    inp_path = os.path.join(sim_dir, "sim_input_multi.txt")
    if os.path.exists(inp_path):
        inputs = read_multi_input(inp_path, unit_count)
    else:
        inputs = generate_unit_inputs(num_values, unit_count)

    for i in range(unit_count):
        inp = inputs[i]
        ref = compute_reference(inp)
        label = f"unit_{i}"
        hw = outputs[i]
        print(f"\n======== {label} ========")
        print_comparison(inp, hw, ref)
        # plot_results(inp, hw, ref,
        #              os.path.join(sim_dir, f"softmax_plot_{label}.png"))


# ----------------- Main function -----------------
def main() -> None:
    global LM_CEILING, LM_IN_BW, LM_OUT_BW, LM_K_BITS, LM_S_BITS
    args = parser.parse_args()
    LM_CEILING = args.ceiling
    LM_IN_BW = args.in_bw
    LM_OUT_BW = args.out_bw
    LM_K_BITS = args.k_bits
    LM_S_BITS = args.s_bits

    base_dir = os.path.dirname(os.path.abspath(__file__))
    sim_dir = os.path.join(base_dir, "sim")

    if args.units >= 2:
        analyze_units(sim_dir, args.units)
    else:
        analyze_single(sim_dir)


if __name__ == "__main__":
    main()
