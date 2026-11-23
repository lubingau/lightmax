"""lightmax.py

LightMax: a softmax approximation using lookup tables for exponential
and log-sum, designed for hardware-efficient inference.

Single implementation: LightMax, with reversed (non-negative)
LUT indexing following the paper's log-domain reformulation (Section III-A).
"""

import math
import torch
import torch.nn as nn


class LightMax(nn.Module):
    """LightMax with reversed (non-negative) LUT indexing.

    Follows the paper's log-domain reformulation (Section III-A) and design
    choices (Section III-B).  Uses in-place tensor reuse to minimise peak
    memory and int32 intermediates.

    Args:
        ceiling: Clamp bound for pre-softmax logits (renamed from static_max).
        in_bw: Bit-width for quantized input representation.
        out_bw: Bit-width for quantized output representation.
        k_bits: Bits for the leading-one-detector (integer part of log2).
        s_bits: Bits for the mantissa of ln(S).
    """

    def __init__(
        self,
        ceiling: float = 24,
        in_bw: int = 10,
        out_bw: int = 8,
        k_bits: int = 4,
        s_bits: int = 6,
    ):
        super().__init__()
        self.ceiling = ceiling
        self.in_bw = in_bw
        self.out_bw = out_bw

        # Quantization scales
        self.input_scale = torch.tensor(
            ceiling / (2 ** (in_bw - 1) - 1), dtype=torch.float16
        )
        self.output_scale = torch.tensor(
            1.0 / (2 ** (out_bw - 1) - 1), dtype=torch.float16
        )
        self.ln_scale = (2 ** (in_bw - 1) - 1) / ceiling

        # Exponent LUT: 2^(in_bw-1) entries spanning [0, ceiling] so that
        # LUT[index] = exp(-index * input_scale)  (step == input_scale).
        self.exp_bw = in_bw - 1
        self.register_buffer(
            "exp_lut", torch.zeros(2**self.exp_bw, dtype=torch.int32)
        )
        self._init_exp_lut()

        # k.ln(2) LUT: 2^k_bits entries storing quantized k.ln(2).ln_scale
        # ln(1+s) LUT: 2^s_bits entries storing quantized ln(1+s/2^s_bits).ln_scale
        self.k_bits = k_bits
        self.s_bits = s_bits
        self.register_buffer(
            "k_lut", torch.zeros(2**k_bits, dtype=torch.int32)
        )
        self.register_buffer(
            "s_lut", torch.zeros(2**s_bits, dtype=torch.int32)
        )
        self._init_k_lut()
        self._init_s_lut()

    def _init_exp_lut(self):
        """Build reversed exp LUT: LUT[i] = exp(-i * input_scale) for i in [0, ceiling].

        Indexed directly by the non-negative quantized value x_stable_rev = x_max - x.
        """
        x_exp = torch.linspace(
            0.0, self.ceiling, 2**self.exp_bw, dtype=torch.float64
        )
        y_exp = torch.exp(-x_exp)  # reversed: LUT[0] = exp(0) = 1
        yq_exp = (
            torch.round(y_exp * (2 ** (self.out_bw - 1) - 1))
            .clamp(min=0, max=2 ** (self.out_bw - 1) - 1)
            .to(torch.int32)
        )
        self.exp_lut = yq_exp

    def _init_k_lut(self):
        """Build k_lut indexed by k_q: quantized (k * ln(2) - ln(out_scale_max)) * ln_scale."""
        ln2 = math.log(2.0)
        ln_out_scale = math.log(2 ** (self.out_bw - 1) - 1)  # correct for scaled exp LUT
        for k in range(2**self.k_bits):
            val = round((k * ln2 - ln_out_scale) * self.ln_scale)
            self.k_lut[k] = torch.tensor(val, dtype=torch.int32)

    def _init_s_lut(self):
        """Build s_lut indexed by s_q: quantized ln(1 + s_q/2^s_bits) * ln_scale."""
        denom = 2**self.s_bits
        for s_q in range(2**self.s_bits):
            s_val = s_q / denom
            val = round(math.log(1.0 + s_val) * self.ln_scale)
            self.s_lut[s_q] = torch.tensor(val, dtype=torch.int32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LightMax approximation with reversed LUT indexing.

        Uses in-place tensor reuse and int32 intermediates.

        Args:
            x: Input logits of any shape (softmax is applied on the last dim).

        Returns:
            Approximate softmax probabilities, same dtype as input.
        """
        orig_dtype = x.dtype
        x = x.to(torch.float16).clamp_(-self.ceiling, self.ceiling)
        xq = (
            torch.round(x / self.input_scale)
            .clamp(-(2 ** (self.in_bw - 1)), (2 ** (self.in_bw - 1)) - 1)
            .to(torch.int32)
        )
        exp_lut = self.exp_lut.to(xq.device)
        k_lut = self.k_lut.to(xq.device)
        s_lut = self.s_lut.to(xq.device)

        # --- hardware start here ---
        x_max = xq.max(dim=-1, keepdim=True).values
        xq = x_max - xq               # x_int32 -- non-negative
        del x_max
        xq.clamp_(0, len(self.exp_lut) - 1)
        exp_rev = exp_lut[xq]
        S = exp_rev.sum(dim=-1, keepdim=True)
        del exp_rev
        k = (
            (S.float().log2().floor())
            .to(torch.int32)
            .clamp(0, (1 << self.k_bits) - 1)
        )
        shift = (k - self.s_bits).clamp(min=0)
        s_int = (S >> shift) & ((1 << self.s_bits) - 1)
        s_q = s_int.clamp(0, (1 << self.s_bits) - 1)
        del S, shift, s_int
        lnsum_q = k_lut[k.clamp(0, len(k_lut) - 1)] + s_lut[s_q]
        del k, s_q
        xq.add_(lnsum_q)
        del lnsum_q
        xq.clamp_(0, len(self.exp_lut) - 1)
        exp_out = exp_lut[xq]
        # --- hardware end here ---
        probabilities = exp_out * self.output_scale
        return probabilities.to(orig_dtype)


def get_lut_sizes(luts):
    """Return the sizes of the LUTs in bytes."""
    n_entries = len(luts)
    n_bits_entry = int(math.ceil(math.log2(n_entries)))
    print(f"[INFO] LUT min: {min(luts)}, max: {max(luts)}")
    if min(luts) < 0:
        print("[WARNING] LUT has negative values, output will be signed")
        max_val = max(abs(min(luts)), max(luts))
        n_bits_output = int(math.ceil(math.log2(max_val + 1))) + 1
    else:
        max_val = max(luts)
        n_bits_output = int(math.ceil(math.log2(max_val + 1)))
    return n_entries, n_bits_entry, n_bits_output, n_entries * n_bits_entry * n_bits_entry


# ----------------- Main function -----------------
def main_lightmax():
    """Run a quick accuracy benchmark of LightMax vs torch.softmax."""
    import argparse

    parser = argparse.ArgumentParser(description="LightMax test")
    parser.add_argument("--ceiling", type=float, default=24)
    parser.add_argument("--in-bw", type=int, default=8)
    parser.add_argument("--out-bw", type=int, default=8)
    parser.add_argument("--k-bits", type=int, default=4)
    parser.add_argument("--s-bits", type=int, default=6)
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--shape", type=int, nargs=2, default=[16, 64])
    args = parser.parse_args()

    model = LightMax(
        ceiling=args.ceiling,
        in_bw=args.in_bw,
        out_bw=args.out_bw,
        k_bits=args.k_bits,
        s_bits=args.s_bits,
    )
    model.eval()

    exp_entries, exp_bits_entry, exp_bits_output, exp_size = get_lut_sizes(model.exp_lut)
    print(f"Exp LUT: {exp_entries} entries of {exp_bits_entry} bits each, {exp_bits_output} output bits, {exp_size} bits total\n")
    k_entries, k_bits_entry, k_bits_output, k_size = get_lut_sizes(model.k_lut)
    print(f"k LUT: {k_entries} entries of {k_bits_entry} bits each, {k_bits_output} output bits, {k_size} bits total\n")
    s_entries, s_bits_entry, s_bits_output, s_size = get_lut_sizes(model.s_lut)
    print(f"s LUT: {s_entries} entries of {s_bits_entry} bits each, {s_bits_output} output bits, {s_size} bits total\n")

    torch.manual_seed(42)
    maes = []
    for _ in range(args.num_trials):
        x = torch.randn(*args.shape) * 5
        with torch.no_grad():
            y_approx = model(x)
        y_ref = torch.softmax(x, dim=-1)
        maes.append((y_approx - y_ref).abs().mean().item())

    avg_mae = sum(maes) / len(maes)

    print(f"LightMax benchmark")
    print(f"  Parameters: ceiling={args.ceiling}, in_bw={args.in_bw}, "
          f"out_bw={args.out_bw}, k_bits={args.k_bits}, s_bits={args.s_bits}")
    print(f"  Input shape: {tuple(args.shape)}, trials: {args.num_trials}")
    print(f"  exp LUT: {len(model.exp_lut)} entries")
    print(f"  k LUT: {len(model.k_lut)} entries,  s LUT: {len(model.s_lut)} entries")
    print(f"  Average MAE vs softmax: {avg_mae:.6f}")
    return avg_mae


if __name__ == "__main__":
    print("=== LightMax Benchmark ===")
    main_lightmax()
