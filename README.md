# LightMax

**LightMax** is an ultra-light, integer-only Softmax approximation for transformer inference on FPGAs. It replaces the exponential with a LUT read and the division with a LUT-based log-domain computation, decomposing the normalization sum into exponent and mantissa representation. No fine-tuning or calibration is required.

Across five model architectures (ViT, Swin, wav2vec 2.0, ModernBERT, PatchTST) and four datasets, LightMax achieves near-lossless accuracy while using **zero DSP slices, a single 36 Kb BRAM, and 142 LUTs** at 300 MHz on a Xilinx UltraScale+ FPGA, a 1.4× LUT reduction over the closest prior work.

## Requirements

- **GHDL** ≥ 1.0 (for VHDL simulation).
- **Python** ≥ 3.10: create a venv and install dependencies with:
  ```bash
  uv venv && uv pip install torch timm transformers datasets pandas matplotlib tqdm pillow
  ```
  Requires [`uv`](https://docs.astral.sh/uv/). The Python environment is needed for both the evaluation pipeline and VHDL simulation analysis.

## Repository structure

- **`python/`**: evaluation pipeline to run LightMax approximation on real models and datasets. See [`python/README.md`](python/README.md).
- **`vhdl/`**: RTL implementation (LightMax core, LUTs, AXI4-Stream wrappers) and GHDL simulation. See [`vhdl/README.md`](vhdl/README.md).
