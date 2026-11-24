# LightMax Hardware - Simulation Guide

## Prerequisites

| Tool    | Version | Check command           |
|---------|---------|-------------------------|
| GHDL    | ≥ 1.0   | `ghdl --version`         |
| Python  | ≥ 3.10  | `python3 --version`      |
| numpy   | any     | `python3 -c "import numpy"` |
| matplotlib | any  | `python3 -c "import matplotlib"` |

## Project structure

```
.
├── rtl/
│   ├── lightmax_core.vhd         # Core datapath (LOD, log-domain computation)
│   ├── exp_lut.vhd               # Decaying exponential LUT (BRAM)
│   ├── k_lut.vhd                 # k·ln(2) LUT (LUT6 primitives)
│   ├── s_lut.vhd                 # ln(1+s) LUT (LUT6 primitives)
│   ├── input_buffer.vhd          # Input storage for multi-unit
│   ├── axis_wrapper.vhd          # Single-unit AXI4-Stream wrapper
│   └── multi_lightmax_wrapper.vhd # P-unit parallel wrapper
├── tb/
│   ├── tb_lightmax.vhd           # Single-unit testbench
│   ├── tb_multi_lightmax.vhd     # Multi-unit testbench
│   └── tb_run_P*.vhd             # Run testbenches for P = 2, 4, 8, 16, 32
├── sim/                          # Simulation outputs (auto‑generated)
├── analyze_results.py            # Plot & compare with numpy reference
├── generate_vhdl_lut.py          # Generate LUT coefficient files
├── open_sim                      # Open waveform viewer (GTKWave)
├── run_sim                       # Single-unit simulation runner
├── run_sim_multi                 # Multi-unit simulation runner
├── run_sim_real.py               # Real-data simulation from ViT-B logits
└── README.md                     # This file
```

## Compilation and simulation

### Single Unit

```bash
./run_sim
```

### Multiple Units

```bash
./run_sim 4
```
Change `4` to the number of units you want to simulate in parallel. Must be even.

### With real data
We provide a Python script that extracts real logits from the attention matrix of the ViT-B ImageNet-1k validation dataset and feeds them to the LightMax core for simulation. The default vector is the first layer, first head, first row of the attention matrix.

```bash
python run_sim_real.py
```

Use `--random` to select a random layer, head and row.
```bash
python run_sim_real.py --random
```

For multi parallel units, you can specify the number of units as an argument:

```bash
python run_sim_real.py --multi 4
```

## Notes

- **`work-obj93.cf`** is GHDL's **library database file**. It is a text file created by `ghdl -a` in the working directory that maps every compiled entity/architecture to its corresponding `.o` object file. Deleting it forces a full recompile from scratch.
- The `shared variable` warnings during compilation are harmless, they come from GHDL's strict VHDL‑2002 checking of the shared variables used in the BRAM inference templates.