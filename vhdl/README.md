# LightMax Hardware — Simulation Guide

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
├── tb/
│   └── tb_lightmax.vhd        # Testbench (random stimulus)
├── sim/                          # Simulation outputs (auto‑generated)
├── analyze_results.py            # Plot & compare with numpy reference
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
We provide a python script that extrcat real logits from attention matrix of ViT-B ImageNet-1k val dataset and feed them to the LightMax core for simulation. Default vector is first layer, first head, first row of the attention matrix.

```bash
python run_sim_real.py
```

Random while select random layer, head and row.
```bash
python run_sim_real.py --random
```

For multi parallel units, you can specify the number of units as an argument:

```bash
python run_sim_real.py --multi 4
```

## Notes

- **`work-obj93.cf`** is GHDL's **library database file**. It is a text file created by `ghdl -a` in the working directory that maps every compiled entity/architecture to its corresponding `.o` object file. Deleting it forces a full recompile from scratch.
- The `shared variable` warnings during compilation are harmless — they come from GHDL's strict VHDL‑2002 checking of the shared variables used in the BRAM inference templates.