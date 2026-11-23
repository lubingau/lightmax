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
│   ├── exp_lut.vhd               # Exponential LUT (128×7-bit BRAM ROM)
│   ├── k_lut.vhd                 # k·ln(2) LUT (64×9-bit signed, LUT6)
│   ├── s_lut.vhd                 # ln(1+s) LUT (64×2-bit unsigned, LUT6)
│   ├── input_buffer.vhd          # Dual-port input buffer BRAM
│   ├── lightmax.vhd           # Core LightMax softmax FSM + combinatorial LOD
│   └── axis_wrapper.vhd          # AXI4-Stream wrapper
├── tb/
│   └── tb_lightmax.vhd        # Testbench (random stimulus)
├── sim/                          # Simulation outputs (auto‑generated)
│   ├── output.vcd                # Waveform dump (view with GTKWave)
│   ├── sim_input.txt             # Injected random values
│   ├── sim_output.txt            # Captured hardware output
│   └── softmax_plot.png          # Comparison plot
├── analyze_results.py            # Plot & compare with numpy reference
└── README.md                     # This file
```

## Step-by-step

### 1. Compile all sources

```bash
cd /home/lubin/Projects/quantization/lightmax_hardware

ghdl -a --std=02 -frelaxed rtl/exp_lut.vhd
ghdl -a --std=02 -frelaxed rtl/k_lut.vhd
ghdl -a --std=02 -frelaxed rtl/s_lut.vhd
ghdl -a --std=02 -frelaxed rtl/input_buffer.vhd
ghdl -a --std=02 -frelaxed rtl/lightmax.vhd
ghdl -a --std=02 -frelaxed rtl/axis_wrapper.vhd
ghdl -a --std=02 -frelaxed tb/tb_lightmax.vhd
```

### 2. Elaborate the testbench

```bash
ghdl -e --std=02 -frelaxed Tb_Lightmax
```

### 3. Run the simulation

```bash
ghdl -r --std=02 -frelaxed Tb_Lightmax --vcd=sim/output.vcd --stop-time=10us
```

This produces inside `sim/`:
- `output.vcd` — waveform dump (viewable with GTKWave)
- `sim_input.txt` — injected random values
- `sim_output.txt` — captured hardware output

### 4. View waveforms (optional)

```bash
gtkwave sim/output.vcd
```

### 5. Analyze and plot results

```bash
python3 analyze_results.py
```

This prints a per-sample comparison table and saves `sim/softmax_plot.png`.

### One-liner (compile → simulate → analyze)

```bash
cd /home/lubin/Projects/quantization/lightmax_hardware && \
ghdl -a --std=02 -frelaxed rtl/exp_lut.vhd rtl/k_lut.vhd rtl/s_lut.vhd rtl/input_buffer.vhd rtl/lightmax.vhd rtl/axis_wrapper.vhd tb/tb_lightmax.vhd && \
ghdl -e --std=02 -frelaxed Tb_Lightmax && \
ghdl -r --std=02 -frelaxed Tb_Lightmax --vcd=sim/output.vcd --stop-time=10us && \
python3 analyze_results.py
```

## Clean up compiled and simulation artifacts

```bash
rm -f work-obj93.cf *.o
rm -f sim/output.vcd sim/sim_input.txt sim/sim_output.txt sim/softmax_plot.png
```

## Notes

- **`work-obj93.cf`** is GHDL's **library database file**. It is a text file created by `ghdl -a` in the working directory that maps every compiled entity/architecture to its corresponding `.o` object file. Deleting it forces a full recompile from scratch.
- The `shared variable` warnings during compilation are harmless — they come from GHDL's strict VHDL‑2002 checking of the shared variables used in the BRAM inference templates.
