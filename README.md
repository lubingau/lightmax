# LightMax

LightMax: a hardware-efficient softmax approximation based on lookup tables
for the exponential and the log-sum.

- **`python/`** — evaluation pipeline (LightMax module, attention hooks, the four
  task eval scripts, run scripts, figures, datasets/models docs). See
  [`python/README.md`](python/README.md).
- **`vhdl/`** — RTL implementation (LightMax core, LUTs, wrappers), GHDL
  testbenches, simulation runner scripts and golden regression outputs. See
  [`vhdl/README.md`](vhdl/README.md) and [`vhdl/DESCRIPTION.md`](vhdl/DESCRIPTION.md).
