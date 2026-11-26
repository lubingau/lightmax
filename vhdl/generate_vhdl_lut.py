"""generate_vhdl_lut.py

Generate VHDL LUT files (exp_lut.vhd, k_lut.vhd, s_lut.vhd) directly from the
Python LightMax reference model, ensuring bit-exact agreement between
Python reference and VHDL simulation.
"""

import argparse
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "python"))
from lightmax import LightMax


parser = argparse.ArgumentParser(
    description="Generate VHDL LUT files from the Python LightMax reference model."
)
parser.add_argument("--output-dir", type=str, default=None,
                    help="Output directory (default: <script_dir>/rtl)")
parser.add_argument("--ceiling", type=float, default=24.0)
parser.add_argument("--in-bw", type=int, default=8)
parser.add_argument("--out-bw", type=int, default=8)
parser.add_argument("--k-bits", type=int, default=4)
parser.add_argument("--s-bits", type=int, default=6)


def generate_exp_lut(model, output_path: str) -> None:
    depth = len(model.exp_lut)
    data_width = model.out_bw - 1
    bit_addr = int(math.log2(depth))

    lines = []
    lines.append("library ieee;")
    lines.append("use ieee.std_logic_1164.all;")
    lines.append("use ieee.numeric_std.all;")
    lines.append("")
    lines.append("entity ExpLut is")
    lines.append("    generic (")
    lines.append(f"        DATA_EXP     : integer := {data_width};")
    lines.append(f"        BIT_ADDR_EXP : integer := {bit_addr};")
    lines.append(f"        EXP_DEPTH    : integer := {depth}")
    lines.append("    );")
    lines.append("    port (")
    lines.append("        clk         : in std_logic;")
    lines.append("        en_a        : in std_logic;")
    lines.append("        addr_a      : in unsigned(BIT_ADDR_EXP-1 downto 0);")
    lines.append("        out_a       : out unsigned(DATA_EXP-1 downto 0)")
    lines.append("    );")
    lines.append("end ExpLut;")
    lines.append("")
    lines.append("architecture Behavioral of ExpLut is")
    lines.append("")
    lines.append("    type array_exp_t is array (0 to EXP_DEPTH-1) of unsigned(DATA_EXP-1 downto 0);")
    lines.append("    signal mem_exp : array_exp_t := (")

    for i in range(depth):
        val = int(model.exp_lut[i])
        sep = "," if i < depth - 1 else ""
        lines.append(f"         {i:2d} => to_unsigned({val:3d}, {data_width}){sep}")

    lines.append("    );")
    lines.append("")
    lines.append("    attribute ram_style : string;")
    lines.append("    attribute ram_style of mem_exp : signal is \"block\";")
    lines.append("")
    lines.append("begin")
    lines.append("")
    lines.append("    process(clk)")
    lines.append("    begin")
    lines.append("        if rising_edge(clk) then")
    lines.append("            if en_a = '1' then")
    lines.append("                out_a <= mem_exp(to_integer(addr_a));")
    lines.append("            end if;")
    lines.append("        end if;")
    lines.append("    end process;")
    lines.append("")
    lines.append("end Behavioral;")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Written {output_path}")


def generate_k_lut(model, output_path: str) -> None:
    depth = len(model.k_lut)
    max_val = max(abs(int(model.k_lut.min())), abs(int(model.k_lut.max())))
    data_width = max(2, int(math.ceil(math.log2(max_val + 1))) + 1)
    bit_addr = int(math.log2(depth))

    lines = []
    lines.append("library ieee;")
    lines.append("use ieee.std_logic_1164.all;")
    lines.append("use ieee.numeric_std.all;")
    lines.append("")
    lines.append("entity KLut is")
    lines.append("    generic (")
    lines.append(f"        DATA_K_LUT   : integer := {data_width};")
    lines.append(f"        BIT_ADDR_K   : integer := {bit_addr};")
    lines.append(f"        K_DEPTH      : integer := {depth}")
    lines.append("    );")
    lines.append("    port (")
    lines.append("        addr      : in unsigned(BIT_ADDR_K-1 downto 0);")
    lines.append("        data_out  : out signed(DATA_K_LUT-1 downto 0)")
    lines.append("    );")
    lines.append("end KLut;")
    lines.append("")
    lines.append("architecture Behavioral of KLut is")
    lines.append("")
    lines.append("    type array_k_t is array (0 to K_DEPTH-1) of signed(DATA_K_LUT-1 downto 0);")
    lines.append("    signal mem_k : array_k_t := (")

    for i in range(depth):
        val = int(model.k_lut[i])
        sep = "," if i < depth - 1 else ""
        lines.append(f"         {i:2d} => to_signed({val:4d}, DATA_K_LUT){sep}")

    lines.append("    );")
    lines.append("")
    lines.append("begin")
    lines.append("")
    lines.append("    data_out <= mem_k(to_integer(addr));")
    lines.append("")
    lines.append("end Behavioral;")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Written {output_path}")


def generate_s_lut(model, output_path: str) -> None:
    depth = len(model.s_lut)
    max_val = int(model.s_lut.max())
    data_width = max(2, int(math.ceil(math.log2(max_val + 1)))) if max_val > 0 else 2
    bit_addr = int(math.log2(depth))

    lines = []
    lines.append("library ieee;")
    lines.append("use ieee.std_logic_1164.all;")
    lines.append("use ieee.numeric_std.all;")
    lines.append("")
    lines.append("entity SLut is")
    lines.append("    generic (")
    lines.append(f"        DATA_S_LUT   : integer := {data_width};")
    lines.append(f"        BIT_ADDR_S   : integer := {bit_addr};")
    lines.append(f"        S_DEPTH      : integer := {depth}")
    lines.append("    );")
    lines.append("    port (")
    lines.append("        addr      : in unsigned(BIT_ADDR_S-1 downto 0);")
    lines.append("        data_out  : out unsigned(DATA_S_LUT-1 downto 0)")
    lines.append("    );")
    lines.append("end SLut;")
    lines.append("")
    lines.append("architecture Behavioral of SLut is")
    lines.append("")
    lines.append("    type array_s_t is array (0 to S_DEPTH-1) of unsigned(DATA_S_LUT-1 downto 0);")
    lines.append("    signal mem_s : array_s_t := (")

    for i in range(0, depth, 2):
        v0 = int(model.s_lut[i])
        v1 = int(model.s_lut[i + 1]) if i + 1 < depth else 0
        sep = "," if i + 1 < depth - 1 else ""
        lines.append(f"        {i:2d} => to_unsigned({v0}, DATA_S_LUT),  {i + 1:2d} => to_unsigned({v1}, DATA_S_LUT){sep}")

    lines.append("    );")
    lines.append("")
    lines.append("begin")
    lines.append("")
    lines.append("    data_out <= mem_s(to_integer(addr));")
    lines.append("")
    lines.append("end Behavioral;")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Written {output_path}")


def generate_dual_exp_lut(model, output_path: str) -> None:
    depth = len(model.exp_lut)
    data_width = model.out_bw - 1
    bit_addr = int(math.log2(depth))

    lines = []
    lines.append("library ieee;")
    lines.append("use ieee.std_logic_1164.all;")
    lines.append("use ieee.numeric_std.all;")
    lines.append("")
    lines.append("entity DualExpLut is")
    lines.append("    generic (")
    lines.append(f"        DATA_EXP     : integer := {data_width};")
    lines.append(f"        BIT_ADDR_EXP : integer := {bit_addr};")
    lines.append(f"        EXP_DEPTH    : integer := {depth}")
    lines.append("    );")
    lines.append("    port (")
    lines.append("        clk         : in std_logic;")
    lines.append("")
    lines.append("        en_a        : in std_logic;")
    lines.append("        addr_a      : in unsigned(BIT_ADDR_EXP-1 downto 0);")
    lines.append("        out_a       : out unsigned(DATA_EXP-1 downto 0);")
    lines.append("")
    lines.append("        en_b        : in std_logic;")
    lines.append("        addr_b      : in unsigned(BIT_ADDR_EXP-1 downto 0);")
    lines.append("        out_b       : out unsigned(DATA_EXP-1 downto 0)")
    lines.append("    );")
    lines.append("end DualExpLut;")
    lines.append("")
    lines.append("architecture Behavioral of DualExpLut is")
    lines.append("")
    lines.append("    type array_exp_t is array (0 to EXP_DEPTH-1) of unsigned(DATA_EXP-1 downto 0);")
    lines.append("    signal mem_exp : array_exp_t := (")

    for i in range(depth):
        val = int(model.exp_lut[i])
        sep = "," if i < depth - 1 else ""
        lines.append(f"         {i:2d} => to_unsigned({val:3d}, {data_width}){sep}")
    lines.append("    );")
    lines.append("")
    lines.append("    attribute ram_style : string;")
    lines.append("    attribute ram_style of mem_exp : signal is \"block\";")
    lines.append("")
    lines.append("begin")
    lines.append("")
    lines.append("    process(clk)")
    lines.append("    begin")
    lines.append("        if rising_edge(clk) then")
    lines.append("            if en_a = '1' then")
    lines.append("                out_a <= mem_exp(to_integer(addr_a));")
    lines.append("            end if;")
    lines.append("            if en_b = '1' then")
    lines.append("                out_b <= mem_exp(to_integer(addr_b));")
    lines.append("            end if;")
    lines.append("        end if;")
    lines.append("    end process;")
    lines.append("")
    lines.append("end Behavioral;")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Written {output_path}")


def main() -> None:
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(pathlib.Path(__file__).resolve().parent / "rtl")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating VHDL LUTs with: ceiling={args.ceiling}, "
          f"in_bw={args.in_bw}, out_bw={args.out_bw}, "
          f"k_bits={args.k_bits}, s_bits={args.s_bits}")

    model = LightMax(
        ceiling=args.ceiling,
        in_bw=args.in_bw,
        out_bw=args.out_bw,
        k_bits=args.k_bits,
        s_bits=args.s_bits,
    )
    model.eval()

    generate_exp_lut(model, os.path.join(args.output_dir, "exp_lut.vhd"))
    generate_k_lut(model, os.path.join(args.output_dir, "k_lut.vhd"))
    generate_s_lut(model, os.path.join(args.output_dir, "s_lut.vhd"))

    generate_dual_exp_lut(model, os.path.join(args.output_dir, "dual_exp_lut.vhd"))

    k_vals = [int(v) for v in model.k_lut]
    s_vals = [int(v) for v in model.s_lut]
    print(f"  k_lut range: [{min(k_vals)}, {max(k_vals)}]  (DATA_K_LUT={int(math.ceil(math.log2(max(abs(min(k_vals)), abs(max(k_vals))) + 1)) + 1)})")
    print(f"  s_lut range: [{min(s_vals)}, {max(s_vals)}]  (DATA_S_LUT={max(2, int(math.ceil(math.log2(max(s_vals) + 1)))) if max(s_vals) > 0 else 2})")


if __name__ == "__main__":
    main()
