library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ExpLut is
    generic (
        DATA_EXP     : integer := 7;
        BIT_ADDR_EXP : integer := 7;
        EXP_DEPTH    : integer := 128
    );
    port (
        clk         : in std_logic;
        en_a        : in std_logic;
        addr_a      : in unsigned(BIT_ADDR_EXP-1 downto 0);
        out_a       : out unsigned(DATA_EXP-1 downto 0)
    );
end ExpLut;

architecture Behavioral of ExpLut is

    type array_exp_t is array (0 to EXP_DEPTH-1) of unsigned(DATA_EXP-1 downto 0);
    signal mem_exp : array_exp_t := (
          0 => to_unsigned(127, 7),
          1 => to_unsigned(105, 7),
          2 => to_unsigned( 87, 7),
          3 => to_unsigned( 72, 7),
          4 => to_unsigned( 60, 7),
          5 => to_unsigned( 49, 7),
          6 => to_unsigned( 41, 7),
          7 => to_unsigned( 34, 7),
          8 => to_unsigned( 28, 7),
          9 => to_unsigned( 23, 7),
         10 => to_unsigned( 19, 7),
         11 => to_unsigned( 16, 7),
         12 => to_unsigned( 13, 7),
         13 => to_unsigned( 11, 7),
         14 => to_unsigned(  9, 7),
         15 => to_unsigned(  7, 7),
         16 => to_unsigned(  6, 7),
         17 => to_unsigned(  5, 7),
         18 => to_unsigned(  4, 7),
         19 => to_unsigned(  4, 7),
         20 => to_unsigned(  3, 7),
         21 => to_unsigned(  2, 7),
         22 => to_unsigned(  2, 7),
         23 => to_unsigned(  2, 7),
         24 => to_unsigned(  1, 7),
         25 => to_unsigned(  1, 7),
         26 => to_unsigned(  1, 7),
         27 => to_unsigned(  1, 7),
         28 => to_unsigned(  1, 7),
         29 => to_unsigned(  1, 7),
         30 => to_unsigned(  0, 7),
         31 => to_unsigned(  0, 7),
         32 => to_unsigned(  0, 7),
         33 => to_unsigned(  0, 7),
         34 => to_unsigned(  0, 7),
         35 => to_unsigned(  0, 7),
         36 => to_unsigned(  0, 7),
         37 => to_unsigned(  0, 7),
         38 => to_unsigned(  0, 7),
         39 => to_unsigned(  0, 7),
         40 => to_unsigned(  0, 7),
         41 => to_unsigned(  0, 7),
         42 => to_unsigned(  0, 7),
         43 => to_unsigned(  0, 7),
         44 => to_unsigned(  0, 7),
         45 => to_unsigned(  0, 7),
         46 => to_unsigned(  0, 7),
         47 => to_unsigned(  0, 7),
         48 => to_unsigned(  0, 7),
         49 => to_unsigned(  0, 7),
         50 => to_unsigned(  0, 7),
         51 => to_unsigned(  0, 7),
         52 => to_unsigned(  0, 7),
         53 => to_unsigned(  0, 7),
         54 => to_unsigned(  0, 7),
         55 => to_unsigned(  0, 7),
         56 => to_unsigned(  0, 7),
         57 => to_unsigned(  0, 7),
         58 => to_unsigned(  0, 7),
         59 => to_unsigned(  0, 7),
         60 => to_unsigned(  0, 7),
         61 => to_unsigned(  0, 7),
         62 => to_unsigned(  0, 7),
         63 => to_unsigned(  0, 7),
         64 => to_unsigned(  0, 7),
         65 => to_unsigned(  0, 7),
         66 => to_unsigned(  0, 7),
         67 => to_unsigned(  0, 7),
         68 => to_unsigned(  0, 7),
         69 => to_unsigned(  0, 7),
         70 => to_unsigned(  0, 7),
         71 => to_unsigned(  0, 7),
         72 => to_unsigned(  0, 7),
         73 => to_unsigned(  0, 7),
         74 => to_unsigned(  0, 7),
         75 => to_unsigned(  0, 7),
         76 => to_unsigned(  0, 7),
         77 => to_unsigned(  0, 7),
         78 => to_unsigned(  0, 7),
         79 => to_unsigned(  0, 7),
         80 => to_unsigned(  0, 7),
         81 => to_unsigned(  0, 7),
         82 => to_unsigned(  0, 7),
         83 => to_unsigned(  0, 7),
         84 => to_unsigned(  0, 7),
         85 => to_unsigned(  0, 7),
         86 => to_unsigned(  0, 7),
         87 => to_unsigned(  0, 7),
         88 => to_unsigned(  0, 7),
         89 => to_unsigned(  0, 7),
         90 => to_unsigned(  0, 7),
         91 => to_unsigned(  0, 7),
         92 => to_unsigned(  0, 7),
         93 => to_unsigned(  0, 7),
         94 => to_unsigned(  0, 7),
         95 => to_unsigned(  0, 7),
         96 => to_unsigned(  0, 7),
         97 => to_unsigned(  0, 7),
         98 => to_unsigned(  0, 7),
         99 => to_unsigned(  0, 7),
         100 => to_unsigned(  0, 7),
         101 => to_unsigned(  0, 7),
         102 => to_unsigned(  0, 7),
         103 => to_unsigned(  0, 7),
         104 => to_unsigned(  0, 7),
         105 => to_unsigned(  0, 7),
         106 => to_unsigned(  0, 7),
         107 => to_unsigned(  0, 7),
         108 => to_unsigned(  0, 7),
         109 => to_unsigned(  0, 7),
         110 => to_unsigned(  0, 7),
         111 => to_unsigned(  0, 7),
         112 => to_unsigned(  0, 7),
         113 => to_unsigned(  0, 7),
         114 => to_unsigned(  0, 7),
         115 => to_unsigned(  0, 7),
         116 => to_unsigned(  0, 7),
         117 => to_unsigned(  0, 7),
         118 => to_unsigned(  0, 7),
         119 => to_unsigned(  0, 7),
         120 => to_unsigned(  0, 7),
         121 => to_unsigned(  0, 7),
         122 => to_unsigned(  0, 7),
         123 => to_unsigned(  0, 7),
         124 => to_unsigned(  0, 7),
         125 => to_unsigned(  0, 7),
         126 => to_unsigned(  0, 7),
         127 => to_unsigned(  0, 7)
    );

    attribute ram_style : string;
    attribute ram_style of mem_exp : signal is "block";

begin

    process(clk)
    begin
        if rising_edge(clk) then
            if en_a = '1' then
                out_a <= mem_exp(to_integer(addr_a));
            end if;
        end if;
    end process;

end Behavioral;
