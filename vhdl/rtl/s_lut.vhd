--------------------------------------------------------------------------------
-- Description: Combinational LUT for the mantissa of ln(1+S).
-- Type: Behavioral
-- Author: Anonymous
-- License: MIT
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity SLut is
    generic (
        DATA_S_LUT   : integer := 3;
        BIT_ADDR_S   : integer := 6;
        S_DEPTH      : integer := 64
    );
    port (
        addr      : in unsigned(BIT_ADDR_S-1 downto 0);
        data_out  : out unsigned(DATA_S_LUT-1 downto 0)
    );
end SLut;

architecture Behavioral of SLut is

    type array_s_t is array (0 to S_DEPTH-1) of unsigned(DATA_S_LUT-1 downto 0);
    signal mem_s : array_s_t := (
         0 => to_unsigned(0, DATA_S_LUT),   1 => to_unsigned(0, DATA_S_LUT),
         2 => to_unsigned(0, DATA_S_LUT),   3 => to_unsigned(0, DATA_S_LUT),
         4 => to_unsigned(0, DATA_S_LUT),   5 => to_unsigned(0, DATA_S_LUT),
         6 => to_unsigned(0, DATA_S_LUT),   7 => to_unsigned(1, DATA_S_LUT),
         8 => to_unsigned(1, DATA_S_LUT),   9 => to_unsigned(1, DATA_S_LUT),
        10 => to_unsigned(1, DATA_S_LUT),  11 => to_unsigned(1, DATA_S_LUT),
        12 => to_unsigned(1, DATA_S_LUT),  13 => to_unsigned(1, DATA_S_LUT),
        14 => to_unsigned(1, DATA_S_LUT),  15 => to_unsigned(1, DATA_S_LUT),
        16 => to_unsigned(1, DATA_S_LUT),  17 => to_unsigned(1, DATA_S_LUT),
        18 => to_unsigned(1, DATA_S_LUT),  19 => to_unsigned(1, DATA_S_LUT),
        20 => to_unsigned(1, DATA_S_LUT),  21 => to_unsigned(2, DATA_S_LUT),
        22 => to_unsigned(2, DATA_S_LUT),  23 => to_unsigned(2, DATA_S_LUT),
        24 => to_unsigned(2, DATA_S_LUT),  25 => to_unsigned(2, DATA_S_LUT),
        26 => to_unsigned(2, DATA_S_LUT),  27 => to_unsigned(2, DATA_S_LUT),
        28 => to_unsigned(2, DATA_S_LUT),  29 => to_unsigned(2, DATA_S_LUT),
        30 => to_unsigned(2, DATA_S_LUT),  31 => to_unsigned(2, DATA_S_LUT),
        32 => to_unsigned(2, DATA_S_LUT),  33 => to_unsigned(2, DATA_S_LUT),
        34 => to_unsigned(2, DATA_S_LUT),  35 => to_unsigned(2, DATA_S_LUT),
        36 => to_unsigned(2, DATA_S_LUT),  37 => to_unsigned(2, DATA_S_LUT),
        38 => to_unsigned(2, DATA_S_LUT),  39 => to_unsigned(3, DATA_S_LUT),
        40 => to_unsigned(3, DATA_S_LUT),  41 => to_unsigned(3, DATA_S_LUT),
        42 => to_unsigned(3, DATA_S_LUT),  43 => to_unsigned(3, DATA_S_LUT),
        44 => to_unsigned(3, DATA_S_LUT),  45 => to_unsigned(3, DATA_S_LUT),
        46 => to_unsigned(3, DATA_S_LUT),  47 => to_unsigned(3, DATA_S_LUT),
        48 => to_unsigned(3, DATA_S_LUT),  49 => to_unsigned(3, DATA_S_LUT),
        50 => to_unsigned(3, DATA_S_LUT),  51 => to_unsigned(3, DATA_S_LUT),
        52 => to_unsigned(3, DATA_S_LUT),  53 => to_unsigned(3, DATA_S_LUT),
        54 => to_unsigned(3, DATA_S_LUT),  55 => to_unsigned(3, DATA_S_LUT),
        56 => to_unsigned(3, DATA_S_LUT),  57 => to_unsigned(3, DATA_S_LUT),
        58 => to_unsigned(3, DATA_S_LUT),  59 => to_unsigned(3, DATA_S_LUT),
        60 => to_unsigned(3, DATA_S_LUT),  61 => to_unsigned(4, DATA_S_LUT),
        62 => to_unsigned(4, DATA_S_LUT),  63 => to_unsigned(4, DATA_S_LUT)
    );

begin

    data_out <= mem_s(to_integer(addr));

end Behavioral;
