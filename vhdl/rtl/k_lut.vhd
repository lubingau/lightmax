--------------------------------------------------------------------------------
-- Description: Combinational LUT for the leading-one-detector (integer part of log2).
-- Type: Behavioral
-- Author: Anonymous
-- License: MIT
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity KLut is
    generic (
        DATA_K_LUT   : integer := 6;
        BIT_ADDR_K   : integer := 4;
        K_DEPTH      : integer := 16
    );
    port (
        addr      : in unsigned(BIT_ADDR_K-1 downto 0);
        data_out  : out signed(DATA_K_LUT-1 downto 0)
    );
end KLut;

architecture Behavioral of KLut is

    type array_k_t is array (0 to K_DEPTH-1) of signed(DATA_K_LUT-1 downto 0);
    signal mem_k : array_k_t := (
          0 => to_signed( -26, DATA_K_LUT),
          1 => to_signed( -22, DATA_K_LUT),
          2 => to_signed( -18, DATA_K_LUT),
          3 => to_signed( -15, DATA_K_LUT),
          4 => to_signed( -11, DATA_K_LUT),
          5 => to_signed(  -7, DATA_K_LUT),
          6 => to_signed(  -4, DATA_K_LUT),
          7 => to_signed(   0, DATA_K_LUT),
          8 => to_signed(   4, DATA_K_LUT),
          9 => to_signed(   7, DATA_K_LUT),
         10 => to_signed(  11, DATA_K_LUT),
         11 => to_signed(  15, DATA_K_LUT),
         12 => to_signed(  18, DATA_K_LUT),
         13 => to_signed(  22, DATA_K_LUT),
         14 => to_signed(  26, DATA_K_LUT),
         15 => to_signed(  29, DATA_K_LUT)
    );

begin

    data_out <= mem_k(to_integer(addr));

end Behavioral;
