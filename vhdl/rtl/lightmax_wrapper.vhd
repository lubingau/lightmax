library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity LightmaxWrapper is
    generic (
        MAX_DEPTH       : integer := 257;
        DATA_INPUT      : integer := 8;
        EXP_LUT_BITS    : integer := 7;
        K_LUT_IN_BITS   : integer := 4;
        S_LUT_IN_BITS   : integer := 6;
        K_LUT_OUT_BITS  : integer := 6;
        S_LUT_OUT_BITS  : integer := 3
    );
    port (
        clk        : in std_logic;
        rst_n      : in std_logic;
        in_valid   : in std_logic;
        in_ready   : out std_logic;
        in_data    : in std_logic_vector(DATA_INPUT-1 downto 0);
        out_valid  : out std_logic;
        out_ready  : in std_logic;
        out_data   : out std_logic_vector(DATA_INPUT-1 downto 0)
    );
end LightmaxWrapper;

architecture Structural of LightmaxWrapper is

    signal buf_en_a    : std_logic;
    signal buf_we_a    : std_logic;
    signal buf_addr_a  : unsigned(DATA_INPUT-1 downto 0);
    signal buf_in_a    : unsigned(DATA_INPUT-1 downto 0);
    signal buf_out_a   : unsigned(DATA_INPUT-1 downto 0);

begin

    input_buf : entity work.InputBuffer
        generic map (
            DATA_INPUT   => DATA_INPUT,
            BIT_ADDR_XIN => DATA_INPUT,
            INPUT_DEPTH  => MAX_DEPTH
        )
        port map (
            clk    => clk,
            en_a   => buf_en_a,
            we_a   => buf_we_a,
            addr_a => buf_addr_a,
            in_a   => buf_in_a,
            out_a  => buf_out_a
        );

    core : entity work.Lightmax
        generic map (
            MAX_DEPTH       => MAX_DEPTH,
            DATA_INPUT      => DATA_INPUT,
            EXP_LUT_BITS    => EXP_LUT_BITS,
            K_LUT_IN_BITS   => K_LUT_IN_BITS,
            S_LUT_IN_BITS   => S_LUT_IN_BITS,
            K_LUT_OUT_BITS  => K_LUT_OUT_BITS,
            S_LUT_OUT_BITS  => S_LUT_OUT_BITS
        )
        port map (
            clk        => clk,
            rst_n      => rst_n,
            in_valid   => in_valid,
            in_ready   => in_ready,
            in_data    => in_data,
            out_valid  => out_valid,
            out_ready  => out_ready,
            out_data   => out_data,
            buf_en_a   => buf_en_a,
            buf_we_a   => buf_we_a,
            buf_addr_a => buf_addr_a,
            buf_in_a   => buf_in_a,
            buf_out_a  => buf_out_a
        );

end Structural;
