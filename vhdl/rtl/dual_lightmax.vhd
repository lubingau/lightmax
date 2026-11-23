library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity DualLightmax is
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
        clk         : in std_logic;
        rst_n       : in std_logic;
        in_valid    : in std_logic;
        in_ready    : out std_logic;
        in_data     : in std_logic_vector(2*DATA_INPUT-1 downto 0);
        out_valid   : out std_logic;
        out_ready   : in std_logic;
        out_data    : out std_logic_vector(2*DATA_INPUT-1 downto 0);

        buf_en_a    : out std_logic;
        buf_we_a    : out std_logic;
        buf_addr_a  : out unsigned(DATA_INPUT-1 downto 0);
        buf_in_a    : out unsigned(2*DATA_INPUT-1 downto 0);
        buf_out_a   : in  unsigned(2*DATA_INPUT-1 downto 0)
    );
end DualLightmax;

architecture Structural of DualLightmax is

    signal core0_in_ready   : std_logic;
    signal core0_out_valid  : std_logic;
    signal core0_out_data   : std_logic_vector(DATA_INPUT-1 downto 0);
    signal core0_buf_in_a   : unsigned(DATA_INPUT-1 downto 0);
    signal core0_exp_en_a   : std_logic;
    signal core0_exp_addr_a : unsigned(EXP_LUT_BITS-1 downto 0);

    signal core1_in_ready   : std_logic;
    signal core1_out_data   : std_logic_vector(DATA_INPUT-1 downto 0);
    signal core1_buf_in_a   : unsigned(DATA_INPUT-1 downto 0);
    signal core1_exp_en_a   : std_logic;
    signal core1_exp_addr_a : unsigned(EXP_LUT_BITS-1 downto 0);

    signal exp_out_a        : unsigned(EXP_LUT_BITS-1 downto 0);
    signal exp_out_b        : unsigned(EXP_LUT_BITS-1 downto 0);

begin

    buf_in_a <= core1_buf_in_a & core0_buf_in_a;

    in_ready  <= core0_in_ready;
    out_valid <= core0_out_valid;
    out_data  <= core1_out_data & core0_out_data;

    exp_lut_inst : entity work.DualExpLut
        generic map (
            DATA_EXP     => EXP_LUT_BITS,
            BIT_ADDR_EXP => EXP_LUT_BITS,
            EXP_DEPTH    => 2**EXP_LUT_BITS
        )
        port map (
            clk    => clk,
            en_a   => core0_exp_en_a,
            addr_a => core0_exp_addr_a,
            out_a  => exp_out_a,
            en_b   => core1_exp_en_a,
            addr_b => core1_exp_addr_a,
            out_b  => exp_out_b
        );

    core_0 : entity work.LightmaxCore
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
            in_ready   => core0_in_ready,
            in_data    => in_data(DATA_INPUT-1 downto 0),
            out_valid  => core0_out_valid,
            out_ready  => out_ready,
            out_data   => core0_out_data,
            buf_en_a   => buf_en_a,
            buf_we_a   => buf_we_a,
            buf_addr_a => buf_addr_a,
            buf_in_a   => core0_buf_in_a,
            buf_out_a  => buf_out_a(DATA_INPUT-1 downto 0),
            exp_en_a   => core0_exp_en_a,
            exp_addr_a => core0_exp_addr_a,
            exp_out_a  => exp_out_a
        );

    core_1 : entity work.LightmaxCore
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
            in_ready   => core1_in_ready,
            in_data    => in_data(2*DATA_INPUT-1 downto DATA_INPUT),
            out_valid  => open,
            out_ready  => out_ready,
            out_data   => core1_out_data,
            buf_en_a   => open,
            buf_we_a   => open,
            buf_addr_a => open,
            buf_in_a   => core1_buf_in_a,
            buf_out_a  => buf_out_a(2*DATA_INPUT-1 downto DATA_INPUT),
            exp_en_a   => core1_exp_en_a,
            exp_addr_a => core1_exp_addr_a,
            exp_out_a  => exp_out_b
        );

end Structural;
