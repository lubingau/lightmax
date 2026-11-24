library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity MultiLightmaxWrapper is
    generic (
        MAX_DEPTH       : integer := 257;
        DATA_INPUT      : integer := 8;
        PARALLEL_COUNT      : integer := 2;
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
        in_data     : in std_logic_vector(PARALLEL_COUNT*DATA_INPUT-1 downto 0);
        out_valid   : out std_logic;
        out_ready   : in std_logic;
        out_data    : out std_logic_vector(PARALLEL_COUNT*DATA_INPUT-1 downto 0)
    );
end MultiLightmaxWrapper;

architecture Structural of MultiLightmaxWrapper is

    constant DUAL_COUNT     : integer := PARALLEL_COUNT / 2;
    constant DATA_BUS_WIDTH : integer := PARALLEL_COUNT * DATA_INPUT;

    signal buf_in_a_wide  : unsigned(DATA_BUS_WIDTH-1 downto 0);
    signal buf_out_a_wide : unsigned(DATA_BUS_WIDTH-1 downto 0);
    signal buf_en_a       : std_logic;
    signal buf_we_a       : std_logic;
    signal buf_addr_a     : unsigned(DATA_INPUT-1 downto 0);

    signal core_in_ready  : std_logic_vector(0 to DUAL_COUNT-1);
    signal core_out_valid : std_logic_vector(0 to DUAL_COUNT-1);

    type t_data_slice is array (0 to DUAL_COUNT-1) of std_logic_vector(2*DATA_INPUT-1 downto 0);
    signal out_data_array : t_data_slice;

begin

    in_ready  <= core_in_ready(0);

    gen_concat: for i in 0 to DUAL_COUNT-1 generate
        out_data((i+1)*2*DATA_INPUT-1 downto i*2*DATA_INPUT) <= out_data_array(i);
    end generate;

    input_buf : entity work.InputBuffer
        generic map (
            DATA_INPUT   => DATA_BUS_WIDTH,
            BIT_ADDR_XIN => DATA_INPUT,
            INPUT_DEPTH  => MAX_DEPTH
        )
        port map (
            clk    => clk,
            en_a   => buf_en_a,
            we_a   => buf_we_a,
            addr_a => buf_addr_a,
            in_a   => buf_in_a_wide,
            out_a  => buf_out_a_wide
        );

    gen_cores: for i in 0 to DUAL_COUNT-1 generate
    begin
        gen_first: if i = 0 generate
            core : entity work.DualLightmax
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
                    in_ready   => core_in_ready(i),
                    in_data    => in_data((i+1)*2*DATA_INPUT-1 downto i*2*DATA_INPUT),
                    out_valid  => core_out_valid(i),
                    out_ready  => out_ready,
                    out_data   => out_data_array(i),
                    buf_en_a   => buf_en_a,
                    buf_we_a   => buf_we_a,
                    buf_addr_a => buf_addr_a,
                    buf_in_a   => buf_in_a_wide((i+1)*2*DATA_INPUT-1 downto i*2*DATA_INPUT),
                    buf_out_a  => buf_out_a_wide((i+1)*2*DATA_INPUT-1 downto i*2*DATA_INPUT)
                );
        end generate;

        gen_others: if i > 0 generate
            core : entity work.DualLightmax
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
                    in_ready   => core_in_ready(i),
                    in_data    => in_data((i+1)*2*DATA_INPUT-1 downto i*2*DATA_INPUT),
                    out_valid  => core_out_valid(i),
                    out_ready  => out_ready,
                    out_data   => out_data_array(i),
                    buf_en_a   => open,
                    buf_we_a   => open,
                    buf_addr_a => open,
                    buf_in_a   => buf_in_a_wide((i+1)*2*DATA_INPUT-1 downto i*2*DATA_INPUT),
                    buf_out_a  => buf_out_a_wide((i+1)*2*DATA_INPUT-1 downto i*2*DATA_INPUT)
                );
        end generate;
    end generate;

    out_valid <= core_out_valid(0);

end Structural;
