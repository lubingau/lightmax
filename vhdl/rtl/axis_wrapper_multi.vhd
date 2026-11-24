----------------------------------------------------------------------------------
-- AxisWrapperMulti
-- Wraps MultiLightmaxWrapper (P-unit) into a full AXI4-Stream.
-- s_axis_tkeep is ignored (internally forced to '1').
-- s_axis_tlast is not needed: end of input is detected by the core via
--   the falling edge of in_valid.
-- m_axis_tkeep is driven to '1'.
-- m_axis_tlast is generated from a 1-stage output pipeline that asserts
--   tlast on the cycle after out_valid goes low.
----------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity AxisWrapperMulti is
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
        clk           : in  std_logic;
        rst_n         : in  std_logic;

        s_axis_tvalid : in  std_logic;
        s_axis_tdata  : in  std_logic_vector(PARALLEL_COUNT*DATA_INPUT-1 downto 0);
        s_axis_tkeep  : in  std_logic_vector((PARALLEL_COUNT*DATA_INPUT+7)/8-1 downto 0);
        s_axis_tlast  : in  std_logic;
        s_axis_tready : out std_logic;

        m_axis_tvalid : out std_logic;
        m_axis_tdata  : out std_logic_vector(PARALLEL_COUNT*DATA_INPUT-1 downto 0);
        m_axis_tkeep  : out std_logic_vector((PARALLEL_COUNT*DATA_INPUT+7)/8-1 downto 0);
        m_axis_tlast  : out std_logic;
        m_axis_tready : in  std_logic
    );
end AxisWrapperMulti;

architecture rtl of AxisWrapperMulti is

    constant DATA_BUS_WIDTH : integer := PARALLEL_COUNT * DATA_INPUT;

    signal core_in_valid : std_logic;
    signal core_in_ready : std_logic;
    signal core_in_data  : std_logic_vector(DATA_BUS_WIDTH-1 downto 0);

    signal core_out_valid : std_logic;
    signal core_out_ready : std_logic;
    signal core_out_data  : std_logic_vector(DATA_BUS_WIDTH-1 downto 0);

    signal pipe_valid : std_logic := '0';
    signal pipe_data  : std_logic_vector(DATA_BUS_WIDTH-1 downto 0) := (others => '0');

begin

    core_in_valid <= s_axis_tvalid;
    core_in_data  <= s_axis_tdata;
    s_axis_tready <= core_in_ready;

    core : entity work.MultiLightmaxWrapper
        generic map (
            MAX_DEPTH       => MAX_DEPTH,
            DATA_INPUT      => DATA_INPUT,
            PARALLEL_COUNT      => PARALLEL_COUNT,
            EXP_LUT_BITS    => EXP_LUT_BITS,
            K_LUT_IN_BITS   => K_LUT_IN_BITS,
            S_LUT_IN_BITS   => S_LUT_IN_BITS,
            K_LUT_OUT_BITS  => K_LUT_OUT_BITS,
            S_LUT_OUT_BITS  => S_LUT_OUT_BITS
        )
        port map (
            clk       => clk,
            rst_n     => rst_n,
            in_valid  => core_in_valid,
            in_ready  => core_in_ready,
            in_data   => core_in_data,
            out_valid => core_out_valid,
            out_ready => core_out_ready,
            out_data  => core_out_data
        );

    core_out_ready <= not pipe_valid or m_axis_tready;

    process(clk, rst_n)
    begin
        if rst_n = '0' then
            pipe_valid <= '0';
            pipe_data  <= (others => '0');
        elsif rising_edge(clk) then
            if pipe_valid = '0' then
                if core_out_valid = '1' then
                    pipe_valid <= '1';
                    pipe_data  <= core_out_data;
                end if;
            else
                if m_axis_tready = '1' then
                    if core_out_valid = '1' then
                        pipe_data <= core_out_data;
                    else
                        pipe_valid <= '0';
                    end if;
                end if;
            end if;
        end if;
    end process;

    m_axis_tvalid <= pipe_valid;
    m_axis_tdata  <= pipe_data;
    m_axis_tkeep  <= (others => '1');
    m_axis_tlast  <= pipe_valid and not core_out_valid;

end rtl;
