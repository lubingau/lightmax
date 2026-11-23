----------------------------------------------------------------------------------
-- Testbench for Lightmax via AxisWrapper
-- Injects random data via AXI4-Stream slave, captures softmax output,
-- and logs results to sim_output.txt for Python analysis.
--
-- Input transaction ends when s_axis_tvalid falls (no tlast).
-- The wrapper generates tlast on the master side.
----------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

entity Tb_Lightmax is
end Tb_Lightmax;

architecture Sim of Tb_Lightmax is

    -- ---------------- Constants ----------------
    constant CLK_PERIOD  : time    := 10 ns;
    constant NUM_VALUES  : integer := 8;
    constant MAX_DEPTH   : integer := NUM_VALUES + 1;

    -- ---------------- DUT Signals ----------------
    signal clk           : std_logic := '0';
    signal rst_n         : std_logic := '0';
    signal s_axis_tvalid : std_logic := '0';
    signal s_axis_tdata  : std_logic_vector(7 downto 0) := (others => '0');
    signal s_axis_tkeep  : std_logic_vector(0 downto 0) := (others => '1');
    signal s_axis_tlast  : std_logic := '0';
    signal s_axis_tready : std_logic;
    signal m_axis_tvalid : std_logic;
    signal m_axis_tdata  : std_logic_vector(7 downto 0);
    signal m_axis_tkeep  : std_logic_vector(0 downto 0);
    signal m_axis_tlast  : std_logic;
    signal m_axis_tready : std_logic := '1';

    signal done       : boolean := false;

    -- ---------------- Captured output arrays ----------------
    type t_int_array is array (0 to NUM_VALUES-1) of integer;
    signal captured_data  : t_int_array := (others => 0);
    signal capture_count  : integer := 0;

begin

    -- ---------------- Clock generation ----------------
    clk <= not clk after CLK_PERIOD / 2 when not done;

    -- ---------------- DUT Instantiation ----------------
    dut : entity work.AxisWrapper
        generic map (
            MAX_DEPTH    => MAX_DEPTH,
            DATA_INPUT   => 8
        )
        port map (
            clk           => clk,
            rst_n         => rst_n,
            s_axis_tvalid => s_axis_tvalid,
            s_axis_tdata  => s_axis_tdata,
            s_axis_tkeep  => s_axis_tkeep,
            s_axis_tlast  => s_axis_tlast,
            s_axis_tready => s_axis_tready,
            m_axis_tvalid => m_axis_tvalid,
            m_axis_tdata  => m_axis_tdata,
            m_axis_tkeep  => m_axis_tkeep,
            m_axis_tlast  => m_axis_tlast,
            m_axis_tready => m_axis_tready
        );

    -- ---------------- Capture output ----------------
    capture_proc : process(clk)
        file fout : text open write_mode is "sim/sim_output.txt";
        variable line_out : line;
    begin
        if rising_edge(clk) then
            if rst_n = '1' then
                if m_axis_tvalid = '1' and m_axis_tready = '1' then
                    if capture_count < NUM_VALUES then
                        captured_data(capture_count) <= to_integer(unsigned(m_axis_tdata));
                        write(line_out, now);
                        write(line_out, string'("  out["));
                        write(line_out, capture_count);
                        write(line_out, string'("] = "));
                        write(line_out, to_integer(unsigned(m_axis_tdata)));
                        writeline(fout, line_out);
                        capture_count <= capture_count + 1;
                    end if;
                end if;
            end if;
        end if;
    end process;

    -- ---------------- Stimulus process ----------------
    stim_proc : process
        type t_inputs is array (0 to 7) of integer;
        constant test_inputs : t_inputs := (36, 28, 20, 12, 4, -4, -12, -20);
        variable line_out : line;
        file fin          : text open write_mode is "sim/sim_input.txt";
    begin
        -- Reset
        rst_n <= '0';
        s_axis_tvalid <= '0';
        s_axis_tdata  <= (others => '0');
        s_axis_tlast  <= '0';
        wait for 100 ns;
        rst_n <= '1';
        wait for 20 ns;

        -- Align to clock edge so first data lasts a full cycle
        wait until rising_edge(clk);

        -- Write input file header
        write(line_out, string'("index  value"));
        writeline(fin, line_out);

        -- Send fixed inputs
        for i in 0 to 7 loop
            s_axis_tdata  <= std_logic_vector(to_signed(test_inputs(i), 8));
            s_axis_tvalid <= '1';
            if i = 7 then
                s_axis_tlast  <= '1';
            else
                s_axis_tlast  <= '0';
            end if;

            -- Log input
            write(line_out, i);
            write(line_out, string'("  "));
            write(line_out, test_inputs(i));
            writeline(fin, line_out);

            wait until rising_edge(clk) and s_axis_tready = '1';
        end loop;

        -- Deassert valid/tlast: falling edge triggers end of input transaction
        s_axis_tvalid <= '0';
        s_axis_tlast  <= '0';

        -- Wait for output to finish (tlast from wrapper marks the last word)
        wait until rising_edge(clk) and m_axis_tlast = '1' and m_axis_tready = '1';

        wait for 500 ns;

        done <= true;
        wait;
    end process;

end Sim;

