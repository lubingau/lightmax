----------------------------------------------------------------------------------
-- Testbench for Lightmax via AxisWrapper — reads inputs from sim/sim_input.txt
--
-- Replaces the hardcoded constant array in tb_lightmax.vhd with a textio file
-- read, so real captured attention logits can be fed into the VHDL simulation.
-- Output is written to sim/sim_output.txt (same format), compatible with
-- analyze_results.py.
--
-- Generic NUM_VALUES must match the number of data lines in sim_input.txt.
----------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

entity Tb_Lightmax_File is
    generic (
        NUM_VALUES      : integer := 8;
        EXP_LUT_BITS    : integer := 7;
        K_LUT_IN_BITS   : integer := 4;
        S_LUT_IN_BITS   : integer := 6;
        K_LUT_OUT_BITS  : integer := 6;
        S_LUT_OUT_BITS  : integer := 3
    );
end Tb_Lightmax_File;

architecture Sim of Tb_Lightmax_File is

    constant CLK_PERIOD  : time    := 10 ns;
    constant MAX_DEPTH   : integer := NUM_VALUES + 1;

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

    signal done          : boolean := false;

    type t_int_array is array (0 to NUM_VALUES-1) of integer;
    signal captured_data  : t_int_array := (others => 0);
    signal capture_count  : integer := 0;

begin

    clk <= not clk after CLK_PERIOD / 2 when not done;

    dut : entity work.AxisWrapper
        generic map (
            MAX_DEPTH       => MAX_DEPTH,
            DATA_INPUT      => 8,
            EXP_LUT_BITS    => EXP_LUT_BITS,
            K_LUT_IN_BITS   => K_LUT_IN_BITS,
            S_LUT_IN_BITS   => S_LUT_IN_BITS,
            K_LUT_OUT_BITS  => K_LUT_OUT_BITS,
            S_LUT_OUT_BITS  => S_LUT_OUT_BITS
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

    -- ---------------- Stimulus process (reads from file) ----------------
    stim_proc : process
        file fin          : text open read_mode is "sim/sim_input.txt";
        variable line_buf : line;
        variable idx      : integer;
        variable val      : integer;
        variable num_read : integer := 0;
        type t_vals is array (0 to NUM_VALUES-1) of integer;
        variable inputs   : t_vals;
    begin
        -- Read all values from file
        readline(fin, line_buf);  -- skip header "index  value"
        while not endfile(fin) and num_read < NUM_VALUES loop
            readline(fin, line_buf);
            read(line_buf, idx);
            read(line_buf, val);
            inputs(num_read) := val;
            num_read := num_read + 1;
        end loop;

        if num_read /= NUM_VALUES then
            report "tb_lightmax_file: expected " &
                   integer'image(NUM_VALUES) & " values, read " &
                   integer'image(num_read)
            severity failure;
        end if;

        -- Reset
        rst_n <= '0';
        s_axis_tvalid <= '0';
        s_axis_tdata  <= (others => '0');
        s_axis_tlast  <= '0';
        wait for 100 ns;
        rst_n <= '1';
        wait for 20 ns;

        wait until rising_edge(clk);

        -- Send values from file
        for i in 0 to NUM_VALUES-1 loop
            s_axis_tdata  <= std_logic_vector(to_signed(inputs(i), 8));
            s_axis_tvalid <= '1';
            if i = NUM_VALUES-1 then
                s_axis_tlast  <= '1';
            else
                s_axis_tlast  <= '0';
            end if;

            wait until rising_edge(clk) and s_axis_tready = '1';
        end loop;

        s_axis_tvalid <= '0';
        s_axis_tlast  <= '0';

        wait until rising_edge(clk) and m_axis_tlast = '1' and m_axis_tready = '1';

        wait for 500 ns;

        done <= true;
        wait;
    end process;

end Sim;
