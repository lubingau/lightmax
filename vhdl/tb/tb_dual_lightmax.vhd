library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

entity Tb_DualLightmax is
end Tb_DualLightmax;

architecture Sim of Tb_DualLightmax is

    constant CLK_PERIOD  : time    := 10 ns;
    constant NUM_VALUES  : integer := 8;
    constant MAX_DEPTH   : integer := NUM_VALUES + 1;

    signal clk           : std_logic := '0';
    signal rst_n         : std_logic := '0';
    signal s_axis_tvalid : std_logic := '0';
    signal s_axis_tdata  : std_logic_vector(15 downto 0) := (others => '0');
    signal s_axis_tkeep  : std_logic_vector(1 downto 0) := (others => '1');
    signal s_axis_tlast  : std_logic := '0';
    signal s_axis_tready : std_logic;
    signal m_axis_tvalid : std_logic;
    signal m_axis_tdata  : std_logic_vector(15 downto 0);
    signal m_axis_tkeep  : std_logic_vector(1 downto 0);
    signal m_axis_tlast  : std_logic;
    signal m_axis_tready : std_logic := '1';

    signal done       : boolean := false;

    type t_int_array is array (0 to NUM_VALUES-1) of integer;
    signal captured_data_0  : t_int_array := (others => 0);
    signal captured_data_1  : t_int_array := (others => 0);
    signal capture_count    : integer := 0;

begin

    clk <= not clk after CLK_PERIOD / 2 when not done;

    dut : entity work.AxisWrapperMulti
        generic map (
            MAX_DEPTH    => MAX_DEPTH,
            DATA_INPUT   => 8,
            PARALLEL_COUNT   => 2
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

    capture_proc : process(clk)
        file fout : text open write_mode is "sim/sim_output_dual.txt";
        variable line_out : line;
    begin
        if rising_edge(clk) then
            if rst_n = '1' then
                if m_axis_tvalid = '1' and m_axis_tready = '1' then
                    if capture_count < NUM_VALUES then
                        captured_data_0(capture_count) <= to_integer(unsigned(m_axis_tdata(7 downto 0)));
                        captured_data_1(capture_count) <= to_integer(unsigned(m_axis_tdata(15 downto 8)));
                        write(line_out, now);
                        write(line_out, string'("  "));
                        write(line_out, to_integer(unsigned(m_axis_tdata(7 downto 0))));
                        write(line_out, string'("  "));
                        write(line_out, to_integer(unsigned(m_axis_tdata(15 downto 8))));
                        writeline(fout, line_out);
                        capture_count <= capture_count + 1;
                    end if;
                end if;
            end if;
        end if;
    end process;

    stim_proc : process
        type t_inputs is array (0 to 7) of integer;
        constant test_inputs_0 : t_inputs := (36, 28, 20, 12, 4, -4, -12, -20);
        constant test_inputs_1 : t_inputs := (40, 30, 10, 0, -10, -20, -30, -40);
        variable line_out : line;
        file fin          : text open write_mode is "sim/sim_input_dual.txt";
    begin
        rst_n <= '0';
        s_axis_tvalid <= '0';
        s_axis_tdata  <= (others => '0');
        s_axis_tlast  <= '0';
        wait for 100 ns;
        rst_n <= '1';
        wait for 20 ns;

        wait until rising_edge(clk);

        write(line_out, string'("index  value_0  value_1"));
        writeline(fin, line_out);

        for i in 0 to 7 loop
            s_axis_tdata  <= std_logic_vector(to_signed(test_inputs_1(i), 8)) & std_logic_vector(to_signed(test_inputs_0(i), 8));
            s_axis_tvalid <= '1';
            if i = 7 then
                s_axis_tlast <= '1';
            else
                s_axis_tlast <= '0';
            end if;

            write(line_out, i);
            write(line_out, string'("  "));
            write(line_out, test_inputs_0(i));
            write(line_out, string'("  "));
            write(line_out, test_inputs_1(i));
            writeline(fin, line_out);

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
