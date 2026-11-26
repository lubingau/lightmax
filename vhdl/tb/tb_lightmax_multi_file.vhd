--------------------------------------------------------------------------------
-- Description: Testbench for multi-unit Lightmax reading inputs from file.
-- Type: Testbench
-- Author: Anonymous
-- License: MIT
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

entity Tb_Lightmax_Multi_File is
    generic (
        PARALLEL_COUNT      : integer := 2;
        NUM_VALUES      : integer := 8;
        EXP_LUT_BITS    : integer := 7;
        K_LUT_IN_BITS   : integer := 4;
        S_LUT_IN_BITS   : integer := 6;
        K_LUT_OUT_BITS  : integer := 6;
        S_LUT_OUT_BITS  : integer := 3
    );
end Tb_Lightmax_Multi_File;

architecture Sim of Tb_Lightmax_Multi_File is

    constant CLK_PERIOD  : time := 10 ns;
    constant DATA_INPUT  : integer := 8;
    constant DATA_WIDTH  : integer := PARALLEL_COUNT * DATA_INPUT;
    constant MAX_DEPTH   : integer := NUM_VALUES + 1;

    type t_int_array is array (0 to PARALLEL_COUNT-1) of integer;

    signal clk           : std_logic := '0';
    signal rst_n         : std_logic := '0';
    signal s_axis_tvalid : std_logic := '0';
    signal s_axis_tdata  : std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
    signal s_axis_tkeep  : std_logic_vector((DATA_WIDTH+7)/8-1 downto 0) := (others => '1');
    signal s_axis_tlast  : std_logic := '0';
    signal s_axis_tready : std_logic;
    signal m_axis_tvalid : std_logic;
    signal m_axis_tdata  : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal m_axis_tkeep  : std_logic_vector((DATA_WIDTH+7)/8-1 downto 0);
    signal m_axis_tlast  : std_logic;
    signal m_axis_tready : std_logic := '1';

    signal done          : boolean := false;
    signal capture_count : integer := 0;

    signal input_values  : t_int_array;

begin

    clk <= not clk after CLK_PERIOD / 2 when not done;

    dut : entity work.AxisWrapperMulti
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

    capture_proc : process(clk, rst_n)
        file fout : text open write_mode is "sim/sim_output_multi.txt";
        variable line_out : line;
    begin
        if rst_n = '0' then
            capture_count <= 0;
        elsif rising_edge(clk) then
            if m_axis_tvalid = '1' and m_axis_tready = '1' then
                if capture_count < NUM_VALUES then
                    write(line_out, now);
                    for i in 0 to PARALLEL_COUNT-1 loop
                        write(line_out, string'("  "));
                        write(line_out, to_integer(unsigned(m_axis_tdata((i+1)*DATA_INPUT-1 downto i*DATA_INPUT))));
                    end loop;
                    writeline(fout, line_out);
                    capture_count <= capture_count + 1;
                end if;
            end if;
        end if;
    end process;

    stim_proc : process
        file fin          : text open read_mode is "sim/sim_input_multi.txt";
        variable line_in  : line;
        variable idx      : integer;
        variable val      : integer;
    begin
        rst_n <= '0';
        s_axis_tvalid <= '0';
        s_axis_tdata  <= (others => '0');
        s_axis_tlast  <= '0';
        wait for 100 ns;
        rst_n <= '1';
        wait for 20 ns;

        wait until rising_edge(clk);

        readline(fin, line_in);

        for j in 0 to NUM_VALUES-1 loop
            readline(fin, line_in);

            read(line_in, idx);

            for i in 0 to PARALLEL_COUNT-1 loop
                read(line_in, val);
                s_axis_tdata((i+1)*DATA_INPUT-1 downto i*DATA_INPUT)
                    <= std_logic_vector(to_signed(val, DATA_INPUT));
            end loop;

            s_axis_tvalid <= '1';
            if j = NUM_VALUES-1 then
                s_axis_tlast <= '1';
            else
                s_axis_tlast <= '0';
            end if;

            wait until rising_edge(clk) and s_axis_tready = '1';
        end loop;

        file_close(fin);

        s_axis_tvalid <= '0';
        s_axis_tlast  <= '0';

        wait until rising_edge(clk) and m_axis_tlast = '1' and m_axis_tready = '1';

        wait for 500 ns;

        done <= true;
        wait;
    end process;

end Sim;
