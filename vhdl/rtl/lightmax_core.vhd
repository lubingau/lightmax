--------------------------------------------------------------------------------
-- Description: Core LightMax datapath: subtract max, index LUTs, compute log-sum, output.
-- Type: Behavioral
-- Author: Anonymous
-- License: MIT
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity LightmaxCore is
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
        in_data     : in std_logic_vector(DATA_INPUT-1 downto 0);
        out_valid   : out std_logic;
        out_ready   : in std_logic;
        out_data    : out std_logic_vector(DATA_INPUT-1 downto 0);

        buf_en_a    : out std_logic;
        buf_we_a    : out std_logic;
        buf_addr_a  : out unsigned(DATA_INPUT-1 downto 0);
        buf_in_a    : out unsigned(DATA_INPUT-1 downto 0);
        buf_out_a   : in  unsigned(DATA_INPUT-1 downto 0);

        exp_en_a    : out std_logic;
        exp_addr_a  : out unsigned(EXP_LUT_BITS-1 downto 0);
        exp_out_a   : in  unsigned(EXP_LUT_BITS-1 downto 0)
    );
end LightmaxCore;

architecture Behavioral of LightmaxCore is

    constant X_OUT_BITS  : integer := 10;
    constant SUM_BITS        : integer := 16;

    signal max_value : signed(DATA_INPUT-1 downto 0) := (others => '0');
    signal index     : integer range 0 to MAX_DEPTH - 1 := 0;
    signal end_index : integer range 0 to MAX_DEPTH - 1 := 0;

    signal sum      : unsigned(SUM_BITS-1 downto 0) := (others => '0');

    signal k_val    : unsigned(K_LUT_IN_BITS-1 downto 0) := (others => '0');
    signal s_val    : unsigned(S_LUT_IN_BITS-1 downto 0) := (others => '0');

    signal k_lut_out  : signed(K_LUT_OUT_BITS - 1 downto 0) := (others => '0');
    signal s_lut_out  : unsigned(S_LUT_OUT_BITS - 1 downto 0) := (others => '0');
    signal ln_sum_reg : unsigned(K_LUT_IN_BITS + S_LUT_IN_BITS - 1 downto 0) := (others => '0');

    type state is (S_WAIT, S_GET_MAX, S_SUM_EXP, S_LOG_DOMAIN);
    signal actual_state : state := S_WAIT;
    signal slave_axis_tready : std_logic;

    signal in_valid_d : std_logic := '0';

    signal valid_sr : std_logic_vector(4 downto 0) := (others => '0');
    signal last_sr  : std_logic_vector(4 downto 0) := (others => '0');

    signal out_data_reg : std_logic_vector(DATA_INPUT-1 downto 0) := (others => '0');
    signal out_valid_sr : std_logic := '0';

begin

    k_lut_inst : entity work.KLut
        generic map (
            DATA_K_LUT   => K_LUT_OUT_BITS,
            BIT_ADDR_K   => K_LUT_IN_BITS,
            K_DEPTH      => 2**K_LUT_IN_BITS
        )
        port map (
            addr     => ln_sum_reg(K_LUT_IN_BITS-1 downto 0),
            data_out => k_lut_out
        );

    s_lut_inst : entity work.SLut
        generic map (
            DATA_S_LUT   => S_LUT_OUT_BITS,
            BIT_ADDR_S   => S_LUT_IN_BITS,
            S_DEPTH      => 2**S_LUT_IN_BITS
        )
        port map (
            addr     => ln_sum_reg(K_LUT_IN_BITS + S_LUT_IN_BITS - 1 downto K_LUT_IN_BITS),
            data_out => s_lut_out
        );

    in_ready <= slave_axis_tready;
    slave_axis_tready <= '1' when (actual_state = S_WAIT or actual_state = S_GET_MAX) else '0';

    out_valid <= out_valid_sr;
    out_data  <= out_data_reg;

    buf_en_a <= '1' when (actual_state /= S_LOG_DOMAIN) or (out_ready = '1' or valid_sr(3) = '0') else '0';

    lod_process : process(sum)
        variable v_k : natural range 0 to 15;
        variable v_shift : natural range 0 to 9;
    begin
        v_k := 0;
        for i in 15 downto 0 loop
            if sum(i) = '1' then
                v_k := i;
                exit;
            end if;
        end loop;
        case v_k is
            when 0 to 6 => v_shift := 0;
            when 7 => v_shift := 1;
            when 8 => v_shift := 2;
            when 9 => v_shift := 3;
            when 10 => v_shift := 4;
            when 11 => v_shift := 5;
            when 12 => v_shift := 6;
            when 13 => v_shift := 7;
            when 14 => v_shift := 8;
            when 15 => v_shift := 9;
        end case;
        k_val <= to_unsigned(v_k, K_LUT_IN_BITS);
        case v_shift is
            when 0 => s_val <= sum(5 downto 0);
            when 1 => s_val <= sum(6 downto 1);
            when 2 => s_val <= sum(7 downto 2);
            when 3 => s_val <= sum(8 downto 3);
            when 4 => s_val <= sum(9 downto 4);
            when 5 => s_val <= sum(10 downto 5);
            when 6 => s_val <= sum(11 downto 6);
            when 7 => s_val <= sum(12 downto 7);
            when 8 => s_val <= sum(13 downto 8);
            when 9 => s_val <= sum(14 downto 9);
            when others => s_val <= sum(5 downto 0);
        end case;
    end process;

    process(rst_n, clk)
        variable v_x_stable : signed(DATA_INPUT downto 0);
        variable v_x_out    : unsigned(EXP_LUT_BITS-1 downto 0);
        variable v_x_signed : signed(X_OUT_BITS - 1 downto 0);
    begin
        if rst_n = '0' then
            actual_state <= S_WAIT;
            index        <= 0;
            max_value    <= (others => '0');
            sum          <= (others => '0');
            in_valid_d   <= '0';

            buf_we_a     <= '0';
            buf_addr_a   <= (others => '0');
            buf_in_a     <= (others => '0');

            valid_sr     <= (others => '0');
            last_sr      <= (others => '0');

            exp_en_a     <= '0';
            exp_addr_a   <= (others => '0');

        elsif rising_edge(clk) then

            buf_we_a <= '0';
            exp_en_a <= '1';

            in_valid_d <= in_valid;

            ln_sum_reg(K_LUT_IN_BITS - 1 downto 0) <= k_val;
            ln_sum_reg(K_LUT_IN_BITS + S_LUT_IN_BITS - 1 downto K_LUT_IN_BITS) <= s_val;

            case actual_state is

                when S_WAIT =>
                    out_valid_sr <= '0';
                    if in_valid = '1' and slave_axis_tready = '1' then
                        buf_we_a   <= '1';
                        buf_in_a   <= unsigned(in_data);
                        max_value  <= signed(in_data);
                        buf_addr_a <= to_unsigned(index, DATA_INPUT);
                        index      <= index + 1;
                        actual_state <= S_GET_MAX;
                    end if;

                when S_GET_MAX =>
                    if in_valid = '1' and slave_axis_tready = '1' then
                        buf_we_a   <= '1';
                        buf_in_a   <= unsigned(in_data);
                        buf_addr_a <= to_unsigned(index, DATA_INPUT);
                        if signed(in_data) > max_value then
                            max_value <= signed(in_data);
                        end if;
                        index <= index + 1;
                    end if;

                    if in_valid = '0' and in_valid_d = '1' then
                        actual_state <= S_SUM_EXP;
                        end_index    <= index - 1;
                        index        <= 0;
                    end if;

                when S_SUM_EXP =>
                    valid_sr(4 downto 1) <= valid_sr(3 downto 0);
                    last_sr(4 downto 1)  <= last_sr(3 downto 0);

                    if index <= end_index then
                        buf_addr_a  <= to_unsigned(index, DATA_INPUT);
                        valid_sr(0) <= '1';
                        if index = end_index then
                            last_sr(0) <= '1';
                        else
                            last_sr(0) <= '0';
                        end if;
                        index <= index + 1;
                    else
                        valid_sr(0) <= '0';
                        last_sr(0)  <= '0';
                    end if;

                    if valid_sr(1) = '1' then
                        v_x_stable := resize(max_value, DATA_INPUT+1) - resize(signed(buf_out_a), DATA_INPUT+1);
                        if v_x_stable < 0 then
                            exp_addr_a <= (others => '0');
                        elsif v_x_stable < 128 then
                            exp_addr_a <= unsigned(v_x_stable(EXP_LUT_BITS-1 downto 0));
                        else
                            exp_addr_a <= (others => '1');
                        end if;
                    end if;

                    if valid_sr(3) = '1' then
                        sum <= sum + unsigned(resize(exp_out_a, SUM_BITS));
                    end if;

                    if valid_sr(4) = '1' and last_sr(4) = '1' then
                        actual_state <= S_LOG_DOMAIN;
                        index        <= 0;
                        valid_sr     <= (others => '0');
                        last_sr      <= (others => '0');
                    end if;

                when S_LOG_DOMAIN =>
                    if (out_ready = '1' or valid_sr(3) = '0') then
                        valid_sr(3 downto 1) <= valid_sr(2 downto 0);
                        last_sr(3 downto 1)  <= last_sr(2 downto 0);

                        if index <= end_index then
                            valid_sr(0) <= '1';
                            buf_addr_a  <= to_unsigned(index, DATA_INPUT);
                            if index = end_index then
                                last_sr(0) <= '1';
                            else
                                last_sr(0) <= '0';
                            end if;
                            index <= index + 1;
                        else
                            valid_sr(0) <= '0';
                            last_sr(0)  <= '0';
                        end if;

                        if valid_sr(1) = '1' then
                            v_x_signed := resize(max_value, X_OUT_BITS) - resize(signed(buf_out_a), X_OUT_BITS) + resize(signed(std_logic_vector(k_lut_out)), X_OUT_BITS) + resize(signed(std_logic_vector(s_lut_out)), X_OUT_BITS);
                            if v_x_signed < 0 then
                                exp_addr_a <= (others => '0');
                            elsif v_x_signed < 128 then
                                exp_addr_a <= unsigned(v_x_signed(EXP_LUT_BITS-1 downto 0));
                            else
                                exp_addr_a <= (others => '1');
                            end if;
                            exp_en_a <= '1';
                        end if;

                        if valid_sr(3) = '1' then
                            out_data_reg <= std_logic_vector(resize(exp_out_a, DATA_INPUT));
                            out_valid_sr <= '1';
                        else
                            out_valid_sr <= '0';
                        end if;
                    end if;

                    if valid_sr(3) = '1' and last_sr(3) = '1' and out_ready = '1' then
                        actual_state <= S_WAIT;
                        index        <= 0;
                        valid_sr     <= (others => '0');
                        last_sr      <= (others => '0');
                        sum          <= (others => '0');
                        exp_en_a     <= '0';
                    end if;

                when others =>
                    actual_state <= S_WAIT;

            end case;
        end if;
    end process;

end Behavioral;