library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity InputBuffer is
    generic (
        DATA_INPUT   : integer := 8;
        BIT_ADDR_XIN : integer := 8;
        INPUT_DEPTH  : integer := 257
    );
    port (
        clk        : in std_logic;

        en_a       : in std_logic;
        we_a       : in std_logic;
        addr_a     : in unsigned(BIT_ADDR_XIN-1 downto 0);
        in_a       : in unsigned(DATA_INPUT-1 downto 0);
        out_a      : out unsigned(DATA_INPUT-1 downto 0)
    );
end InputBuffer;

architecture Behavioral of InputBuffer is

    type mem_input_t is array (0 to INPUT_DEPTH-1) of unsigned(DATA_INPUT-1 downto 0);
    shared variable mem_xin : mem_input_t := (others => (others => '0'));

    attribute ram_style : string;
    attribute ram_style of mem_xin : variable is "block";

begin

    process(clk)
    begin
        if rising_edge(clk) then
            if en_a = '1' then
                if we_a = '1' then
                    mem_xin(to_integer(addr_a)) := in_a;
                end if;
                out_a <= mem_xin(to_integer(addr_a));
            end if;
        end if;
    end process;

end Behavioral;
