library ieee;
use ieee.std_logic_1164.all;

entity Tb_Run_P32 is
end Tb_Run_P32;

architecture sim of Tb_Run_P32 is
begin
    tb : entity work.Tb_MultiLightmax
        generic map (
            PARALLEL_COUNT => 32,
            NUM_VALUES => 8
        );
end sim;
