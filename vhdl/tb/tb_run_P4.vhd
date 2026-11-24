library ieee;
use ieee.std_logic_1164.all;

entity Tb_Run_P4 is
end Tb_Run_P4;

architecture sim of Tb_Run_P4 is
begin
    tb : entity work.Tb_MultiLightmax
        generic map (
            PARALLEL_COUNT => 4,
            NUM_VALUES => 8
        );
end sim;
