library ieee;
use ieee.std_logic_1164.all;

entity Tb_Run_L2 is
end Tb_Run_L2;

architecture sim of Tb_Run_L2 is
begin
    tb : entity work.Tb_MultiLightmax
        generic map (
            LANE_COUNT => 2,
            NUM_VALUES => 8
        );
end sim;
