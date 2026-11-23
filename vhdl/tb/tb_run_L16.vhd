library ieee;
use ieee.std_logic_1164.all;

entity Tb_Run_L16 is
end Tb_Run_L16;

architecture sim of Tb_Run_L16 is
begin
    tb : entity work.Tb_MultiLightmax
        generic map (
            LANE_COUNT => 16,
            NUM_VALUES => 8
        );
end sim;
