--------------------------------------------------------------------------------
-- Description: Configuration instantiating Tb_MultiLightmax with PARALLEL_COUNT = 4.
-- Type: Testbench
-- Author: Anonymous
-- License: MIT
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;

entity Tb_Run_P4 is
end Tb_Run_P4;

architecture Sim of Tb_Run_P4 is
begin
    tb : entity work.Tb_MultiLightmax
        generic map (
            PARALLEL_COUNT => 4,
            NUM_VALUES => 8
        );
end Sim;
