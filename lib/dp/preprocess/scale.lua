-----------------------------------------------------------------------
--[[ Binarize ]]--
-- A Preprocessor that sets to 0 any pixel strictly below the
-- threshold, sets to 1 those above (or equal) to the threshold.
-----------------------------------------------------------------------
local Scale = torch.class("dp.Scale", "dp.Preprocess")
Scale.isScale = true

function Scale:__init()
end

function Scale:apply(dv)
   local data = dv:input()
   local max = torch.max(data)
   data:mul(1/max)
   dv:input(data)
end
