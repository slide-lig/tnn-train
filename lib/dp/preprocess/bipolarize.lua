-----------------------------------------------------------------------
--[[ BinarizeFraction ]]--
-- A Preprocessor that sets to 0 any pixel strictly below the
-- threshold, sets to 1 those above (or equal) to the threshold.
-- threshold is determined that splits the dataset by the given fraction
-----------------------------------------------------------------------
local Bipolarize = torch.class("dp.Bipolarize", "dp.Preprocess")
Bipolarize.isBipolarize = true

function Bipolarize:__init(bipolarizationType)
   assert(type(bipolarizationType) == 'number', 'input is not a number!')
   self._bipolarizationType = bipolarizationType
end

function Bipolarize:apply(dv)
    local data = dv:input()

    if self._bipolarizationType == 2 then
      --  make all zero and positive values  1
      data[ data:ge(0) ] = 1;
      -- make all other values -1
      data[ data:lt(0) ] = -1;
    elseif self._bipolarizationType == 1 then
      --  make all positive values  1
      data[ data:gt(0) ] = 1;
      -- make all other values -1
      data[ data:le(0) ] = -1;
    else
      --  make all positive values  1
      data[ data:gt(0) ] = 1;
      -- make all negative values -1
      data[ data:lt(0) ] = -1;
    end

    dv:input(data)
end
