-----------------------------------------------------------------------
--[[ BinarizeFraction ]]--
-- A Preprocessor that sets to 0 any pixel strictly below the
-- threshold, sets to 1 those above (or equal) to the threshold.
-- threshold is determined that splits the dataset by the given fraction
-----------------------------------------------------------------------
local BinarizeFraction = torch.class("dp.BinarizeFraction", "dp.Preprocess")
BinarizeFraction.isBinarizeFraction = true

function BinarizeFraction:__init(thresholdFraction)
   self._thresholdFraction = thresholdFraction or 0.5
end

function BinarizeFraction:apply(dv)
    local data = dv:input()
    local min = torch.min(data)
    local max = torch.max(data)
    local range = max - min
    local threshold = (self._thresholdFraction * range) + min
    data[data:lt(threshold)] = 0;
    data[data:ge(threshold)] = 1;
    dv:input(data)
end
