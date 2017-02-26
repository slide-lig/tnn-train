-----------------------------------------------------------------------
--[[ Narize ]]--
-- A Preprocessor that sets to 0 any pixel strictly below the
-- threshold, sets to 1 those above (or equal) to the threshold.
-- threshold is determined that splits the dataset by the given fraction
-----------------------------------------------------------------------
local Narize = torch.class("dp.Narize", "dp.Preprocess")
Narize.isNarize = true

function Narize:__init(_bitCount, _signed)
  self.bitCount = _bitCount or 1
  self.signed = _signed or 0
end

function Narize:apply(dv)
    local data = dv:input()
    local min = torch.min(data)
    local max = torch.max(data)
    local range = max - min

    data:csub(min)
    data:div(range)
    data:mul( (2^self.bitCount) - 1)

    -- round to nearest int
    data:add(0.5)
    data:floor()

    if self.signed > 0 then
      if self.bitCount == 1 then -- special case: signed n-arize to 1 bit results in -1/1
        data[data:eq(0)] = -1
      else -- standard two's complement
        data:csub( (2^(self.bitCount-1)) )
      end
    end

    dv:input(data)
end
