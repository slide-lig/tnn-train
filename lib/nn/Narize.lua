local Narize, parent = torch.class('nn.Narize', 'nn.Module')

function Narize:__init(_bitCount, _signed)
   parent.__init(self)
   assert(type(_bitCount) == 'number', 'input is not a number!')
   self.bitCount = _bitCount or 1
   self.signed = _signed or 0
end

function Narize:updateOutput(input)
  self.output:resizeAs(input):copy(input)

  local min = torch.min(input)
  local max = torch.max(input)
  local range = max - min

  self.output:csub(min)
  self.output:div(range)
  self.output:mul( (2^self.bitCount) - 1)

  -- round to nearest int
  self.output:add(0.5)
  self.output:floor()

  if self.signed > 0 then
    if self.bitCount == 1 then -- special case: signed n-arize to 1 bit results in -1/1
      self.output[self.output:eq(0)] = -1
    else -- standard two's complement
      self.output:csub( (2^(self.bitCount-1)) )
    end
  end

  return self.output

end

function Narize:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(gradOutput)
    self.gradInput:copy(gradOutput)
   return self.gradInput
end
