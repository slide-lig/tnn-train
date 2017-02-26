local Binarize, parent = torch.class('nn.Binarize', 'nn.Module')

function Binarize:__init(_thresholdFraction)
   parent.__init(self)
   assert(type(_thresholdFraction) == 'number', 'input is not a number!')
   self._thresholdFraction = _thresholdFraction or 0.5
end

function Binarize:updateOutput(input)

  self.output:resizeAs(input)

  local min = torch.min(input)
  local max = torch.max(input)
  local range = max - min
  local threshold = (self._thresholdFraction * range) + min

  self.output[ input:lt(threshold) ] = 0;
  self.output[ input:ge(threshold) ] = 1;
  return self.output

end

function Binarize:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(gradOutput)
    self.gradInput:copy(gradOutput)
   return self.gradInput
end
