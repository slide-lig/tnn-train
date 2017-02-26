
local NegInversion, parent = torch.class('nn.NegInversion', 'nn.Module')

function NegInversion:__init(inversion)
   parent.__init(self)
   self.inversion = inversion or nil
end

function NegInversion:updateOutput(input)
  self.output = input
  --print(self.inversion:nElement().."->"..self.output:size(2))
  if self.inversion ~= nil and self.inversion:nElement() == self.output:size(2) then
    for j=1,self.output:size(2) do
      self.output[{{},j}] = (self.inversion[j] > 0) and torch.mul(self.output[{{},j}],-1) or self.output[{{},j}] -- inefficient
    end
  end
  return self.output
end


function NegInversion:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end

function NegInversion:__tostring__()
  return torch.type(self)
end
