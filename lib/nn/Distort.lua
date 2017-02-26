local Distort, parent = torch.class('nn.Distort', 'nn.Module')

function Distort:__init(limits, trainOnly)
   parent.__init(self)
   self._limits = limits
   self._trainOnly = trainOnly or false
end

function Distort:updateOutput(input)

  self.output = input:clone()

  if not self._trainOnly or self.train then
    self.output = TensorHandler.randScale(self.output:type(torch.Tensor():type()), self._limits.scale)
    self.output = TensorHandler.randRotate(self.output:type(torch.Tensor():type()), self._limits.rotate)
    self.output = TensorHandler.randTranslate(self.output:type(torch.Tensor():type()), self._limits.translate)
  end

  self.output = self.output:type(input:type())
  
  return self.output
end


function Distort:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end
