local RecordConvZero, parent = torch.class('nn.RecordConvZero', 'nn.Module')

function RecordConvZero:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH
   self.padW = padW or 0
   self.padH = padH or self.padW

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW):zero()
   self.biasHi = torch.Tensor(nOutputPlane):zero()
   self.biasLo = torch.Tensor(nOutputPlane):zero()
   self.bias  = torch.Tensor(nOutputPlane):zero()

   self:reset()


   self.zeroCount = 0
   self.totalCount = 0
   self.halfZeroCount = 0
   self.totalMuls = 0
end

function RecordConvZero:setWeight(weight)
  self.weight = torch.Tensor(weight:size()):copy(weight:eq(0))
end

function RecordConvZero:reset(stdv)
   self.weight:zero()
   self.biasHi:zero()
   self.biasLo:zero()
   self.bias:zero()
end

local function backCompatibility(self)
   self.finput = self.finput or self.weight.new()
   self.fgradInput = self.fgradInput or self.weight.new()
   if self.padding then
      self.padW = self.padding
      self.padH = self.padding
      self.padding = nil
   else
      self.padW = self.padW or 0
      self.padH = self.padH or 0
   end
   if self.weight:dim() == 2 then
      self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
   if self.gradWeight and self.gradWeight:dim() == 2 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
end

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
	 self._gradOutput = self._gradOutput or gradOutput.new()
	 self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
	 gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

-- function to re-view the weight layout in a way that would make the MM ops happy
local function viewWeight(self)
   self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
   if self.gradWeight and self.gradWeight:dim() > 0 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
   end
end

local function unviewWeight(self)
   self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   if self.gradWeight and self.gradWeight:dim() > 0 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
end

function RecordConvZero:updateOutput(input)

  self.zeroCount = self.zeroCount + torch.sum(input:eq(0))
  self.totalCount = self.totalCount + input:nElement()

  inputMask = input.new():resize(input:size()):copy(input:ne(0))
  outputDummy = input.new()


   backCompatibility(self)
   viewWeight(self)
   inputMask = makeContiguous(self, inputMask)
   inputMask.THNN.SpatialConvolutionMM_updateOutput(
      inputMask:cdata(),
      outputDummy:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH
   )
   unviewWeight(self)


   self.halfZeroCount = self.halfZeroCount + torch.sum(outputDummy)

   if self.totalMuls == nil then
     self.totalMuls = 0
   end

   self.totalMuls = self.totalMuls +  self.kW*self.kH*self.nInputPlane*outputDummy:nElement()

   self.output = input

   return self.output
end

function RecordConvZero:updateGradInput(input, gradOutput)
   if self.gradInput then
      self.gradInput:resizeAs(input):zero()
      return self.gradInput
   end
end

function RecordConvZero:updateParameters(learningRate)
end

function RecordConvZero:type(type,tensorCache)
   self.finput = self.finput and torch.Tensor()
   self.fgradInput = self.fgradInput and torch.Tensor()
   return parent.type(self,type,tensorCache)
end

function RecordConvZero:__tostring__()

  local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
   return s .. ')'
end

function RecordConvZero:clearState()
   nn.utils.clear(self, 'finput', 'fgradInput', '_input', '_gradOutput')
   return parent.clearState(self)
end
