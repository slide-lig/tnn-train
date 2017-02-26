local TernaryConnectSpatialConvolution, Parent = torch.class('nn.TernaryConnectSpatialConvolution', 'nn.SpatialConvolution')

function TernaryConnectSpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, stochFlag)
  Parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  -- ternary parameters for propagation
  self.rvWeight = self.weight -- pointer to real-valued weights by default
  self.ternWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
  self.stochFlag = stochFlag or 0
  self.randmat = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
  self:reset()
end


function TernaryConnectSpatialConvolution:weight_ternarize(terndata, rvdata, stochFlag)
  terndata:copy(rvdata)
  if stochFlag > 0 then
    terndata:add(self.randmat:rand(self.randmat:size())) -- add randomly selected [0,1) range to the weight range of [-1,1]
    terndata:floor() -- take the floor to get {-1, 0, 1}
  else
    terndata:round()
  end
  return terndata
end

function TernaryConnectSpatialConvolution:updateOutput(input)

  if self.train then
    self.weight:clamp(-1,1) -- clip weights
    self.ternWeight = self:weight_ternarize(self.ternWeight, self.rvWeight, self.stochFlag)
  end
  self.weight = self.ternWeight -- switch to ternary weights
  self.output = Parent.updateOutput(self, input)
  self.weight = self.rvWeight

  return self.output
end

function TernaryConnectSpatialConvolution:updateGradInput(input, gradOutput)
  if self.gradInput then
    self.weight = self.ternWeight -- switch to ternary weights
    self.gradInput = Parent.updateGradInput(self, input, gradOutput)
    self.weight = self.rvWeight
    return self.gradInput
  end
end

function TernaryConnectSpatialConvolution:updateParameters(learningRate)
  Parent.updateParameters(self, learningRate) -- update real-valued parameters
end
