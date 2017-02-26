--This function is written by Itay Hubara and distributed as part of the following publication
--
--"Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1",
--Matthieu Courbariaux, Itay Hubara, Daniel Soudry, Ran El-Yaniv, Yoshua Bengio'
--
--The original version of the file is found in
--https://github.com/itayhubara/BinaryNet repository
--We provide this function here convenience reasons.

--[[
This Function implement the squared hinge loss criterion
]]
local SqrHingeEmbeddingCriterion, parent = torch.class('nn.SqrHingeEmbeddingCriterion', 'nn.Criterion')

function SqrHingeEmbeddingCriterion:__init(margin)
   parent.__init(self)
   self.margin = margin or 1
   self.sizeAverage = true
end

function SqrHingeEmbeddingCriterion:updateOutput(input,y)
   self.buffer = self.buffer or input.new()
   if not torch.isTensor(y) then
      self.ty = self.ty or input.new():resize(1)
      self.ty[1]=y
      y=self.ty
   end

   self.buffer:resizeAs(input):copy(input)
   self.buffer:cmul(y):mul(-1):add(self.margin)
   self.buffer[torch.le(self.buffer ,0)]=0
   self.output=self.buffer:clone():pow(2):sum()

   if (self.sizeAverage == nil or self.sizeAverage == true) then
      self.output = self.output / input:nElement()
   end

   return self.output
end

function SqrHingeEmbeddingCriterion:updateGradInput(input, y)
   if not torch.isTensor(y) then self.ty[1]=y; y=self.ty end
   self.gradInput:resizeAs(input):copy(y):mul(-2):cmul(self.buffer)
   self.gradInput[torch.cmul(y,input):gt(self.margin)] = 0
   if (self.sizeAverage == nil or self.sizeAverage == true) then
      self.gradInput:mul(1 / input:nElement())
   end
   return self.gradInput
end
