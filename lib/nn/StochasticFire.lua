local StochasticFire, Parent = torch.class('nn.StochasticFire', 'nn.Module')

function StochasticFire:__init(enable, binaryOutputFlag)
   Parent.__init(self)
   self.inplace = false
   self.enable = enable or 1
   self.binaryOutputFlag = binaryOutputFlag or 0
   self.randmat = torch.Tensor()
end

function StochasticFire:updateOutput(input)
   if self.inplace then
      self.output = input
   else
      self.output:resizeAs(input):copy(input)
   end

   self.output:clamp(-1,1)
   if self.train == false or self.enable == 0 then
    if ( self.binaryOutputFlag == 1 ) then
     self.output:sign()
    else
     self.output:round()
    end
   else
    if ( self.binaryOutputFlag == 1 ) then
     self.output:add(1):div(2):add(-self.randmat:resizeAs(self.output):rand(self.randmat:size())) -- subtract randomly selected [0,1) range from the weight range of [0,1]
     self.output:sign() -- take sign to get {-1, 1}
    else
     self.output:add(self.randmat:resizeAs(self.output):rand(self.randmat:size())) -- add randomly selected [0,1) range to the weight range of [-1,1]
     self.output:floor() -- take the floor to get {-1, 0, 1}
    end
   end
   return self.output
end

function StochasticFire:updateGradInput(input, gradOutput)
   if self.train then
      if self.inplace then
         self.gradInput = gradOutput
      else
         self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      end
   else
      error('backprop only defined while training')
   end
   return self.gradInput
end


function StochasticFire:__tostring__()
  if self.enable > 0 then
    return torch.type(self)
  end
  return torch.type(self) .. ' disabled'
end
