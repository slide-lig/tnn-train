local OneHotEncode, parent = torch.class('nn.OneHotEncode', 'nn.Module')

function OneHotEncode:__init(classCount,signed)
   parent.__init(self)
   self.classCount = classCount or 10
   self.signed = signed or 0
end

function OneHotEncode:updateOutput(input)

  self.output = torch.zeros(input:size(1),self.classCount)
  self.output = self.output:scatter(2, input:long():view(-1,1), 1)
  if self.signed > 0 then
    self.output = self.output:mul(2):float():add(-1)
  end

  return self.output
end

function OneHotEncode:__tostring__()
  return torch.type(self) .. "(" .. self.classCount .. "," .. self.signed .. ")"
end
