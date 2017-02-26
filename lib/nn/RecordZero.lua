
local RecordZero, parent = torch.class('nn.RecordZero', 'nn.Module')

function RecordZero:__init(_spillPath, _spillAfter)
   parent.__init(self)
   self.isRecording = 0
   self.RecordZero = torch.Tensor()
   self.spillPath = _spillPath
   self.spillAfter = _spillAfter
   self.spillCount = 0

   self.zeroCount = 0
   self.totalCount = 0

   self.halfZeroCount = 0
   self.totalMuls = 0
end

function RecordZero:setWeight(weight)
  self.weight = torch.Tensor(weight:size()):copy(weight:eq(0))
end

function RecordZero:updateOutput(input)

  if self.isRecording > 0 then -- RecordZero input
    if self.RecordZero:nElement() > 0 then
      self.RecordZero = torch.cat(self.RecordZero, input, 1)
    else
      self.RecordZero = input:clone()
    end
    self:flush()
  end

  self.zeroCount = self.zeroCount + torch.sum(input:eq(0))
  self.totalCount = self.totalCount + input:nElement()

  if self.weight ~= nil then

    if self.totalMuls == nil then
      self.totalMuls = 0
    end

    inputMask = input.new():resize(input:size()):copy(input:ne(0))
    self.halfZeroCount = self.halfZeroCount + torch.sum(inputMask * self.weight:t())
    self.totalMuls = self.totalMuls +  inputMask:size(1)*self.weight:nElement()
  end

   self.output = input
   return self.output
end


function RecordZero:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end

function RecordZero:enable()
   self.isRecording = 1
   self.zeroCount = 0
   self.totalCount = 0
   self.halfZeroCount = 0
   return self
end
function RecordZero:disable()
   self.isRecording = 0
   return self
end
function RecordZero:flush()
   if self.spillPath ~= nil and self.RecordZero:nElement() >= self.spillAfter then
      torch.save(self.spillPath..string.format('%04d',self.spillCount),self.RecordZero)
      self.spillCount = self.spillCount + 1
      self.RecordZero = torch.Tensor()
   end
end

function RecordZero:__tostring__()
  if self.isRecording > 0 then
    return torch.type(self) .. ' recording'
  end
  return torch.type(self)
end
