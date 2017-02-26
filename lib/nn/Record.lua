
local Record, parent = torch.class('nn.Record', 'nn.Module')

function Record:__init(_spillPath, _spillAfter)
   parent.__init(self)
   self.isRecording = 0
   self.record = torch.Tensor()
   self.spillPath = _spillPath
   self.spillAfter = _spillAfter
   self.spillCount = 0
end

function Record:updateOutput(input)

  if self.isRecording > 0 then -- record input
    if self.record:nElement() > 0 then
      self.record = torch.cat(self.record, input, 1)
    else
      self.record = input:clone()
    end
    self:flush()
  end

   self.output = input
   return self.output
end


function Record:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end

function Record:enable()
   self.isRecording = 1
   return self
end
function Record:disable()
   self.isRecording = 0
   return self
end
function Record:flush()
   if self.spillPath ~= nil and self.record:nElement() >= self.spillAfter then
      torch.save(self.spillPath..string.format('%04d',self.spillCount),self.record)
      self.spillCount = self.spillCount + 1
      self.record = torch.Tensor()
   end
end

function Record:__tostring__()
  if self.isRecording > 0 then
    return torch.type(self) .. ' recording'
  end
  return torch.type(self)
end
