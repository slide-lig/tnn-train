
local model

if #(opt.arch.neuronPerLayerCount) <= 1 then Log.write("<ModelHandler.loadModel> WARNING: neuronPerLayerCount has only [".. #(opt.arch.neuronPerLayerCount) .."] entries") end

model = nn.Sequential()

local prepArchOutputSize = (opt.arch.prep and opt.arch.prep.convChannelSize) and opt.arch.prep.convChannelSize[#opt.arch.prep.convChannelSize] or nil
-- assumes original input has only 1 channel like mnist
local inputChannelSize = prepArchOutputSize and prepArchOutputSize*ds:featureSize() or ds:featureSize()

model:add(nn.Convert(ds:ioShapes(), 'bf'))

if opt.arch.dropout_in > 0 then
  model:add(nn.Dropout(opt.arch.dropout_in, 1))
end


local inputHeight = inputChannelSize
for i,hiddenSize in ipairs(opt.arch.neuronPerLayerCount) do
   model:add(nn.Linear(inputHeight, hiddenSize))

   if opt.arch.batchNorm > 0 then
      model:add(nn.BatchNormalization(hiddenSize, opt.arch.batchnorm.epsilon))
   end

   model:add(nn[opt.arch.activationFn]())

   if opt.arch.dropout_hidden > 0 then
      model:add(nn.Dropout(opt.arch.dropout_hidden, 1))
   end
   model:add(nn.StochasticFire(opt.arch.stochFireDuringTraining))
   inputHeight = hiddenSize
end
-- output layer
model:add(nn.Linear(inputHeight, #(ds:classes())))

return model
