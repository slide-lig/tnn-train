
local model

if #(opt.arch.neuronPerLayerCount) <= 1 then Log.write("<ModelHandler.loadModel> WARNING: neuronPerLayerCount has only [".. #(opt.arch.neuronPerLayerCount) .."] entries") end

model = nn.Sequential()
-- assumes square kernels/pool_windows
local repeatConvChannel = opt.arch.convChannelRepeat or 1 --2
local outputChannelSize = opt.arch.convChannelSize  --{6,16}
local neuronPerLayerCount = opt.arch.neuronPerLayerCount --{120,84}
local kernelSize = opt.arch.convKernelSize
local poolSize = opt.arch.convPoolSize
local pad = opt.arch.convPadding or 0
local step = opt.arch.convStep or 1

if pad > 0 then pad = (kernelSize - 1)/2 else pad = nil end

local prepArchOutputSize = (opt.arch.prep and opt.arch.prep.convChannelSize) and opt.arch.prep.convChannelSize[#opt.arch.prep.convChannelSize] or nil
local inputChannelSize = prepArchOutputSize or ds:get('train','input','bchw'):size(2)
local convOutputSize = ds:get('train','input','bchw'):size(3)

model:add(nn.Convert(ds:ioShapes(), 'bchw'))

if opt.arch.dropout_in > 0 then
  model:add(nn.Dropout(opt.arch.dropout_in,1))
end

for i=1,#outputChannelSize do
  for j=1,repeatConvChannel do
    model:add(nn.SpatialConvolution(inputChannelSize, outputChannelSize[i], kernelSize, kernelSize, step, step, pad, pad)) -- 1 input image channel, 6 output channels, 5x5 convolution kernel

    inputChannelSize = outputChannelSize[i]

    if pad == nil then
      convOutputSize = (convOutputSize - kernelSize + 1) -- subtract kernelSize as a result of conv
    end

    if j==repeatConvChannel then
       model:add(nn.SpatialMaxPooling(poolSize,poolSize))
    end

    if opt.arch.batchNorm and opt.arch.batchNorm > 0 then
      model:add(nn.SpatialBatchNormalization(outputChannelSize[i], opt.arch.batchnorm.epsilon))
    end
    model:add(nn[opt.arch.activationFn]())                      -- non-linearity

    if opt.arch.dropout_conv > 0 and j~=repeatConvChannel then
       model:add(nn.Dropout(opt.arch.dropout_conv,1))
    end

    model:add(nn.StochasticFire(opt.arch.stochFireDuringTraining))
  end

  convOutputSize = convOutputSize / poolSize -- devide by poolSize for MaxPooling

end

local fullyConnectedInputHeight = outputChannelSize[#outputChannelSize]*convOutputSize*convOutputSize

model:add(nn.View(fullyConnectedInputHeight))  -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5

if opt.arch.dropout_hidden > 0 then
   model:add(nn.Dropout(opt.arch.dropout_hidden,1))
end


local inputHeight = fullyConnectedInputHeight
for i,hiddenSize in ipairs(opt.arch.neuronPerLayerCount) do
   model:add(nn.Linear(inputHeight, hiddenSize))

   if opt.arch.batchNorm > 0 then
      model:add(nn.BatchNormalization(hiddenSize, opt.arch.batchnorm.epsilon))
   end

   model:add(nn[opt.arch.activationFn]())
   if opt.arch.dropout_hidden > 0 then
      model:add(nn.Dropout(opt.arch.dropout_hidden,1))
   end
   model:add(nn.StochasticFire(opt.arch.stochFireDuringTraining))
   inputHeight = hiddenSize
end

model:add(nn.Linear(inputHeight, #(ds:classes())))   
if opt.arch.finalBN>0 then
   model:add(nn.BatchNormalization(#(ds:classes()), opt.arch.batchnorm.epsilon))
end
return model
