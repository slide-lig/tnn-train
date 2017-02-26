
local model = nn.Sequential()
-- assumes square kernels/pool_windows
local outputChannelSize = opt.arch.prep.convChannelSize
local kernelSize = opt.arch.prep.convKernelSize or 3
local pad = opt.arch.prep.convPadding or 0
local step = opt.arch.prep.convStep or 1

if pad > 0 then pad = (kernelSize - 1)/2 else pad = nil end

local inputChannelSize = ds:get('train','input','bchw'):size(2)
local convOutputSize = ds:get('train','input','bchw'):size(3)

if not ConfHandler.IsOnlineDistortEn(opt) then
  model:add(nn.Convert(ds:ioShapes(), 'bchw'))
end

for i=1,#outputChannelSize do
    model:add(nn.SpatialConvolution(inputChannelSize, outputChannelSize[i], kernelSize, kernelSize, step, step, pad, pad)) -- 1 input image channel, 6 output channels, 5x5 convolution kernel

    inputChannelSize = outputChannelSize[i]

    if pad == nil then
      convOutputSize = (convOutputSize - kernelSize + 1) -- subtract kernelSize as a result of conv
    end

    if opt.arch.prep.batchNorm and opt.arch.prep.batchNorm > 0 then
      model:add(nn.SpatialBatchNormalization(outputChannelSize[i], opt.arch.batchnorm.epsilon))
    end
    model:add(nn[opt.arch.activationFn]())                      -- non-linearity

    model:add(nn.StochasticFire(opt.arch.prep.stcFlag>0, opt.arch.prep.outputTernary==0))

end

return model
