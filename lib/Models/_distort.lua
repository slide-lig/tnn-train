
local distort_model = nil

if ConfHandler.IsOnlineDistortEn(opt) then
  distort_model = nn.Sequential()
  distort_model:add(nn.Convert(ds:ioShapes(), 'bchw'))
  distort_model:add(nn.Distort(opt.input.distort, true)) -- true flag restricts distortions to training
  if opt.input.narize.bitCount > 0 then
    distort_model:add(nn.Narize(opt.input.narize.bitCount, opt.input.narize.signed))
  end

end

return distort_model
