torch = require 'torch'
image = require 'image'
TensorHandler = {}

function TensorHandler.edgeEvenZeroPad(tens,d)

  local dx1 = math.floor(d/2)
  local dx2 = d - dx1

  local temp1 = TensorHandler.zeroPad(tens, dx1, dx1)

  return TensorHandler.zeroPad(temp1, -dx2, -dx2)
end
function TensorHandler.zeroPad(tens, x, y)
  -- positive x => additional columns prepended to tensor
  -- positive y => additional rows prepended to tensor

  local padded = torch.Tensor(tens:size(1) + math.abs(x), tens:size(2) + math.abs(y)):fill(0)

  if x>0 then x1 = x+1 x2 = padded:size(1) else x1 = 1 x2 = padded:size(1) + x end
  if y>0 then y1 = y+1 y2 = padded:size(2) else y1 = 1 y2 = padded:size(2) + y end

  padded[{{x1,x2},{y1,y2}}] = tens

  return padded
end

function TensorHandler.randDistort(tens, lim)
  local addTens = torch.Tensor(tens:size()):copy(tens)
  addTens = TensorHandler.randScale(addTens, lim.scale)
  addTens = TensorHandler.randRotate(addTens, lim.rotate)
  addTens = TensorHandler.randTranslate(addTens, lim.translate)
  return addTens
end

-- lim is pixel/pos count
function TensorHandler.randTranslate(tens, lim)
  if lim == 0 then return tens end
  local t = torch.Tensor(tens)

  local dx = 0
  local dy = 0

  -- translate t by dx,dy with zero-padding

  --Log.write("<TensorHandler.randTranslate> translating tensor by [".. dx ..",".. dy .."]")
  if t:dim() == 4 then
    for i=1,t:size(1) do
      for j=1,t:size(2) do

        dx = math.random(-lim,lim)
        dy = math.random(-lim,lim)
        --Log.write("<TensorHandler.randTranslate> translating tensor by [".. dx ..",".. dy .."]")
        t[{{i},{j}}] = image.translate(t[{{i},{j}}]:squeeze(), dx, dy)
      end
    end
  else
    Log.write("<TensorHandler.randTranslate> failed to handle unexpected tensor size")
    error("<TensorHandler.randTranslate> panic")
  end

  return t
end

-- lim is radial degree
function TensorHandler.randRotate(tens, lim)

  if lim == 0 then return tens end
  local t = torch.Tensor(tens)

  local dd = 0


  if t:dim() == 4 then
      for i=1,t:size(1) do
        for j=1,t:size(2) do
          dd = math.random(-lim,lim)
          --if i == t:size(1) then Log.write("<TensorHandler.randRotate> rotating tensor by [".. dd .."°]") end
          t[{{i},{j}}] = image.rotate(t[{{i},{j}}]:squeeze(), math.rad(dd), 'simple')
        end
      end
  else
    Log.write("<TensorHandler.randRotate> failed to handle unexpected tensor size")
    error("<TensorHandler.randRotate> panic")
  end
  return t
end

-- lim is percentage of original to increase/decrease
function TensorHandler.randScale(tens, lim)
  if lim == 0 then return tens end
  local t = torch.Tensor(tens:size()):copy(tens)

  local dz = 0

  if t:dim() == 4 then
    for i=1,t:size(1) do
      for j=1,t:size(2) do

        dz = math.random(-lim,lim) / 100
  --      if i == t:size(1) then Log.write("<TensorHandler.randScale> scaling tensor by [".. dz*100 .."%]") end
        local temp = image.scale(t[{{i},{j}}]:squeeze(), math.floor(t:size(3)*(1+dz)), math.floor(t:size(4)*(1+dz)), 'simple')

        if(dz > 0) then
          t[{{i},{j}}] = image.crop(temp, 'c', t:size(3), t:size(4))
        else
          t[{{i},{j}}] = TensorHandler.edgeEvenZeroPad(temp:squeeze(), t:size(3) - temp:size(3))
        end

      end
    end
  else
    Log.write("<TensorHandler.randScale> failed to handle unexpected tensor size")
    error("<TensorHandler.randScale> panic")
  end

  return t
end

-- for each vector in tens2, perform mapVector
function TensorHandler.mapAsTensors(tens1, tens2, func)

  local t =  tens2:chunk(tens2:size(1),1)
  local result = torch.Tensor(tens1:size(1),tens2:size(1))

  for i,v in ipairs(t) do
    result[{ {},i }] = TensorHandler.mapVector(tens1, v, func)
  end

  return result:t()
end
-- perform func on each vector in tens with vec
function TensorHandler.mapVector(tens, vec, func)
  -- TODO dimensionality checking
  local t =  tens:chunk(tens:size(1),1)
  local result = torch.Tensor(tens:size(1))
  for i,v in ipairs(t) do
    result[i] = func(v,vec)
  end
  return result
end

-- perform xnor on vectors, and count resulting bits
function TensorHandler.vecEqSum(vec1,vec2)
    return torch.eq(vec1,vec2):sum()
end
