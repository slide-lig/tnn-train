torch = require 'torch'

CalcHandler = {}

function CalcHandler.getLearningRate(lr, lrd, t, method)
  if type(method) ~= 'string' then error("<CalcHandler.getLearningRate> error: bad method to apply learning rate") return nil end
  -- step decay where lrd has the form <multiplier>/<stepsize>
  if method:upper() == 'STEP' then
    local stepMultiplier = tonumber(string.match(lrd,'^%d*%.*%d+')) or nil
    local stepDistance = tonumber(string.match(lrd,'%d*%.*%d+$')) or nil
    if t%stepDistance ~= 0 then stepMultiplier = 1 end -- no change
    return lr * stepMultiplier
  end
  -- exponential decay
  if method:upper() == 'EXP' then return lr*math.exp(-t*lrd) end
  -- default to 1/t decay (annealing)
  return lr / (1 + t*lrd )
end

function CalcHandler.evalLearningRate(lr, lrd, epoch_num, batch_num)
  if type(lrd) == 'string' then --formula
    local tmp_lrd_formula = string.format('%s %s','return',string.gsub(string.gsub(string.gsub(string.gsub(lrd,'epoch',epoch_num),'batch',batch_num),'lr',lr),'[\'\"]',''))
    return loadstring(tmp_lrd_formula)()
  elseif type(lrd) == 'table' then --schedule
    return lrd[epoch_num] or lr
  else -- use learningRateDecay as a number with 1/t decay formula
    return lr / (1 + batch_num*lrd)
  end
end
