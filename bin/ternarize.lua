Ternarizer = {}
Ternarizer.Complete = false
package.path = '../lib/?;../lib/?.lua;../bin/?.lua;'..package.path

-- core
require 'Runtime.Coredp'
-- utils
require 'utils.ConfHandler'

package.path = os.getenv("LUA_PATH")
require 'dp'
package.path = '../lib/?;../lib/?.lua;'..package.path

nninit = require 'nninit.nninit'
require 'dp.observer.resultlogger'
require 'dp.observer.networksaver'
require 'dp.observer.epochlogger'
require 'dp.preprocess.scale'
require 'dp.transform.transformmatch'
require 'dp.transform.removematch'
require 'dp.transform.insertatmatch'
require 'dp.preprocess.narize'
require 'dp.sampler.ligsampler'
require 'dp.data.gtsrb'
require 'dp.data.custom'
require 'dp.data.svhnstd'
require 'dp.data.cacheloader'
require 'nn.Distort'
require 'nn.Narize'
require 'nn.Record'
require 'nn.StochasticFire'
require 'nn.SqrHingeEmbeddingCriterion'
require 'nn.OneHotEncode'
require 'nn.TernaryConnectLinear'
require 'nn.TernaryConnectSpatialConvolution'
require 'nn.NegInversion'
require 'nn.TernaryLinear'
require 'nn.TernSpatialConvolution'



function Ternarizer.getModulesForTernarization(module)

    print("Ternarizing:")
    print(module)

    local nonTernMods, nonTernContainers = module:findModules('nn.TernaryConnectSpatialConvolution')
    local linMods, linContainers = module:findModules('nn.TernaryConnectLinear')

    for i=1,#linMods do
      table.insert(nonTernMods, linMods[i])
    end
    for i=1,#linContainers do
      table.insert(nonTernContainers, linContainers[i])
    end

    return nonTernMods, nonTernContainers

end

function Ternarizer.getLayerTypesForFirstTernarization(nonTernMods, nonTernContainers)
  local layers_to_replace = {}
  local found = false
  local working_from = 0
  for i = 1, #(nonTernContainers[1].modules) do
    if found then
      --table.insert(layers_to_replace, nonTernContainers[1].modules[i]:__tostring())
      table.insert(layers_to_replace, torch.type(nonTernContainers[1].modules[i]))
    end
    if nonTernContainers[1].modules[i] == nonTernMods[1] then
      found = true -- from here we will note the layers to include into calcs
      working_from = i
    elseif nonTernContainers[1].modules[i] == nonTernMods[2] then
      table.remove(layers_to_replace) -- remove this following layers
      found = false
    end
  end
  return layers_to_replace, working_from
end

function Ternarizer.ternarizeModel(conf, module)

  while #(Ternarizer.getModulesForTernarization(module)) > 0 and not Ternarizer.Complete do

    print("<Ternarize> converting layer to ternary layer")
    module = Ternarizer.replaceFirstNonTernaryLayer(module)

    -- if not at last layer
    if #(Ternarizer.getModulesForTernarization(module)) > 0 then
      -- remove BN, Non-Linear, Dropout & Stochastic
      module = (dp.RemoveMatch{contains={'nn.BatchNormalization','nn.SpatialBatchNormalization','nn.'..conf.arch.activationFn,'cudnn.'..conf.arch.activationFn,'nn.Dropout'}, match_count=3}):apply(module)
      module = (dp.RemoveMatch{contains={'nn.StochasticFire'}, match_count=1}):apply(module)
    end
  end

  return module

end

function Ternarizer.replaceFirstNonTernaryLayer(module)

    local nonTernMods, nonTernContainers = Ternarizer.getModulesForTernarization(module)

    local threshold_hi, threshold_lo, inversion = Ternarizer.calcTernaryLayerThresholds(module)

    if #nonTernMods > 0 then
      for i=1, #nonTernContainers do
        for j = 1, #(nonTernContainers[i].modules) do
          if nonTernContainers[i].modules[j] == nonTernMods[1] then

            -- switch to ternary weights
            nonTernMods[1].weight = nonTernMods[1].ternWeight

            if #nonTernMods > 1 then

              print('<Ternarize> creating ternary layer')
              local replacementLayer = nn.TernaryLinear(nonTernMods[1].weight:size(2),nonTernMods[1].weight:size(1),threshold_hi, threshold_lo)
              if (#(module:findModules('nn.TernaryConnectSpatialConvolution'))) > 0 then
                replacementLayer = nn.TernSpatialConvolution(nonTernMods[1].nInputPlane, nonTernMods[1].nOutputPlane, nonTernMods[1].kW, nonTernMods[1].kH, nonTernMods[1].dW, nonTernMods[1].dH, nonTernMods[1].padW, nonTernMods[1].padH)
                replacementLayer.biasHi = threshold_hi
                replacementLayer.biasLo = threshold_lo
              end
              replacementLayer.inversion = inversion

              nonTernContainers[i].modules[j] = replacementLayer
            else
              nonTernContainers[i].modules[j] = nn.Linear(nonTernMods[1].weight:size(2),nonTernMods[1].weight:size(1))
              nonTernContainers[i].modules[j].bias:copy(nonTernMods[1].bias)
              Ternarizer.Complete = true
            end

            -- source weights are ternary
            nonTernContainers[i].modules[j].weight = nonTernMods[1].weight

            print('<Ternarize> finished replacing layer with ternary layer')
          end
        end
      end
    end

    return module

end

function Ternarizer.handleSpecialMaxPoolBeforeBatchNormInsert(module, layers_to_replace, working_from)
  local bnIx = tablex.find(layers_to_replace, 'nn.SpatialBatchNormalization') or 0
  local mpIx = tablex.find_if(layers_to_replace,function(v) return v:find('SpatialMaxPooling') end) or bnIx
  if bnIx > mpIx then
    -- need to insert NegInversion modules
    print("Inserting NegInversion layers since MaxPool exists before BatchNorm")
    if bnIx > 0 then print("Found "..layers_to_replace[bnIx].." at: "..bnIx) end
    if mpIx > 0 then print("Found "..layers_to_replace[mpIx].." at: "..mpIx) end
    local mpBlock = tablex.index_by(layers_to_replace,tablex.range(mpIx,#layers_to_replace))
    module = (dp.InsertAtMatch{start_at=working_from, contains=mpBlock, match_count=#mpBlock, add_module=nn.NegInversion(inversion)}):apply(module)
    module = (dp.InsertAtMatch{start_at=working_from, begins_with={'nn.NegInversion'}, contains={layers_to_replace[mpIx]}, match_count=2, insert_after=true,add_module=nn.NegInversion(inversion)}):apply(module)
  end
  return module
end

function Ternarizer.calcTernaryLayerThresholds(module)

  local nonTernMods, nonTernContainers = Ternarizer.getModulesForTernarization(module)
  assert(#nonTernMods > 0, "<Ternarize> Failed to find SpatialConvolution/Linear layers")

  local layers_to_replace, working_from = Ternarizer.getLayerTypesForFirstTernarization(nonTernMods, nonTernContainers)
  assert(#nonTernMods > 0, "<Ternarize> Failed to find layers to calculate thresholds")
  print(layers_to_replace)

  -- special case: final layer gets original bias
  if tablex.find(layers_to_replace, 'nn.StochasticFire') == nil then
    return torch.Tensor(nonTernMods[1].bias:nElement()):copy(nonTernMods[1].bias)
  end

  local tl, inversion = reverse(module, layers_to_replace, -0.5)
  local th, inversion = reverse(module, layers_to_replace, 0.5)

  -- special case: if MaxPool precedes BatchNormalization, where BN weight < 0, we invert before and after MaxPool
  module = Ternarizer.handleSpecialMaxPoolBeforeBatchNormInsert(module, layers_to_replace, working_from)

  tl = tl - nonTernMods[1].bias
  th = th - nonTernMods[1].bias
  tl[inversion] = tl[inversion] + torch.mul(nonTernMods[1].bias,2)[inversion]
  th[inversion] = th[inversion] + torch.mul(nonTernMods[1].bias,2)[inversion]

  -- discretize thresholds, although not required
  tl:ceil()
  th:floor()
  -- converting to default tensor type
  local threshold_hi = torch.Tensor(th:nElement()):copy(th)
  local threshold_lo = torch.Tensor(tl:nElement()):copy(tl)

  return threshold_hi, threshold_lo, inversion

end

function reverse(module, layers_to_replace, x)

  local bnMods, bnContainers = module:findModules('nn.SpatialBatchNormalization')
  if #bnMods <= 0 then
    bnMods, bnContainers = module:findModules('nn.BatchNormalization')
  end

  if tablex.find_if(layers_to_replace,function(v) return tablex.find({'nn.BatchNormalization','nn.SpatialBatchNormalization'},v)~= nil end) ~= nil then
    assert(#bnMods > 0, "<Ternarize> Failed to find any BatchNormalization layers")
  end

  local reverseFns = {['nn.Tanh']={},['cudnn.Tanh']={},['nn.HardTanh']={},['nn.BatchNormalization']={},['nn.SpatialBatchNormalization']={}}
  reverseFns['nn.Tanh']['fn'] = 'reverseTanh'
  reverseFns['cudnn.Tanh']['fn'] = 'reverseTanh'
  reverseFns['nn.HardTanh']['fn'] = 'reverseHardTanh'
  reverseFns['nn.BatchNormalization']['fn'] = 'reverseBN'
  reverseFns['nn.SpatialBatchNormalization']['fn'] = 'reverseBN'

  reverseFns['nn.Tanh']['args'] = nil
  reverseFns['cudnn.Tanh']['args'] = nil
  reverseFns['nn.HardTanh']['args'] = nil
  reverseFns['nn.BatchNormalization']['args'] = {bnMods[1].weight, bnMods[1].bias, bnMods[1].running_mean, bnMods[1].running_var, bnMods[1].eps}
  reverseFns['nn.SpatialBatchNormalization']['args'] = {bnMods[1].weight, bnMods[1].bias, bnMods[1].running_mean, bnMods[1].running_var, bnMods[1].eps}

  for i = #layers_to_replace,1,-1 do
    local layer_str = layers_to_replace[i]
    if reverseFns[layer_str] then
      local rev_fn_str = reverseFns[layer_str]['fn']
      local args = reverseFns[layer_str]['args']
      print("Reversing through: "..layer_str.." using "..rev_fn_str)
      if args~=nil then
        x, aux = Ternarizer[rev_fn_str](x, unpack(args))
      else
        x, aux = Ternarizer[rev_fn_str](x)
      end
    else
      print("Skipping: "..layer_str)
    end
  end
  return x,aux
end

function Ternarizer.reverseBN(x, weight, bias, mean, var, eps)
  --
  --                     √ (var + eps)
  -- y = ( x - bias ) * --------------- + mean
  --                         weight
  --
  local v = torch.sqrt(var + eps)
  local a = torch.cdiv(weight,v)
  local result = torch.add( torch.cdiv(torch.add(torch.mul(bias, -1), x), a), mean)

  -- invert thresholds along with weights of preceding layer
  local inversion = a:le(0)
  result[inversion] = result[inversion]:mul(-1)

  return result, inversion
end

function Ternarizer.reverseTanh(x)
  if x==0.5 then return 0.549306 end
  if x==-0.5 then return -0.549306 end
  error('<Ternarizer> reverseTanh: Only (+/-) 0.5 is supported! (I know!!)')
end

function Ternarizer.reverseHardTanh(x)
  if x>1 then return 1 end
  if x<-1 then return -1 end
  return x
end



-- Main program


torch.setdefaulttensortype('torch.DoubleTensor')


    require 'cutorch'
    require 'cunn'
    require 'cudnn'

  local model
  model = torch.load('/home/slide/alemdar/workspace/BinaryNet/Results/SVHN/Net')
  print(model)

  --
  -- ternarize
  --
  if torch.type(model.modules[1]) == 'nn.Sequential' then -- preprocess_model is exists
    model.modules[#model.modules] = Ternarizer.ternarizeModel(opt, model.modules[#model.modules])
  else
    model = Ternarizer.ternarizeModel(opt, model)
  end

  --
  -- save
  --
  local ternaryNetPath ='ternarized.net'
  DataHandler.saveModule(ternaryNetPath,model,opt)
  print("<Ternarize> progressed net saved to file [".. ternaryNetPath .."]")

--end
