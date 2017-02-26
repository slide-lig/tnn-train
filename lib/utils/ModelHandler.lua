--require 'torch'
--require 'nn'
require 'utils.DataHandler'

ModelHandler = {}

function ModelHandler.configureTorch(opt)

  torch.setnumthreads(opt.main.threads)

  --local seed = tonumber(tostring(os.time()):reverse():sub(1,6))
  local seed = tonumber(opt.run.randseed) or 1
  torch.manualSeed(seed)

  math.randomseed( seed )
  math.random(); math.random(); math.random()

  Log.write("<ModelHandler.configureTorch> using default built-in torch seed generator, giving [".. torch.initialSeed() .."]")
  Log.write("<ModelHandler.configureTorch> using os.time() to seed math lib generator, giving [".. seed .."]")

  torch.setdefaulttensortype('torch.FloatTensor')

  if opt.main.cuda > 0 then
    require 'cutorch'
    require 'cunn'
    if opt.main.cudnn ~= nil and opt.main.cudnn > 0 then
      require 'cudnn'
    end
    cutorch.setDevice(opt.main.cuda)
    cutorch.manualSeedAll(tonumber(opt.run.randseed) or 1)
    Log.write("<ModelHandler.configureTorch> including CUDA modules")
  end

end

function ModelHandler.getModelArchLongName(shortname)

    if shortname:upper() == 'MLP' then return 'MLP' end
    if shortname:upper() == 'PROGRESSIVEBINARIZATION' then return 'ProgressiveBinarization' end
    if shortname:upper() == 'CONV' then return 'CONV' end

    return shortname
end

function ModelHandler.getTrainer(opt)

  local trainer

  if opt.arch.modelArch == 'MLP' then trainer = BatchTrainer end
  if opt.arch.modelArch == 'ProgressiveBinarization' then trainer = BatchTrainer end
  if opt.arch.modelArch == 'CONV' then trainer = BatchTrainer end

  if trainer ~= nil and opt.main.verbose > 0 then Log.write("<ModelHandler.getTrainer> trainer type: "..trainer._classname) end

  return trainer

end

function ModelHandler.getModel(ds, opt)

  local model = nil

  if opt.main.loadNet == nil or opt.main.loadNet == "" then

    Log.write("<ModelHandler.getModel> generating model ["..opt.arch.modelArch.."]")
    --model = ModelHandler.generateModel(opt.arch.modelArch, opt.main._modelInputDims, opt.main._modelOutputHeight, opt.arch.neuronPerLayerCount, opt)
    model = ModelHandler.generateModel(ds, opt)
    opt.main._parameters, opt.main._gradParameters = model:getParameters()

    if opt.main.loadParams ~= nil and opt.main.loadParams ~= "" then
      if opt.main.verbose > 0 then
        Log.write("<ModelHandler.getModel> loading model parameters [".. opt.main.loadParams .."]")
      end
      local loadModel = DataHandler.loadModule(opt.main.loadParams)
      p,g = loadModel:getParameters()
      opt.main._parameters:copy(p)
      model:updateParameters(0)
    end

  else

    Log.write("<ModelHandler.getModel> loading network ["..opt.main.loadNet.."]")
    model = DataHandler.loadModule(opt.main.loadNet)
    if opt.main.cuda > 0 then
      model:cuda()
    end
    opt.main._parameters, opt.main._gradParameters = model:getParameters()

  end

  assert(model, "<ModelHandler.getModel> failed to get model")

  if opt.main.cuda > 0 then
    if opt.main.cudnn ~= nil and opt.main.cudnn > 0 then
      cudnn.convert(model, cudnn)
    end
    model:cuda()
  end

  Log.write("<ModelHandler.getModel> model: "..tostring(model))
  return model
end

function ModelHandler.generateModel(ds, opt)

  local tort_model = opt.input.distort  and ModelHandler.loadModelFile('_distort')
  local prep_model = opt.arch.prep      and ModelHandler.loadModelFile(opt.arch.prep.modelArch)
  local arch_model = opt.arch.modelArch and ModelHandler.loadModelFile(opt.arch.modelArch)

  arch_model = ModelHandler.initializeTracing(arch_model, opt)

  local complete_model = nn.Sequential()

  result = (type(tort_model)=='table') and complete_model:add(tort_model)
  result = (type(prep_model)=='table') and complete_model:add(prep_model)
  result = (type(arch_model)=='table') and complete_model:add(arch_model)


  complete_model = ModelHandler.initializeModelParams(complete_model, opt)
  complete_model = ModelHandler.postModelConfiguration(complete_model, opt)

  assert(complete_model, "<ModelHandler.generateModel> failed to generate new model ["..tostring(opt.arch.modelArch).."]")

  return complete_model
end

function ModelHandler.loadModelFile(arch_name)
  local modelsDirPath = 'Models.'
  local modelFilePath = modelsDirPath .. string.lower(arch_name)
  Log.write('loading model from '.. modelFilePath)
  local model = require(modelFilePath)
  return model
end

function ModelHandler.initializeModelParams(model, opt)
  if opt.arch.paramInit ~= nil then
    if opt.arch.paramInit.x ~= nil and opt.arch.paramInit.fn ~= nil then
      local initLayers = {'nn.Linear','nn.TernaryConnectLinear',
                          'nn.SpatialConvolution','cudnn.SpatialConvolution', 'nn.TernaryConnectSpatialConvolution'}
      for k,tensor in ipairs(opt.arch.paramInit.x:split(',')) do
        _.invoke(initLayers, function(modType)
            local mods = model:findModules(modType)
            for k=1,#mods do
              Log.write("<ModelHandler> calling [nninit."..opt.arch.paramInit.fn.."] on "..modType.."."..tensor.."#"..k)
              if _.isTable(opt.arch.paramInit.args) and #(opt.arch.paramInit.args)==_.size(opt.arch.paramInit.args) then -- determine if keys are ints from 1 or not
                mods[k]:init(tensor,nninit[opt.arch.paramInit.fn],unpack(opt.arch.paramInit.args))
              else
                mods[k]:init(tensor,nninit[opt.arch.paramInit.fn],opt.arch.paramInit.args)
              end
            end
          end)
        end
    end
  end
  return model
end

function ModelHandler.initializeTracing(model, opt)
  if opt.main.trace.acts ~= nil then
    local traceLayers = {'nn.Linear','nn.TernaryConnectLinear'}

    -- trace input to first layer
    model = (dp.InsertAtMatch{contains=traceLayers,
              add_module=nn.SampleRecord(paths.concat(opt.main.dataDstPath,'act_0.csv'),true),
              match_count=1, insert_after=false}):apply(model)

    istart, iend = (dp.AtMatch{contains=traceLayers,match_count=1}):apply(model)

    local i = 0
    while istart > 0 do
      i=i+1
      -- trace activations
      model = (dp.InsertAtMatch{start_at=istart, contains=traceLayers,
                add_module=nn.SampleRecord(paths.concat(opt.main.dataDstPath,'act_'..i..'.csv'),true),
                match_count=1, insert_after=true}):apply(model)

      istart, iend = (dp.AtMatch{start_at=iend+1, contains=traceLayers,match_count=1}):apply(model)

    end

    -- record last layer as 1d tensor
    local traceMods = model:findModules('nn.SampleRecord')
    traceMods[#traceMods].square=false

  end
  return model
end

function ModelHandler.postModelConfiguration(model, opt)

  local nGPU = opt.main.nGPU or 1
  if opt.main.cuda > 0 and nGPU > 1 then
      local net = model
      model = nn.DataParallelTable(1)
      for i = 1, nGPU do
          cutorch.setDevice(i)
          model:add(net:clone():cuda(), i)  -- Use the ith GPU
      end
      cutorch.setDevice(opt.main.cuda)
      cutorch.manualSeedAll(tonumber(opt.run.randseed) or 1)
  end

  if opt.main.cuda > 0 then
    model:cuda()
  end

  return model
end

function ModelHandler.getInputPreprocessors(opt)
  local input_preprocess = {}
  if opt.input.normalize > 0 then
    Log.write("<ModelHandler.getInputPreprocessors> adding Standardize ")
     table.insert(input_preprocess, dp.Standardize())
  elseif opt.input.scale > 0 then
    Log.write("<ModelHandler.getInputPreprocessors> adding Scale ")
    table.insert(input_preprocess, dp.Scale())
  end
  if opt.input.zca > 0 then
    Log.write("<ModelHandler.getInputPreprocessors> adding ZCA ")
    table.insert(input_preprocess, dp.ZCA())
  end
  if opt.input.gcn > 0 then
    Log.write("<ModelHandler.getInputPreprocessors> adding GCN ")
    table.insert(input_preprocess, dp.GCN())
  end
  if opt.input.lecunlcn > 0 then
    Log.write("<ModelHandler.getInputPreprocessors> adding LeCunLCN ")
    table.insert(input_preprocess, dp.LeCunLCN{progress=true})
  end
  if opt.input.narize.bitCount > 0 and not ConfHandler.IsOnlineDistortEn(opt) then
    Log.write("<ModelHandler.getInputPreprocessors> adding Narize")
    table.insert(input_preprocess, dp.Narize(opt.input.narize.bitCount, opt.input.narize.signed))
  end
  return input_preprocess
end

function ModelHandler.getDataSource(opt, input_preprocess)
  dp.DATA_DIR = 'dataSrc'
  local dsName = ""
  local dsConf = {}
  if opt.input.dataset:upper() == 'MNIST' then
     dsConf = {input_preprocess = input_preprocess, fracture_data=opt.input.fractureData, valid_ratio = opt.input.validRatio}
     dsName = "Mnist"
  elseif opt.input.dataset:upper() == 'NOTMNIST' then
     dsConf = {input_preprocess = input_preprocess, fracture_data=opt.input.fractureData, valid_ratio = opt.input.validRatio}
     dsName = "NotMnist"
  elseif opt.input.dataset:upper() == 'CIFAR10' then
     dsConf = {input_preprocess = input_preprocess, fracture_data=opt.input.fractureData, valid_ratio = opt.input.validRatio}
     dsName = "Cifar10"
  elseif opt.input.dataset:upper() == 'NCIFAR10' then -- handmade ternary cifar10 tensor loader
     dsConf = {input_preprocess = input_preprocess, fracture_data=opt.input.fractureData, valid_ratio = opt.input.validRatio, scale={}}
     dsName = "nCifar10"
  elseif opt.input.dataset:upper() == 'CIFAR100' then
     dsConf = {input_preprocess = input_preprocess, fracture_data=opt.input.fractureData, valid_ratio = opt.input.validRatio}
     dsName = "Cifar100"
  elseif opt.input.dataset:upper() == 'SVHN' then
    dsConf = {input_preprocess = input_preprocess, fracture_data=opt.input.fractureData, valid_ratio = opt.input.validRatio}
    dsName = "SvhnStd"
  elseif opt.input.dataset:upper() == 'SVHNCANNY' then
    dsConf = {input_preprocess = input_preprocess, fracture_data=opt.input.fractureData, valid_ratio = opt.input.validRatio}
    dsName = "SvhnCanny"
  elseif opt.input.dataset:upper() == 'GTSRB32' then
    dsConf = {input_preprocess = input_preprocess, fracture_data=opt.input.fractureData, valid_ratio = opt.input.validRatio, img_size = 32, shuffle=true, balance=false, crop=true}
    dsName = "Gtsrb"
  elseif opt.input.dataset:upper() == 'GTSRB48' then
    dsConf = {input_preprocess = input_preprocess, fracture_data=opt.input.fractureData, valid_ratio = opt.input.validRatio, img_size = 48, shuffle=true, balance=false, crop=true}
    dsName = "Gtsrb"
  elseif opt.input.dataset:upper() == 'CUSTOM' then
    dsConf = {input_preprocess = input_preprocess, fracture_data=opt.input.fractureData, valid_ratio = opt.input.validRatio}
    dsName = "Custom"
  else
      error("Unknown Dataset")
  end

  return dp.CacheLoader():load(dsName,dsConf,opt.input.cacheOnMiss)
end
