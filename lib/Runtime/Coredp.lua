require 'optim'

require 'utils.Log'
require 'utils.DataHandler'
require 'utils.ModelHandler'
require 'utils.TensorHandler'
require 'utils.ConfHandler'
require 'utils.CalcHandler'

Coredp = {}

function Coredp.run(options)
  opt = options -- assigning to global namespace

  package.path = os.getenv("LUA_PATH") -- avoiding naming conflicts
  require 'dp'
  require 'gnuplot'
  package.path = '../lib/?;../lib/?.lua;'..package.path

  local seed = os.time()
  if opt.run.randseed ~= nil and tonumber(opt.run.randseed) > 0 then
    seed = tonumber(opt.run.randseed) or 1
    torch.manualSeed(seed)
    math.randomseed( seed )
    math.random(); math.random(); math.random()
    Log.write("<ModelHandler.configureTorch> using default built-in torch seed generator, giving [".. torch.initialSeed() .."]")
  end

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

  nninit = require 'nninit.nninit'
  require 'dp.observer.resultlogger'
  require 'dp.observer.networksaver'
  require 'dp.observer.epochlogger'
  require 'dp.preprocess.scale'
  require 'dp.preprocess.narize'
  require 'dp.sampler.ligsampler'
  require 'dp.data.gtsrb'
  require 'dp.data.custom'
  require 'dp.data.svhnstd'
  require 'dp.data.svhncanny'
  require 'dp.data.ncifar10'
  require 'dp.data.cacheloader'
  require 'nn.Distort'
  require 'nn.Narize'
  require 'nn.Record'
  require 'nn.StochasticFire'
  require 'nn.SqrHingeEmbeddingCriterion'
  require 'nn.OneHotEncode'
  require 'nn.TernaryConnectLinear'
  require 'nn.TernaryConnectSpatialConvolution'
  require 'nn.TernaryLinear'
  require 'nn.TernSpatialConvolution'
  require 'gnuplot.gnuplotx'

  torch.setdefaulttensortype('torch.DoubleTensor')
  -- check codehash of existing main config (if exists)
  local codehash = DataHandler.getCurrentGitCommit()
  if opt.main.codehash ~= nil and codehash ~= opt.main.codehash then print("<Core.run> warning codehash difference") end
  opt.main.codehash = codehash

  -- path config
  assert(paths.dir(paths.concat(opt.main.binPath,'dataSrc')) ~= nil,"<Core.run> Failed to find target of sym link bin/dataSrc")
  assert(paths.dir(paths.concat(opt.main.binPath,'dataDst')) ~= nil,"<Core.run> Failed to find target of sym link bin/dataDst")

  opt.main.dataDstPath = opt.main.dataDstPath or paths.concat(paths.concat(opt.main.binPath,'dataDst'), opt.arch.modelArch, opt.main.dstDir)
  assert(paths.dir(opt.main.dataDstPath) or paths.mkdir(opt.main.dataDstPath),"<Core.run> failed to make destination directory [".. opt.main.dataDstPath .."]")


  opt.main.filepaths = opt.main.filepaths or {}
  opt.main.filepaths.mainlog = opt.main.filepaths.mainlog or paths.concat(opt.main.dataDstPath, 'main.log')
  opt.main.filepaths.mainconf = opt.main.filepaths.mainconf or paths.concat(opt.main.dataDstPath, 'main.conf')
  opt.main.filepaths.results = opt.main.filepaths.results or paths.concat(opt.main.dataDstPath, 'results.json')

  Log.start(opt.main.filepaths.mainlog, opt)
  Log.write("<core> config: log output directory: "..opt.main.dataDstPath)

  --[[preprocessing]]--
  local input_preprocess = ModelHandler.getInputPreprocessors(opt)

  --[[data]]--
  ds = ModelHandler.getDataSource(opt, input_preprocess) -- assigning to global namespace
  Log.write("<core> dataset loaded: "..ds:name())

  --[[Model]]--
  if opt.main.loadExp ~= nil and paths.filep(opt.main.loadExp) then
    xp = torch.load(opt.main.loadExp)
    model = xp:model().module
    train = xp._optimizer -- ref needed for end-of-batch callback
    if opt.main.cuda > 0 then
      if opt.main.cudnn ~= nil and opt.main.cudnn > 0 then
        cudnn.convert(model, cudnn)
      end
      model:cuda()
    end
    opt.main._parameters, opt.main._gradParameters = model:getParameters()
    Log.write("<core> loaded previously saved experiment")
    Log.write(pretty.write(ConfHandler.scrubTbl(opt)))
    xp:run(ds)
    return opt
  end
  model = ModelHandler.getModel(ds, opt)

  --[[Optimization]]--
  if tablex.find(ConfHandler._supports.optim.fn,opt.run.optim.fn) == nil then
    error("Unknown optim fn: ["..opt.run.optim.fn.."]")
  end
  local optimize = optim[opt.run.optim.fn]
  Log.write("<core> optimization method: "..opt.run.optim.fn)
  local optimConf = tablex.copy(opt.run.optim.conf) or {}
  local optimState = {}
  if opt.run.optim.conf ~= nil then --special: boolean conversion in optim conf
    tablex.transform(ConfHandler.TryBoolean,optimConf) -- apply TryBoolean to each conf value
  end
  if opt.run.optim.fn == 'adaMax_binary_clip_shift' then --special: BinaryNet optim
    optimConf = tablex.merge(optimConf,BinaryNetUtils.getOptimConf(model, opt),true) -- no need to use ConfHandler.mergeTbl here as second arg is not a nested table
  end
  Log.write(optimConf)

  --[[Criterion]]--
  local loss_crit_args = nil
  local target_mod = nil
  if opt.arch.criterion == 'MultiMarginCriterion' then
    loss_crit_args = tonumber(opt.arch.criterionArg) or opt.arch.criterionArg
    if loss_crit_args=='' then loss_crit_args = nil end
  elseif opt.arch.criterion == 'SqrHingeEmbeddingCriterion' then
    target_mod = nn.OneHotEncode(#ds:classes(),1) -- criterion takes one-hot encoded targets
  end
  local loss_crit = nn[opt.arch.criterion](loss_crit_args)
  Log.write("<core> criterion: "..tostring(loss_crit))

  --[[Propagators]]--

  local lrEvalOnBatch = false

  train = dp.Optimizer{
     acc_update = false,
     loss = nn.ModuleCriterion(loss_crit, nil, nn.Convert()),
     epoch_callback = function(model, report) -- called every epoch (at start)
      opt.main._batch_num = opt.main._batch_num or 1
      -- TODO should NetworkSaver not be doing this?

      -- update lr... note: sgd & adagrad update themselves according to 1/t decay formula
      if tablex.find({"adam","adamax","rmsprop"},opt.run.optim.fn) ~= nil and optimConf.learningRateDecay ~= nil then
        if type(optimConf.learningRateDecay) == 'table' or (type(optimConf.learningRateDecay) == 'string' and not string.find(optimConf.learningRateDecay,'batch')) then
          optimConf.learningRate = CalcHandler.evalLearningRate(optimConf.learningRate, optimConf.learningRateDecay, report.epoch + 1, opt.main._batch_num)-- +1 since report updated following all batches
          Log.write("<core> evaluating learningRate before every epoch, learningRate=["..optimConf.learningRate.."]")
        else
          lrEvalOnBatch = true
          Log.write("<core> evaluating learningRate after every batch, learningRate=["..optimConf.learningRate.."]")
        end
      end

      if report.epoch > 0 then
        opt.main.loadExp = paths.concat(opt.main.dataDstPath,'best_epoch.dat')
        ConfHandler.scrubWrite(opt)
      end

      if opt.main.hist ~= nil and opt.main.hist.params ~= nil and opt.main.hist.params > 0 then
        p,gp = model:parameters()
        if p~=nil and type(p) == 'table' then
          Log.write("Plotting model parameters before epoch #"..report.epoch+1)
          local t_min = torch.min(p[1])
          local t_max = torch.max(p[1])
          for i=1,#p do
            t_min = math.min(t_min,torch.min(p[i]))
            t_max = math.max(t_max,torch.max(p[i]))
          end
          for i=1,#p do
            Log.write("------")
            Log.write(i..") count: ["..p[i]:nElement().."] min: ["..torch.min(p[i]).."] max: ["..torch.max(p[i]).."] std: ["..torch.std(p[i]).."] mean: ["..torch.mean(p[i]).."]")
            Log.write(gnuplot.hist_nofx(p[i],100,t_min,t_max))
            Log.write("------")
          end
        end
      end

     end,
     callback = function(model, report) -- called every batch (at end)
      report.tester.feedback.batch = report.tester.feedback.batch or {}

      -- update lr... note: sgd & adagrad update themselves according to 1/t decay formula
      if tablex.find({"adam","adamax","rmsprop"},opt.run.optim.fn) ~= nil and optimConf.learningRateDecay ~= nil then
        if lrEvalOnBatch then -- optimConf.learningRateDecay is an integer or string containing 'batch'
          optimConf.learningRate = CalcHandler.evalLearningRate(optimConf.learningRate, optimConf.learningRateDecay, report.epoch + 1, opt.main._batch_num)-- +1 since report updated following all batches
        end
      end

      feval = function(x_new)
        if opt.main._parameters ~= x_new then
         opt.main._parameters:copy(x_new)
        end

        if opt.main.trace ~= nil and opt.main.trace.loss ~= nil and opt.main.trace.loss > 0 then -- trace what lr was
         report.tester.feedback.batch.loss = report.tester.feedback.batch.loss or {}
         table.insert(report.tester.feedback.batch.loss, train.err)
        end

        return train.err, opt.main._gradParameters
       end

       optimize(feval, opt.main._parameters, optimConf, optimState)

       if opt.main.trace ~= nil and opt.main.trace.learningRate ~= nil and opt.main.trace.learningRate > 0 then -- trace what lr was
         report.tester.feedback.batch.learningRate = report.tester.feedback.batch.learningRate or {}
         local effectiveLR = optimConf.learningRate
         if tablex.find({"sgd","adagrad"},opt.run.optim.fn) ~= nil then -- recompute what is used internally in these optim fns
           effectiveLR = optimConf.learningRate / (1 + optimState.evalCounter*optimConf.learningRateDecay)
         end
         table.insert(report.tester.feedback.batch.learningRate, effectiveLR)
       end

       opt.main._gradParameters:zero()
  --      if opt.dp.accUpdate then
  --         model:accUpdateGradParameters(model.dpnn_input, model.output, opt.run.initLR)
  --      else
  --         model:updateGradParameters(opt.arch.momentum) -- affects gradParams
  --         model:updateParameters(opt.run.initLR) -- affects params
  --      end
  --      model:maxParamNorm(opt.dp.maxOutNorm) -- affects params
  --      model:zeroGradParameters() -- affects gradParams

        opt.main._batch_num = opt.main._batch_num + 1
     end,
     feedback = dp.Confusion(),
     sampler = dp.LigSampler{batch_size = opt.run.batchSize}, -- ShuffleSampler
     progress = true
  }
  valid = dp.Evaluator{
     feedback = dp.Confusion(),
     sampler = dp.LigSampler{batch_size = opt.run.batchSize}
  }
  test = dp.Evaluator{
     feedback = dp.Confusion(),
     sampler = dp.LigSampler{batch_size = opt.run.batchSize}
  }

  --[[Experiment]]--

  xp = dp.Experiment{
     model = model,
     optimizer = train,
     validator = valid,
     tester = test,
     observer = {
        --dp.FileLogger(opt.main.dataDstPath),
        dp.EpochLogger(),
        dp.ResultLogger(opt.main.dataDstPath),
        dp.NetworkSaver{
           error_report = {'validator','feedback','confusion','accuracy'},
           maximize = true,
           max_epochs = opt.main.stopEarly,
           save_dir=opt.main.dataDstPath
        }
     },
     random_seed = seed,
     max_epoch = opt.main.epochCount,
     target_module = target_mod
  }


  if opt.main.cuda > 0 then
    xp:cuda()
  end

  torch.save(paths.concat(opt.main.dataDstPath,'xp0.dat'),xp)

  Log.write(pretty.write(ConfHandler.scrubTbl(opt)))
  Log.write(model)
  xp:run(ds)

  return opt
end
