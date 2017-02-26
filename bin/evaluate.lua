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
require 'dp.preprocess.narize'
require 'dp.sampler.ligsampler'
require 'dp.data.gtsrb'
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

if #arg <= 0 then
  print("<mainProgressTernarization> missing arguments")
  os.exit()
end

torch.setdefaulttensortype('torch.DoubleTensor')

--
-- config
--
local configs = ConfHandler.GetFullConfs(arg)

for optNum,opt in pairs(configs) do

  if opt.main.cuda > 0 then
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    cutorch.setDevice(opt.main.cuda)
    Log.write("<ModelHandler.configureTorch> including CUDA modules")
  end

  --
  -- data
  --
  input_preprocess = ModelHandler.getInputPreprocessors(opt)
  ds = ModelHandler.getDataSource(opt, input_preprocess)

  --
  -- model
  --
  local model = nil
  if opt.main.loadExp ~= nil and paths.filep(opt.main.loadExp) then
    xp = torch.load(opt.main.loadExp)
    model = xp:model().module
    Log.write("<evaluate> loaded model from experiement ["..opt.main.loadExp.."]")
  else
    model = ModelHandler.getModel(ds, opt)
  end

  --
  -- experiment
  --
  train = dp.Evaluator{
     feedback = dp.Confusion(),
     sampler = dp.LigSampler{batch_size = opt.run.batchSize}
  }
  valid = dp.Evaluator{
     feedback = dp.Confusion(),
     sampler = dp.LigSampler{batch_size = opt.run.batchSize}
  }
  test = dp.Evaluator{
     feedback = dp.Confusion(),
     sampler = dp.LigSampler{batch_size = opt.run.batchSize}
  }

  xp = dp.Experiment{
     model = model,
     optimizer = train,
     validator = valid,
     tester = test,
     observer = {
        dp.EpochLogger(),
        dp.NetworkSaver{
           error_report = {'validator','feedback','confusion','accuracy'},
           maximize = true,
           max_epochs = 1,
           save_dir=paths.concat(opt.main.dataDstPath,'evaluate_results')
        }
     },
     random_seed = opt.run.randseed,
     max_epoch = 1
  }

  xp:run(ds)
end
