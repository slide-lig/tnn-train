package.path = '../lib/?;../lib/?.lua;../bin/?.lua;'..package.path

-- core
require 'Runtime.Coredp'
-- utils
require 'utils.ConfHandler'
require 'ProgressiveTernarization.Progress'


package.path = os.getenv("LUA_PATH")
require 'dp'
package.path = '../lib/?;../lib/?.lua;'..package.path

nninit = require 'nninit'
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
require 'dp.data.svhncanny'
require 'dp.data.cacheloader'
require 'dp.data.ncifar10'
require 'nn.Distort'
require 'nn.Narize'
require 'nn.Record'
require 'nn.StochasticFire'
require 'nn.SqrHingeEmbeddingCriterion'
require 'nn.OneHotEncode'

if #arg <= 0 then
  print("<mainProgressTernarization> missing arguments")
  os.exit()
end

  torch.setdefaulttensortype('torch.DoubleTensor')
-- compile configurations
local configs = ConfHandler.GetFullConfs(arg)

for optNum,opt in pairs(configs) do

  local default_progress = {}
  default_progress.javaCmd = "java -Xmx80G -cp"
  default_progress.jarPath = "~/NeuronBinarization/target/NeuronBinarization-1.0-jar-with-dependencies.jar"
  default_progress.jarMainClass1 = "fr.liglab.esprit.binarization.BinarizeAllPrecomp"
  default_progress.jarMainClass2 = "fr.liglab.esprit.binarization.BinarizeAllConvPrecomp"
  default_progress.jarMainClass3 = "fr.liglab.esprit.binarization.BinarizeAllFinalLayer"
  default_progress.jarLogFileName = "NeuronBinarization.log"
  default_progress.jarArg_ei = "0"
  default_progress.jarArg_ec = "0"
  default_progress.jarArg_e = "0"
  default_progress.trDataFilename = "trData"
  default_progress.actDataFilename = "actData"
  default_progress.refDataFilename = "refData"
  default_progress.groundTruthDataFilename = "groundTruth.csv"
  default_progress.groundTruthDataDirPath = "" -- change this to an existing dir with groundData, preventing recalculation
  default_progress.retrain = 1

  opt.main._progress = opt.main._progress or default_progress

  if opt.main.cuda > 0 then
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    cutorch.setDevice(opt.main.cuda)
    Log.write("<ModelHandler.configureTorch> including CUDA modules")
  end

  if (opt.main.loadExp ~= nil and opt.main.loadExp ~= "") then -- Core.run has already ran at least once

      print("<mainProgressTernarization> assuming network is already trained at least once")
      opt = Progress.run(opt)
  end

  while opt do
    -- force retraining for now
    --if opt.main._progress.retrain > 0 then
      opt.main.loadExp = ""
      opt = Coredp.run(opt)
    --else
    --  opt.main.bestNet = opt.main.loadNet -- is bestnet a net or experiement??
    --end
    opt = Progress.run(opt)
  end

end
