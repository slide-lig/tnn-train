package.path = '../lib/?;../lib/?.lua;'..package.path

require 'torch'
require 'paths'
require 'csvigo'
-- core
require 'Runtime.Coredp'
-- utils
require 'utils.DataHandler'
require 'utils.ConfHandler'
--
--require 'Public.nn.TernaryLinear'
--require 'Public.nn.StochasticFire'

-- function Progress.run(conf)
--
--    check conf or dataDstPath for best net
--    check net for progress num
--    check current progress status by examing filesystem
--
--    create archive dir
--    create progress dir
--    check last progress dir exists
--
--    copy bestnet to progress dir
--    move all but .conf to archive
--    recreate input data (if not found)
--    test bestnet recording first layer
--
--    convert bestnet to csv
--    convert recordings to csv
--
--    run java using bestnet.csv, REF from last progress, ACT in current progress dir
--    use params from jar to ternarize next linear layer of network
--    set network as main.loadNet, and return config
--


Progress = {}
Progress.currentFileStatus = {}
Progress.datasource = nil
Progress.finalLayerTernarized = false
Progress.isModuleCNNOverride = 0
Progress.skipCleanOverride = false

function Progress.run(conf)
  print("---------------------------------------------------")
  print("---------------------------------------------------")
  print("<Progress> progressing ternarization")
  print("---------------------------------------------------")
  print("---------------------------------------------------")

  print("<Progress> checking current state of filesystem")

  local patternBestNet = "^best_epoch.dat$"

  conf.main.bestNet = conf.main.bestNet or paths.concat(conf.main.dataDstPath, Progress.findFirstFileMatch(conf.main.dataDstPath, patternBestNet))
  assert(paths.filep(conf.main.bestNet), "<Progress> missing main.bestNet from config/results")

  print("<Progress> examining best net file")
  print(conf.main.bestNet)

  local bestNetExp = DataHandler.loadExperiment(conf.main.bestNet)
  local bestNetModule = bestNetExp:model().module
  local isModuleCNN = Progress.isModuleCNN(bestNetModule)
  local hasModuleConv = Progress.hasModuleConvolution(bestNetModule)
  local progressNum = Progress.getCurrentProgressNum(bestNetModule)
  local remainingProgressNum = Progress.getRemainingProgressNum(bestNetModule)

  if remainingProgressNum == 0 then
    print("<Progress> no more work to do")
    Progress.performFinalTest(conf, bestNetModule)
    return nil
  end

  Progress.findFileStatusFromDir(conf, progressNum)

  if Progress.currentFileStatus.progressDir == "" then
    Progress.createWorkingDirs(conf, progressNum)
  end

  if Progress.currentFileStatus.netFile == "" then
    Progress.moveFiles(conf, progressNum)
  end

  if Progress.currentFileStatus.recordInFile == "" or (remainingProgressNum > 1 and Progress.currentFileStatus.recordOutFile == "") then
    Progress.generateRecordFiles(conf, progressNum, bestNetModule)
  end

  if Progress.currentFileStatus.netWeightCSVFile == "" then
    Progress.generateNetWeightCSV(conf, progressNum, bestNetModule)
  end

  if Progress.currentFileStatus.netBiasCSVFile == "" then
    Progress.generateNetBiasCSV(conf, progressNum, bestNetModule)
  end

  if Progress.currentFileStatus.recordInCSVFile == "" then
    Progress.generateRecordInCSVFiles(conf, progressNum, isModuleCNN)
  else
      local record = torch.load(Progress.currentFileStatus.recordInFile)
      Progress.latestTrChSize = record:dim() > 2 and record:size(2) or 1
      Progress.latestTrSize = record:dim() > 2 and record:size(3) or record:size(2)
      Progress.latestTrMax = torch.max(record)
  end

  if Progress.currentFileStatus.recordActCSVFile == "" then
    Progress.generateRecordActCSVFiles(conf, progressNum, isModuleCNN)
  end

  if Progress.currentFileStatus.recordOutCSVFile == "" then
    if remainingProgressNum > 1 then
      Progress.generateRecordOutCSVFiles(conf, progressNum, isModuleCNN)
    end
  end

  if Progress.currentFileStatus.ternaryParamsFile == "" then
    Progress.callJavaForTernaryParams(conf, progressNum, remainingProgressNum, hasModuleConv)
  end

  -- quick fix: reload best net module, to resolve failed write "Unwritable object <cdata> at <?>.<?>.modules.3.<?>.poolDesc"
  --local progressDirPath = Progress.getProgressDirPath(conf, progressNum)
  --bestNetExp = DataHandler.loadExperiment(paths.concat(progressDirPath,paths.basename(conf.main.bestNet)))
  --bestNetModule = bestNetExp:model().module

  if Progress.currentFileStatus.ternaryNetFile == "" then
    Progress.convertTernaryNetLayer(conf, progressNum, bestNetModule)
  end

  bestNetModule = DataHandler.loadModule(Progress.currentFileStatus.ternaryNetFile)

  if Progress.finalLayerTernarized then
    print("<Progress> no more work to do")
    Progress.performFinalTest(conf, bestNetModule)
    return nil
  end

  if paths.filep(Progress.currentFileStatus.ternaryNetFile) and not Progress.skipCleanOverride then
    Progress.cleanProgressDir(conf, progressNum)
  end

  conf.main.loadNet = Progress.currentFileStatus.ternaryNetFile
  return conf

end

function Progress.createWorkingDirs(conf, progressNum)

    print("<Progress> creating working directories")
    local lastProgressDirPath = Progress.getProgressDirPath(conf, progressNum-1)
    local archiveDirPath = Progress.getArchiveDirPath(conf, progressNum)
    local progressDirPath = Progress.getProgressDirPath(conf, progressNum)

    if progressNum > 1 then
      assert(paths.dirp(lastProgressDirPath), "<Progress> missing progress dir ["..lastProgressDirPath.."]")
    end

    paths.mkdir(archiveDirPath)
    print(archiveDirPath)
    paths.mkdir(progressDirPath)
    print(progressDirPath)

end

function Progress.moveFiles(conf, progressNum)

    print("<Progress> moving files to working directories")
    local archiveDirPath = Progress.getArchiveDirPath(conf, progressNum)
    local progressDirPath = Progress.getProgressDirPath(conf, progressNum)
    os.execute("cp "..conf.main.bestNet..' '..progressDirPath)
    for f in paths.iterfiles(conf.main.dataDstPath) do
        if f ~= paths.basename(conf.main.filepaths.mainconf) then
          os.execute("mv "..paths.concat(conf.main.dataDstPath,f)..' '..archiveDirPath)
        else
          os.execute("cp "..paths.concat(conf.main.dataDstPath,f)..' '..archiveDirPath)
        end
    end
    local bestNetPath = paths.concat(progressDirPath, paths.basename(conf.main.bestNet))
    assert(paths.filep(bestNetPath), "<Progress> failed to move files, best net not in working dir")
    Progress.currentFileStatus.netFile = bestNetPath

end

function Progress.generateRecordFiles(conf, progressNum, module)

    print("<Progress> prepraring model/data for testing")

---
    local tempDistort = conf.input.distort
    conf.input.distort = {count=0,rotate=0,scale=0,translate=0} -- hack: to ignore rotations
    local input_preprocess = ModelHandler.getInputPreprocessors(conf)
    Progress.datasource = ModelHandler.getDataSource(conf, input_preprocess)
    conf.input.distort = tempDistort

    -- does ground truth need generating?

    local groundTruthDirPath = Progress.getGroundTruthDirPath(conf)
    local groundTruthPathCSV = paths.concat(groundTruthDirPath, conf.main._progress.groundTruthDataFilename)

    if not paths.filep(groundTruthPathCSV) then

      print("<Progress> creating GroundTruth file")
      DataHandler.write1DTensorToCSV(Progress.datasource:get('train','target','default'), groundTruthPathCSV)
      collectgarbage()
      DataHandler.write1DTensorToCSV(Progress.datasource:get('valid','target','default'), groundTruthPathCSV)
      collectgarbage()
      DataHandler.write1DTensorToCSV(Progress.datasource:get('test','target','default'), groundTruthPathCSV)
      collectgarbage()
      print(groundTruthPathCSV)

    end

    -- test the network
    local progressDirPath = Progress.getProgressDirPath(conf, progressNum)
    local spillTo = paths.concat(progressDirPath,DataHandler.getRecordSaveName('test_1',1)) -- spill to this file
    local spillAfter = 50000000 -- spill after this many tensor elements are recorded

    -- remove preprocessing modules
    module = (dp.RemoveMatch{contains={'nn.Distort'},match_count=1}):apply(module)
    -- insert recording modules
    module = (dp.InsertAtMatch{contains={'nn.Linear','nn.SpatialConvolution','cudnn.SpatialConvolution'},add_module=nn.Record(spillTo,spillAfter):enable()}):apply(module)
    -- this may need to match on StochasticFire rather than the activationFn
    spillTo = paths.concat(progressDirPath,DataHandler.getRecordSaveName('test_2',1))
    module = (dp.InsertAtMatch{contains={'nn.'..conf.arch.activationFn,'cudnn.'..conf.arch.activationFn},add_module=nn.Record(spillTo,spillAfter):enable(), insert_after=true}):apply(module)

    local search_start = tablex.find_if(module.modules,function(v) return tablex.find({'nn.Linear','nn.SpatialConvolution','cudnn.SpatialConvolution'},torch.type(v))~= nil end) or #module.modules
    spillTo = paths.concat(progressDirPath,DataHandler.getRecordSaveName('test_3',1))
    module = (dp.InsertAtMatch{contains={'nn.Linear','nn.SpatialConvolution','cudnn.SpatialConvolution'},add_module=nn.Record(spillTo,spillAfter):enable(), start_at=search_start+1}):apply(module)

    print("<Progress> testing model")
    print(module)

---
    train = dp.Evaluator{
       feedback = dp.Confusion(),
       sampler = dp.Sampler{batch_size = conf.run.batchSize}
    }
    valid = dp.Evaluator{
       feedback = dp.Confusion(),
       sampler = dp.Sampler{batch_size = conf.run.batchSize}
    }
    test = dp.Evaluator{
       feedback = dp.Confusion(),
       sampler = dp.Sampler{batch_size = conf.run.batchSize}
    }

    xp = dp.Experiment{
       model = module,
       optimizer = train,
       validator = valid,
       tester = test,
       observer = {
          dp.EpochLogger(),
          dp.NetworkSaver{
             error_report = {'validator','feedback','confusion','accuracy'},
             maximize = true,
             max_epochs = conf.main.stopEarly,
             save_dir=paths.concat(conf.main.dataDstPath,'recordtest')
          }
       },
       random_seed = os.time(),
       max_epoch = 1
    }

    xp:run(Progress.datasource)

    DataHandler.saveModuleStateEpoch(module, progressDirPath, "test", conf, 1) -- save the recordings

    -- remove all Record modules
    module = (dp.RemoveMatch{contains={'nn.Record'},global=true}):apply(module)

    local patternBestNetInput = "^best_epoch.+1%.net%.record"
    local patternBestNetActs = "^best_epoch.+2%.net%.record"
    local patternBestNetOutput = "^best_epoch.+3%.net%.record"

    DataHandler.mergeTensorFiles(progressDirPath,patternBestNetInput.."%d*")
    DataHandler.mergeTensorFiles(progressDirPath,patternBestNetActs.."%d*")
    DataHandler.mergeTensorFiles(progressDirPath,patternBestNetOutput.."%d*")

    Progress.currentFileStatus.recordInFile = Progress.findFirstFileMatch(progressDirPath, patternBestNetInput.."$")
    if Progress.currentFileStatus.recordInFile ~= "" then
      Progress.currentFileStatus.recordInFile = paths.concat(progressDirPath, Progress.currentFileStatus.recordInFile)
    end

    Progress.currentFileStatus.recordActFile = Progress.findFirstFileMatch(progressDirPath, patternBestNetActs.."$")
    if Progress.currentFileStatus.recordActFile ~= "" then
      Progress.currentFileStatus.recordActFile = paths.concat(progressDirPath, Progress.currentFileStatus.recordActFile)
    end

    Progress.currentFileStatus.recordOutFile = Progress.findFirstFileMatch(progressDirPath, patternBestNetOutput.."$")
    if Progress.currentFileStatus.recordOutFile ~= "" then
      Progress.currentFileStatus.recordOutFile = paths.concat(progressDirPath, Progress.currentFileStatus.recordOutFile)
    end


end

function Progress.generateNetWeightCSV(conf, progressNum, module)

  print("<Progress> generating net weight csv file")

  local progressDirPath = Progress.getProgressDirPath(conf, progressNum)
  local bestNetWeightsPath = paths.concat(progressDirPath,paths.basename(conf.main.bestNet)) .. '.weight.csv'
  local recMods = {}
  if Progress.isModuleCNN(module) then
    recMods = module:findModules('nn.SpatialConvolution')
    if #recMods<=0 then
      recMods = module:findModules('cudnn.SpatialConvolution')
    end
  end
  local linMods = module:findModules('nn.Linear')
  for i=1,#linMods do
    table.insert(recMods, linMods[i])
  end
  local w = torch.totable(recMods[1].weight)

  if recMods[1].weight:dim() == 4 then
    w = torch.totable(recMods[1].weight:reshape(recMods[1].weight:size(1), recMods[1].weight:size(2)*recMods[1].weight:size(3)*recMods[1].weight:size(4)))
  end
  -- export FIRST Linear layer to csv
  csvigo.save({path=bestNetWeightsPath, data=w, seperator=',', mode='raw', verbose=false})
  collectgarbage()
  print(bestNetWeightsPath)
  Progress.currentFileStatus.netWeightCSVFile = bestNetWeightsPath

end

function Progress.generateNetBiasCSV(conf, progressNum, module)

  print("<Progress> generating net bias csv file")

  local progressDirPath = Progress.getProgressDirPath(conf, progressNum)
  local bestNetBiasPath = paths.concat(progressDirPath,paths.basename(conf.main.bestNet)) .. '.bias.csv'
  local recMods = {}
  if Progress.isModuleCNN(module) then
    recMods = module:findModules('nn.SpatialConvolution')
    if #recMods<=0 then
      recMods = module:findModules('cudnn.SpatialConvolution')
    end
  end
  local linMods = module:findModules('nn.Linear')
  for i=1,#linMods do
    table.insert(recMods, linMods[i])
  end
  local b = torch.totable(recMods[1].bias)
  for l=1,#b do b[l] = {b[l]} end -- make 2nd dimension, by putting each value into separate table
  csvigo.save({path=bestNetBiasPath, data=b, seperator=',', mode='raw', verbose=false})
  collectgarbage()
  print(bestNetBiasPath)
  Progress.currentFileStatus.netBiasCSVFile = bestNetBiasPath

end

function Progress.generateRecordInCSVFiles(conf, progressNum, isModuleCNN)

  print("<Progress> converting first linear layer recording to binary files")

  local progressDirPath = Progress.getProgressDirPath(conf, progressNum)
  local recordInCSVPath = paths.concat(progressDirPath, conf.main._progress.trDataFilename)
  local record = torch.load(Progress.currentFileStatus.recordInFile)

  Progress.latestTrMax = torch.max(record)
  Progress.latestTrChSize = record:dim() > 1 and record:size(2) or 1
  Progress.latestTrSize = record:dim() > 2 and record:size(3) or record:size(2)

  for i=1, record:size(1) do
    DataHandler.writeSignedChars(record[{{i}}], recordInCSVPath)
  end

  collectgarbage()
  print(recordInCSVPath)
  Progress.currentFileStatus.recordInCSVFile = recordInCSVPath

end

function Progress.generateRecordActCSVFiles(conf, progressNum, isModuleCNN)

  print("<Progress> converting non-linear layer activations recording to binary files")

  local progressDirPath = Progress.getProgressDirPath(conf, progressNum)
  local recordActCSVPath = paths.concat(progressDirPath, conf.main._progress.actDataFilename)
  local record = torch.load(Progress.currentFileStatus.recordActFile)

  if record:dim() > 2 then
    for i=1,record:size(2) do
      DataHandler.writeFloats(record[{{},{i}}], recordActCSVPath..tostring(i-1))
    end
  else
    DataHandler.writeFloats(record, recordActCSVPath..'0')
  end

  collectgarbage()
  print(recordActCSVPath)
  Progress.currentFileStatus.recordActCSVFile = recordActCSVPath..'0'

end

function Progress.generateRecordOutCSVFiles(conf, progressNum, isModuleCNN)

  print("<Progress>  converting second linear layer recording to binary files")

  local progressDirPath = Progress.getProgressDirPath(conf, progressNum)
  local recordOutCSVPath = paths.concat(progressDirPath, conf.main._progress.refDataFilename)
  local record = torch.load(Progress.currentFileStatus.recordOutFile)

  for i=1, record:size(1) do
    DataHandler.writeSignedChars(record[{{i}}], recordOutCSVPath)
  end

  collectgarbage()
  print(recordOutCSVPath)
  Progress.currentFileStatus.recordOutCSVFile = recordOutCSVPath

end

function Progress.cleanProgressDir(conf, progressNum)
  -- some of this is duplicated work
  local progressDirPath = Progress.getProgressDirPath(conf, progressNum)
  -- rm *.record
  os.execute("rm "..paths.concat(progressDirPath,"*.record"))
  -- rm actdata*
  os.execute("rm "..paths.concat(progressDirPath,conf.main._progress.actDataFilename).."*")
  -- rm trData.csv
  os.execute("rm "..paths.concat(progressDirPath,conf.main._progress.trDataFilename).."*")
  -- TODO how about ref data from previous progressDirPath??
  -- rm oldParts/
  os.execute("rm -r "..paths.concat(progressDirPath,"oldParts"))
  -- rm best_epoch.dat* except .net
  os.execute("rm "..paths.concat(progressDirPath,"*.record"))
end

function Progress.callJavaForTernaryParams(conf, progressNum, remainingProgressNum, hasModuleConv)

  print("<Progress> using jar to create ternary params")

  local groundTruthDirPath = Progress.getGroundTruthDirPath(conf)
  local lastProgressDirPath = Progress.getProgressDirPath(conf, progressNum-1)
  local progressDirPath = Progress.getProgressDirPath(conf, progressNum)

  print(conf.main._progress.jarPath)

  local paramsFilename = "ternary.params.csv"
  if remainingProgressNum == 1 then
    paramsFilename = "ec_"..conf.main._progress.jarArg_ec .. "_" .. paramsFilename
    paramsFilename = "ei_"..conf.main._progress.jarArg_ei .. "_" .. paramsFilename
  else
    paramsFilename = "e_"..conf.main._progress.jarArg_e .. "_" .. paramsFilename
  end

  local jarArgs = {trData=paths.concat(progressDirPath, conf.main._progress.trDataFilename),
                    actData=paths.concat(progressDirPath, conf.main._progress.actDataFilename),
                    refData=paths.concat(lastProgressDirPath, conf.main._progress.refDataFilename),
                    groundTruthData=paths.concat(groundTruthDirPath, conf.main._progress.groundTruthDataFilename),
                    weightData=Progress.currentFileStatus.netWeightCSVFile,
                    biasData=Progress.currentFileStatus.netBiasCSVFile,
                    outFile=paths.concat(progressDirPath,paramsFilename)}

  local cmdJavaTernarize = conf.main._progress.javaCmd.." "..conf.main._progress.jarPath

  if remainingProgressNum == 1 then
    cmdJavaTernarize = cmdJavaTernarize .. " " .. conf.main._progress.jarMainClass3
  elseif hasModuleConv then
    cmdJavaTernarize = cmdJavaTernarize .. " " .. conf.main._progress.jarMainClass2
  else
    cmdJavaTernarize = cmdJavaTernarize .. " " .. conf.main._progress.jarMainClass1
  end

  cmdJavaTernarize = cmdJavaTernarize.." "..
                      "-t "..jarArgs.trData.." "..
                      "-w "..jarArgs.weightData.." "..
                      "-o "..jarArgs.outFile.." "..
                      "-ic "..Progress.latestTrChSize.." "..
                      "-ix "..Progress.latestTrSize.." "..
                      "-iy "..Progress.latestTrSize

  if remainingProgressNum == 1 then
    cmdJavaTernarize = cmdJavaTernarize.." -g "..jarArgs.groundTruthData
    cmdJavaTernarize = cmdJavaTernarize.." -ei "..conf.main._progress.jarArg_ei
    cmdJavaTernarize = cmdJavaTernarize.." -ec "..conf.main._progress.jarArg_ec
  elseif hasModuleConv then
    cmdJavaTernarize = cmdJavaTernarize.." -a "..jarArgs.actData
    cmdJavaTernarize = cmdJavaTernarize.." -e "..conf.main._progress.jarArg_e

    cmdJavaTernarize = cmdJavaTernarize.." -cp "..conf.arch.convPadding or 1
    cmdJavaTernarize = cmdJavaTernarize.." -imax "..Progress.latestTrMax
    cmdJavaTernarize = cmdJavaTernarize.." -cx "..conf.arch.convKernelSize
    cmdJavaTernarize = cmdJavaTernarize.." -cy "..conf.arch.convKernelSize
  else
    cmdJavaTernarize = cmdJavaTernarize.." -a "..jarArgs.actData
    cmdJavaTernarize = cmdJavaTernarize.." -e "..conf.main._progress.jarArg_e
  end
  if progressNum > 1 then
    cmdJavaTernarize = cmdJavaTernarize.." -r "..jarArgs.refData
  end

  cmdJavaTernarize = cmdJavaTernarize.." > "..paths.concat(progressDirPath, conf.main._progress.jarLogFileName)

  print(cmdJavaTernarize)

  local javaTernarizeTime = sys.clock()
  os.execute(cmdJavaTernarize)
  print("<Progress> completed")
  print("<Progress> time taken : " .. string.format("%.2f",(sys.clock() - javaTernarizeTime)) .. "s")

  print("<Progress> ternary params saved to file")
  print(jarArgs.outFile)

  assert(paths.filep(jarArgs.outFile), "<Progress> ternary params missing, problem with java binarization?")

  Progress.currentFileStatus.ternaryParamsFile = jarArgs.outFile

end

function Progress.convertTernaryNetLayer(conf, progressNum, module)

  print("<Progress> converting net layer to ternary layer")

  local dataTbl = csvigo.load({path=Progress.currentFileStatus.ternaryParamsFile, seperator=',', mode='raw', verbose=false})
  assert(#dataTbl[1]>=4, "<Progress> found less than 4 params on first line of ternary params file... quitting!")

  local netParamsTbl = {}
  for k=1,#dataTbl do
    table.insert(netParamsTbl,{dataTbl[k][1],dataTbl[k][2],dataTbl[k][3],dataTbl[k][4]})
  end
  local netParamsTensor = torch.Tensor(netParamsTbl)

  module = Progress.replaceFirstNonTernaryLayer(module, netParamsTensor)

  -- remove Non-Linear, Stochastic etc Modules
  module = (dp.RemoveMatch{contains={'nn.BatchNormalization','nn.SpatialBatchNormalization','nn.'..conf.arch.activationFn,'cudnn.'..conf.arch.activationFn,'nn.Dropout'}, match_count=3}):apply(module)
  module = (dp.RemoveMatch{contains={'nn.StochasticFire'}, match_count=1}):apply(module)
  -- save net
  local ternaryNetName = paths.basename(conf.main.bestNet)..'.ternarized.'..tostring(progressNum)..'.net'
  local progressDirPath = Progress.getProgressDirPath(conf, progressNum)
  local ternaryNetPath = paths.concat(progressDirPath, ternaryNetName)
  DataHandler.saveModule(ternaryNetPath,module,conf)
  print("<Progress> progressed net saved to file")
  print(ternaryNetPath)
  Progress.currentFileStatus.ternaryNetFile = ternaryNetPath

end

function Progress.getArchiveDirPath(conf, progressNum)

  local archivePrefix = "archive" -- before number
  return paths.concat(conf.main.dataDstPath, archivePrefix..tostring(progressNum))

end

function Progress.getGroundTruthDirPath(conf)
  if conf.main._progress.groundTruthDataDirPath ~= nil and conf.main._progress.groundTruthDataDirPath ~= "" then
    return conf.main._progress.groundTruthDataDirPath
  end
  return Progress.getProgressDirPath(conf, 1)
end

function Progress.getProgressDirPath(conf, progressNum)

  local progressPrefix = "progress" -- before number
  return paths.concat(conf.main.dataDstPath, progressPrefix..tostring(progressNum))

end

function Progress.binarizeWeights(weights, posWeightCounts, negWeightCounts)
  -- for each neuron
  for i=1,weights:size(1) do
    -- find threshold values specific to this neuron
    local wCount = weights[i]:nElement()
    local orderedWeights, orderedIndices = weights[i]:resize(wCount):sort()
    -- select the indicies to fire
    local lowIndices = torch.Tensor()
    local highIndices = torch.Tensor()

    weights[i]:zero()

    if negWeightCounts[i] > 0 then
      lowIndices = orderedIndices[{{1,negWeightCounts[i]}}]
      lowIndices:apply(function(x)
        weights[i][x] = -1
      end)
    end

    if posWeightCounts[i] > 0 then
      highIndices = orderedIndices[{{wCount-posWeightCounts[i]+1,wCount}}]
      highIndices:apply(function(x)
        weights[i][x] = 1
      end)
    end

  end
  return weights
end

function Progress.findFileStatusFromDir(conf, progressNum)
  Progress.currentFileStatus = {
    progressDir="",
    netFile="",
    recordInFile="",
    recordActFile="",
    recordOutFile="",
    netWeightCSVFile="",
    netBiasCSVFile="",
    recordInCSVFile="",
    recordActCSVFile="",
    recordOutCSVFile="",
    ternaryParamsFile="",
    ternaryNetFile=""
  }

  local patternBestNet = "^best_epoch.dat$"
  local patternBestNetInput = "^best_epoch.+1%.net%.record$"
  local patternBestNetActs = "^best_epoch.+2%.net%.record$"
  local patternBestNetOutput = "^best_epoch.+3%.net%.record$"
  local patternBestNetWeight = "^best_epoch.+%.weight%.csv$"
  local patternBestNetBias = "^best_epoch.+%.bias%.csv$"
  local patternBestNetInputCSV = conf.main._progress.trDataFilename
  local patternBestNetActCSV = conf.main._progress.actDataFilename.."%d*"
  local patternBestNetOutputCSV = conf.main._progress.refDataFilename
  local patternTernaryParams = ".+ternary%.params%.csv"
  local patternTernarizedNet = ".+ternarized."..progressNum..".net"

  local progressDirPath = Progress.getProgressDirPath(conf, progressNum)
  if not paths.dirp(progressDirPath) then return Progress.currentFileStatus end

  Progress.currentFileStatus.progressDir = progressDirPath
  Progress.currentFileStatus.netFile = Progress.findFirstFileMatch(progressDirPath, patternBestNet)
  Progress.currentFileStatus.recordInFile = Progress.findFirstFileMatch(progressDirPath, patternBestNetInput)
  Progress.currentFileStatus.recordActFile = Progress.findFirstFileMatch(progressDirPath, patternBestNetActs)
  Progress.currentFileStatus.recordOutFile = Progress.findFirstFileMatch(progressDirPath, patternBestNetOutput)
  Progress.currentFileStatus.netWeightCSVFile = Progress.findFirstFileMatch(progressDirPath, patternBestNetWeight)
  Progress.currentFileStatus.netBiasCSVFile = Progress.findFirstFileMatch(progressDirPath, patternBestNetBias)
  Progress.currentFileStatus.recordInCSVFile = Progress.findFirstFileMatch(progressDirPath, patternBestNetInputCSV)
  Progress.currentFileStatus.recordActCSVFile = Progress.findFirstFileMatch(progressDirPath, patternBestNetActCSV)
  Progress.currentFileStatus.recordOutCSVFile = Progress.findFirstFileMatch(progressDirPath, patternBestNetOutputCSV)
  Progress.currentFileStatus.ternaryParamsFile = Progress.findFirstFileMatch(progressDirPath, patternTernaryParams)
  Progress.currentFileStatus.ternaryNetFile = Progress.findFirstFileMatch(progressDirPath, patternTernarizedNet)

  for i,v in pairs(Progress.currentFileStatus) do
    if v ~= "" then
      Progress.currentFileStatus[i] = paths.concat(progressDirPath,v)
    end
  end

end

function Progress.findFirstFileMatch(path, filenamePattern)

  print("<Progress> looking for ["..filenamePattern.."] in files of ["..path.."]")
  for f in paths.iterfiles(path) do
    local firstMatch = string.match(f, filenamePattern) or ""
    if firstMatch ~= "" then print("<Progress> found ["..firstMatch.."]") return firstMatch end
  end
  print("<Progress> found nothing")
  return ""
end

function Progress.performFinalTest(conf, module)

    local tempDistort = conf.input.distort
    conf.input.distort = {count=0,rotate=0,scale=0,translate=0} -- hack: to ignore rotations
    local input_preprocess = ModelHandler.getInputPreprocessors(conf)
    Progress.datasource = ModelHandler.getDataSource(conf, input_preprocess, 0) -- 0 flag indicates no valid set
    conf.input.distort = tempDistort


    -- test the network

    -- remove preprocessing modules
    module = (dp.RemoveMatch{begins_with={'nn.Convert'},contains={'nn.Distort'},match_count=2}):apply(module)

    print("<Progress> testing final model")
    print(module)

---
    trainvalid = dp.Evaluator{
       feedback = dp.Confusion(),
       sampler = dp.Sampler{batch_size = conf.run.batchSize}
    }
    test = dp.Evaluator{
       feedback = dp.Confusion(),
       sampler = dp.Sampler{batch_size = conf.run.batchSize}
    }

    xp = dp.Experiment{
       model = module,
       optimizer = trainvalid, -- propagate train&valid set together
       tester = test,
       observer = {
          dp.EpochLogger(),
          dp.NetworkSaver{
             error_report = {'optimizer','feedback','confusion','accuracy'},
             maximize = true,
             save_dir=conf.main.dataDstPath
          }
       },
       random_seed = os.time(),
       max_epoch = 1
    }

    xp:run(Progress.datasource)

    print("<Progress> [accuracy on train/valid set = " .. string.format("%.2f",xp:report().optimizer.feedback.confusion.accuracy) .. "] [accuracy on test set = " .. string.format("%.2f",xp:report().tester.feedback.confusion.accuracy) .. "]")

    -- TODO log json results
    --DataHandler.logJson(conf.main.filepaths.train,"Accuracy", "epoch", tostring(1), tostring(bestOnTrainValid))
    --DataHandler.logJson(conf.main.filepaths.test,"Accuracy", "epoch", tostring(1), tostring(bestOnTest))

end


function Progress.getCurrentProgressNum(module)

  local currentProgressNum = 0

  if Progress.isModuleCNN(module) then
    currentProgressNum = #(module:findModules('nn.TernSpatialConvolution'))
  end

  local linModCount = #(module:findModules('nn.TernaryLinear'))
  currentProgressNum = currentProgressNum + linModCount + 1

  return currentProgressNum

end

function Progress.getRemainingProgressNum(module)

  local remainingProgressNum = 1

  if Progress.isModuleCNN(module) then
    remainingProgressNum = #(module:findModules('nn.SpatialConvolution')) + #(module:findModules('cudnn.SpatialConvolution')) + #(module:findModules('nn.Linear'))
  else
    remainingProgressNum = #(module:findModules('nn.Linear'))
  end

  -- special case: since final Linear layer is counted as any other...
  if Progress.finalLayerTernarized and #(module:findModules('nn.Linear')) > 0 then
    remainingProgressNum = remainingProgressNum - 1
  end

  return remainingProgressNum

end

function Progress.isModuleCNN(module)
  local numSmpMods = #(module:findModules('nn.SpatialMaxPooling')) + #(module:findModules('cudnn.SpatialMaxPooling'))
  return numSmpMods + Progress.isModuleCNNOverride > 0
end

function Progress.hasModuleConvolution(module)
  local numScMods = #(module:findModules('nn.SpatialConvolution')) + #(module:findModules('cudnn.SpatialConvolution'))
  return numScMods > 0
end


function Progress.replaceFirstNonTernaryLayer(module, netParamsTensor)

    local nonTernMods, nonTernContainers = module:findModules('nn.SpatialConvolution')
    if #nonTernMods <= 0 then
      nonTernMods, nonTernContainers = module:findModules('cudnn.SpatialConvolution')
    end
    local linMods, linContainers = module:findModules('nn.Linear')

    for i=1,#linMods do
      table.insert(nonTernMods, linMods[i])
    end
    for i=1,#linContainers do
      table.insert(nonTernContainers, linContainers[i])
    end

    if #nonTernMods > 0 then
      for i=1, #nonTernContainers do
        for j = 1, #(nonTernContainers[i].modules) do
          if nonTernContainers[i].modules[j] == nonTernMods[1] then

            print('<Progress> creating ternary layer')
            local replacementLayer = nn.TernaryLinear(nonTernMods[1].weight:size(2),nonTernMods[1].weight:size(1),netParamsTensor[{{},3}], netParamsTensor[{{},4}])
            if Progress.isModuleCNN(module) and (#(module:findModules('nn.SpatialConvolution')) + #(module:findModules('cudnn.SpatialConvolution'))) > 0 then
              replacementLayer = nn.TernSpatialConvolution(nonTernMods[1].nInputPlane, nonTernMods[1].nOutputPlane, nonTernMods[1].kW, nonTernMods[1].kH, nonTernMods[1].dW, nonTernMods[1].dH, nonTernMods[1].padW, nonTernMods[1].padH)
              replacementLayer.biasHi = netParamsTensor[{{},3}]
              replacementLayer.biasLo = netParamsTensor[{{},4}]
            end

            if #nonTernMods > 1 then
              nonTernContainers[i].modules[j] = replacementLayer
            else
              nonTernContainers[i].modules[j] = nn.Linear(nonTernMods[1].weight:size(2),nonTernMods[1].weight:size(1))
              nonTernContainers[i].modules[j].bias:resizeAs(netParamsTensor[{{},3}]):copy(netParamsTensor[{{},3}])
              Progress.finalLayerTernarized = true
            end

            local origWeightDim = nonTernMods[1].weight:size()
            if Progress.isModuleCNN(module) and nonTernMods[1].weight:dim() > 2 then
              nonTernMods[1].weight = nonTernMods[1].weight:reshape(nonTernMods[1].weight:size(1),nonTernMods[1].weight:size(2)*nonTernMods[1].weight:size(3)*nonTernMods[1].weight:size(4))
            end

            print('<Progress> ternarizing weights of ternary layer')
            local ternaryWeights = Progress.binarizeWeights(nonTernMods[1].weight, netParamsTensor[{{},1}], netParamsTensor[{{},2}])

            nonTernMods[1].weight = nonTernMods[1].weight:reshape(origWeightDim)
            ternaryWeights = ternaryWeights:reshape(origWeightDim)
            nonTernContainers[i].modules[j].weight = ternaryWeights
            print('<Progress> finished replacing layer with ternary layer')
          end
        end
      end
    end

    return module

end
