torch = require 'torch'
dkjson = require 'dkjson'

DataHandler = {}

function DataHandler.weight_binarize(data, bin_type)
  local bindata = data:clone() -- non-destructive
  if bin_type == 'stoch' then

    bindata:add(1):div(2):clamp(0,1) -- hard-sigmoid
    if type(p.cbernoulli) == 'function' then
      bindata:cbernoulli():mul(2):add(-1) -- stochastic bipolarization
    else
      bindata:apply(function(x) return (torch.bernoulli(x)*2)-1 end) -- stochastic bipolarization
    end


  elseif bin_type == 'det' then

    if type(bindata.snapd) == 'function' then
      bindata:snapd(0, -1, 0, 1,1)
    else
      bindata:apply(function(x) if x >= 0 then return 1 end return -1 end)
    end
  end

  return bindata
end

function DataHandler.mergeTensorFiles(srcDir,srcFilePattern)

  local tensorFiles = DataHandler.findAllFileMatch(srcDir, srcFilePattern)
  local finalTensorPath = nil
  local finalTensor = nil
  local partsTensor = nil
  for i,v in tablex.sortv(tensorFiles) do
    if finalTensorPath == nil then finalTensorPath = paths.concat(srcDir,v) end

    local nxtNartTensor = torch.load(paths.concat(srcDir,v))

    if finalTensor == nil then
      finalTensor = torch.Tensor(nxtNartTensor:typeAs(torch.Tensor()))
    elseif partsTensor == nil then
      partsTensor = torch.Tensor(nxtNartTensor:typeAs(torch.Tensor()))
    else
      partsTensor = torch.cat(partsTensor, nxtNartTensor:typeAs(torch.Tensor()), 1)
    end
  end

  if finalTensor ~= nil and partsTensor ~= nil then
    finalTensor = torch.cat(partsTensor, finalTensor, 1)
  end

  if finalTensor ~= nil then
    torch.save(finalTensorPath,finalTensor)
    if paths.filep(finalTensorPath) then
      for i,v in ipairs(tensorFiles) do
        if v ~= paths.basename(finalTensorPath) then
          os.remove(paths.concat(srcDir,v))
        end
      end
    end
    return
  end
  error("<DataHandler.mergeTensorFiles> no resulting tensor")
end


function DataHandler.findAllFileMatch(path, filenamePattern)

  print("<Progress> looking for ["..filenamePattern.."] in files of ["..path.."]")
  local matches = {}
  for f in paths.iterfiles(path) do
    print(f.."?")
    local matchFilename = string.match(f, filenamePattern) or ""
    if matchFilename ~= "" then
      print("<Progress> found ["..matchFilename.."]")
      table.insert(matches,matchFilename)
    end
  end
  print("<Progress> found "..#(matches).." files")
  return matches
end


function DataHandler.genConfHash(opt)
  local tempFilePath = '/tmp/lsr.main.conf.tmp'
  -- write conf to tmp
  ConfHandler.scrubWrite({arch=opt.arch,input=opt.input,run=opt.run}, tempFilePath)
  -- run md5/confhash.sh on conf and store result into it
  os.execute("../confhash.sh "..tempFilePath.." | awk '{print \"main.confhash=\"$0}' >> "..tempFilePath)
  -- read result and merge
  opt = ConfHandler.mergeTbl(ConfHandler.read(tempFilePath),opt)
  -- cleanup
  os.remove(tempFilePath)
  -- return resulting merge
  return opt
end

function DataHandler.isNetFile(filepath)
  return string.find(filepath,'.net$') ~= nil
end

function DataHandler.getCurrentGitCommit()

  local f = io.open("../.git/HEAD","r")

  local headRef = "refs/heads/master"
  if f~=nil then
    local line = f:read("*line")
    for i in string.gmatch(line, "%S+") do
      headRef = i
    end
    f:close()
  end


  f = io.open(paths.concat("../.git/",headRef),"r")

  local ref = ""
  if f~=nil then
    ref = f:read("*line")
    f:close()
  end

  return ref

end

function DataHandler.splitTrainValidSet(dataPath, fraction, verbose)

    -- read dataset
    -- determine class proportions via histogram
    -- randomly select out valid set
    -- confirm proportions are maintained
    -- repeat if neccessary

    local dataset = torch.load(dataPath) -- assuming binary format

    local hist = {}
    local datasetSize = dataset.labels:squeeze():size()[1]
    local trainsetSize= datasetSize*fraction
    local validsetSize= datasetSize - trainsetSize

    dataset.labels:apply(function(x) if hist[x] == nil then hist[x] = 0 end hist[x] = hist[x] + 1 end)


    local trainset = {}
    local validset = {}

    if dataset.data:dim() == 2 then
      trainset['data'] = torch.Tensor(trainsetSize, dataset.data:size(2))
      validset['data'] = torch.Tensor(validsetSize, dataset.data:size(2))
    elseif dataset.data:dim() == 3 then
      trainset['data'] = torch.Tensor(trainsetSize, dataset.data:size(2), dataset.data:size(3))
      validset['data'] = torch.Tensor(validsetSize, dataset.data:size(2), dataset.data:size(3))
    end

    trainset['labels'] = torch.Tensor(trainsetSize,1)
    validset['labels'] = torch.Tensor(validsetSize,1)

    local retrySplit = true
    local retryCount = 0
    while retrySplit and retryCount < 3 do

      local randomIx = torch.randperm(datasetSize)

      retryCount = retryCount + 1
      if verbose > 0 then Log.write("<DataHandler.splitTrainValidSet> attempt #".. retryCount .." to split dataset and maintain approx class distribution") end

      for i=1,datasetSize do
        if i <= trainsetSize then
          trainset['data'][{i}] = dataset.data[{{randomIx[i]}}]
          trainset['labels'][{i}] = dataset.labels[{{randomIx[i]}}]
        else
          validset['data'][{i-trainsetSize}] = dataset.data[randomIx[i]]
          validset['labels'][{i-trainsetSize}] = dataset.labels[randomIx[i]]
        end
      end

      local validHist = {}

      validset.labels:apply(function(x) if validHist[x] == nil then validHist[x] = 0 end validHist[x] = validHist[x] + 1 end)

      retrySplit = false

      for i,v in ipairs(hist) do
        -- check if each class count is within 10% of original
        if validHist[i]  / (1 - fraction) > hist[i] * 1.2 then retrySplit = true end
        if validHist[i]  / (1 - fraction) < hist[i] * 0.8 then retrySplit = true end
      end

      if verbose > 0 and retrySplit == true then Log.write("<DataHandler.splitTrainValidSet> failed to maintain approx class distribution") end

    end

    if retrySplit == true then trainset = nil end
    if retrySplit == true then validset = nil end

    return trainset, validset

end

function DataHandler.genClasslist(dataset)

  assert(dataset.labels ~= nil, "<DataHandler.genClasslist> failed to find labels in dataset")

  local tblClasslist = {}
  local sortedLabels = torch.sort(dataset.labels:squeeze(), 1)
  sortedLabels:apply(function(x) tblClasslist[x] = tostring(x) end)

  return tblClasslist

end

function DataHandler.mat2t7(matfile)
  matio = require 'matio'
  local dataset = matio.load(matfile)
  local t7dataset = {}

  for k,v in pairs(dataset) do
    if string.find(k:lower(), 'data') ~= nil then
      t7dataset['data'] = v:squeeze()
    end
    if string.find(k:lower(), 'label') ~= nil then
      t7dataset['labels'] = v:squeeze()
    end
  end

  assert(t7dataset.data ~= nil, "<DataHandler.mat2t7> failed to find data in matlab file")
  assert(t7dataset.labels ~= nil, "<DataHandler.mat2t7> failed to find labels in matlab file")

  return t7dataset

end


-- fns of previous DataLoader
function DataHandler.findInTable(f, l) -- find element v of l satisfying f(v)
  for i, v in ipairs(l) do
    if f(v) then
      return i,v
    end
  end
  return nil
end
-- (2,3) returns [0,1,0]
function DataHandler.vectorizeVal(val, vecWidth)
  -- TODO confirm val is either 1x1 or Mx1
  local vecHeight = val:size(1)
  local vec = torch.ByteTensor( vecHeight, vecWidth ):fill(0)
  for i = 1,vecHeight do
    vec[ { i , val[{i,1}] }] = 1
  end
  return vec
end

function DataHandler.oneHotEncode(labels, dataSize, classlist)
 local newlabels = torch.ByteTensor( dataSize, #classlist ):fill(0)
 for i = 1,dataSize do
   local classIx,_ = DataHandler.findInTable(function(v) return v == tostring(labels[{i,1}]) end,classlist)
   assert(classIx~= nil, "<DataHandler.oneHotEncode> failed to find value in classlist")
   newlabels[ { i , classIx }] = 1
 end

 return newlabels
end

function DataHandler.loadDataset(fileName, fractionData, classlist)

 local f = torch.load(fileName)
 local data = f.data:type(torch.getdefaulttensortype())
 local labels = f.labels

 local classSize = #classlist
 local nExample = f.data:size(1)
 local maxLoad = nExample * fractionData

 if maxLoad and maxLoad > 0 and maxLoad < nExample then
   nExample = maxLoad
 end

 Log.write('<DataHandler.loadDataset> loading ' .. nExample .. ' examples from '..fileName)
 data = data[{{1,nExample}}]
 labels = labels[{{1,nExample}}]

 local newlabels = DataHandler.oneHotEncode(labels,nExample,classlist)

 local dataset = {}
 dataset.data = data
 dataset.labels = newlabels

  function dataset:size()
     return (#dataset.data)[1]
  end

  local labelvector = torch.zeros(classSize)

  setmetatable(dataset, {__index = function(self, index)
          local input = self.data[index]
          local class = self.labels[index]
          local label = labelvector:zero()
          label[class] = 1
          local example = {input, label}
                                   return example
  end})

  return dataset, nExample
end

-- fns of previous DataPrinter
function DataHandler.readJsonData(filepath)

  local f = io.open(filepath,"r")

  local jsonData = nil
  if f~=nil then
    jsonData = dkjson.decode(f:read("*all"))
    f:close()
  end

  if jsonData.data ~= nil and type(jsonData.data) == 'table' then
    jsonData.data = tablex.map(function(x) return tonumber(x) or x end, jsonData.data)
  end

  return jsonData.data

end

function DataHandler.readJsonRecord(filepath, recordKey, subRecordKey)

  local f = io.open(filepath,"r")

  local jsonData = nil
  if f~=nil then
    jsonData = dkjson.decode(f:read("*all"))
    f:close()
  end

  local results = {}
  if jsonData ~= nil and type(jsonData) == 'table' then
    for _,v in pairs(jsonData) do
      if v[recordKey] ~= nil then
        if type(v[recordKey]) == 'table' and v[recordKey][subRecordKey] ~= nil then
          table.insert(results,v[recordKey][subRecordKey])
        else
          table.insert(results,v[recordKey])
        end
      end
    end
  end

  return results

end
function DataHandler.stringifyTable(tab)
  local jsonData = dkjson.decode("{}")

  for k,v in pairs(tab) do
    jsonData[k] = tostring(v)
  end

  return jsonData
end
-- deprecated, replace by logJsonEpochTbl
function DataHandler.logJson(filePath, class, period, index, value)

      local f = io.open(filePath,"r")

      local jsonData = nil
      if f~=nil then
        jsonData = dkjson.decode(f:read("*all"))
        f:close()
      end

      local f = assert(io.open(filePath,"w+"))

      local template = "{ \
        \"class\":\"\",       \
        \"period\":\"\",      \
        \"data\": {}          \
      }"

      if jsonData == nil then jsonData = dkjson.decode(template) end

      jsonData["class"] = class
      jsonData["period"] = period

      if type(value) == 'table' then
          jsonData["data"][index] = DataHandler.stringifyTable(value)
      else
        jsonData["data"][index] = value
      end
      sequential = {}    -- new array.. this is so stupid
      for i=1, math.max(1000,tonumber(index) or 1000) do
        sequential[i] = tostring(i)
      end
      f:write(dkjson.encode(jsonData, {indent=true, keyorder=sequential}))
      f:flush()
      f:close()

      collectgarbage()
end


function DataHandler.logJsonRecord(filePath, tableKey, tableValue, recordKey, recordValue) -- tableKey/tableValue identifies where to hang the record

      local f = io.open(filePath,"r")

      local jsonData = nil
      if f~=nil then
        jsonData = dkjson.decode(f:read("*all"))
        f:close()
      end

      local f = assert(io.open(filePath,"w+"))

      if jsonData == nil then jsonData = dkjson.decode("[]") end
      local cmpTableVal = function(a,b) return a[tableKey]==b end
      local ix = tablex.find_if(jsonData,cmpTableVal,tableValue)

      if ix ~= nil then
        jsonData[ix][recordKey] = recordValue
      else
        local newRecord = {}
        newRecord[tableKey] = tableValue
        newRecord[recordKey] = recordValue
        table.insert(jsonData,newRecord)
      end

      sequential = {}    -- new array.. this is so stupid
      for i=1, math.max(1000,tonumber(index) or 1000) do
        sequential[i] = i
      end
      f:write(dkjson.encode(jsonData, {indent=true, keyorder=sequential}))
      f:flush()
      f:close()

      collectgarbage()
end

-- fns of previous DataProcessor


function DataHandler.prepareDataset(dataset, setsize, transpose)
  local input  = dataset.data:resize( (#dataset.data)[1] , (#dataset.data)[2]*(#dataset.data)[2]  )
  local labels = dataset.labels
  if transpose then
    input = input:t()
    labels = labels:t()
  end
  local x = {input , labels}
  function x:size() return setsize end

  return x
end


function DataHandler.normalize(data, mean_, std_)
  local mean = mean_ or data:view(data:size(1), -1):mean(1)
  local std = std_ or data:view(data:size(1), -1):std(1, true)
  for i=1,data:size(1) do
    data[i]:add(-mean[1][i])
    if std[1][i] > 0 then
      tensor:select(2, i):mul(1/std[1][i])
    end
  end
  return data, mean, std
end

function DataHandler.normalizeGlobal(data, mean_, std_)
  local std = std_ or data:std()
  local mean = mean_ or data:mean()
  data:add(-mean)
  data:mul(1/std)
  return data, mean, std
end

function DataHandler.scale(data)
  local max = torch.max(data)
  data:mul(1/max)
  return data
end

function DataHandler.getBinarizeThreshold(data, rangeFraction)
  rangeFraction = rangeFraction or 0.5
  local min = torch.min(data)
  local max = torch.max(data)
  local range = max - min
  return math.floor((rangeFraction * range) + min)
end

function DataHandler.binarize(data, rangeFraction)
  if rangeFraction == 0 then return data end
  local threshold = DataHandler.getBinarizeThreshold(data, rangeFraction)
  data = data:apply(function(v) if v > threshold then return 1 else return 0 end end)
  return data
end


function DataHandler.bipolarize(data,type)
  if type == 2 then
    --  make all zero and positive values  1
    data[ data:ge(0) ] = 1;
    -- make all other values -1
    data[ data:lt(0) ] = -1;
  elseif type == 1 then
    --  make all positive values  1
    data[ data:gt(0) ] = 1;
    -- make all other values -1
    data[ data:le(0) ] = -1;
  else
    --  make all positive values  1
    data[ data:gt(0) ] = 1;
    -- make all negative values -1
    data[ data:lt(0) ] = -1;
  end

  return data
end

function DataHandler.zeroDataSize(data)
  if type(data) == 'table' then
    for i = 1, #data do
      data[i] = zeroDataSize(data[i])
    end
  elseif type(data) == 'userdata' then
    data = torch.Tensor():typeAs(data)
  end
  return data
end

function DataHandler.getNetSaveName(epoch, isBestNet)
  local isBestNet = isBestNet or 0
  if isBestNet > 0 then
    return 'best_epoch_' .. epoch .. '.net'
  elseif isBestNet < 0 then
      return 'first_epoch_' .. epoch .. '.net'
  end
  return 'last_epoch_' .. epoch .. '.net'
end
function DataHandler.getStateSaveName(epoch, isBestNet)
  return DataHandler.getNetSaveName(epoch, isBestNet)..'.state'
end
function DataHandler.getInputSaveName(epoch, isBestNet)
  return DataHandler.getNetSaveName(epoch, isBestNet)..'.input'
end
function DataHandler.getOutputSaveName(epoch, isBestNet)
  return DataHandler.getNetSaveName(epoch, isBestNet)..'.output'
end
function DataHandler.getRecordSaveName(epoch, isBestNet)
  return DataHandler.getNetSaveName(epoch, isBestNet)..'.record'
end

function DataHandler.saveModuleEpoch(module, path, epoch, opt, isBestNet)
  local filepath = paths.concat(path, DataHandler.getNetSaveName(epoch, isBestNet))
  DataHandler.saveModule(filepath, module, opt)
  return filepath
end

function DataHandler.saveModule(path,module,opt)
  if opt.main.cuda > 0 then
    module:float()
    if opt.main.cudnn ~= nil and opt.main.cudnn > 0 then
      cudnn.convert(module, nn)
    end
  end

  torch.save(path, module)
  collectgarbage()

  if opt.main.cuda > 0 then
    module:cuda()
    if opt.main.cudnn ~= nil and opt.main.cudnn > 0 then
      cudnn.convert(module, cudnn)
    end
  end
end

function DataHandler.saveModuleStateEpoch(module, path, epoch, opt, isBestNet)
  -- saving stateful layers
  local bnModules = module:findModules('nn.BatchNormalization')
  if #bnModules > 0 then
    local bnState = {}
    for k = 1, #bnModules do
      bnState[k] =  { ["weight"]=bnModules[k].weight,["bias"]=bnModules[k].bias,["running_mean"]=bnModules[k].running_mean, ["running_var"]=bnModules[k].running_var }
    end
    local filepath = paths.concat(path, DataHandler.getStateSaveName(epoch, isBestNet))
    torch.save(filepath,bnState)
  end

  local recMods = module:findModules('nn.Record')

  if #recMods > 0 then
    for k = 1, #recMods do
      if recMods[k].isRecording ~= nil and recMods[k].isRecording > 0 then
        local filepath = paths.concat(path, DataHandler.getRecordSaveName(epoch..'_'..k, isBestNet))
        torch.save(filepath,recMods[k].record)
      end
    end
  end

  collectgarbage()

end

function DataHandler.removeFiles(path, name)
  local filepath = paths.concat(path, name)
  os.execute('rm '..filepath)
end

function DataHandler.removeModuleEpoch(path, epoch)
  local filepath = paths.concat(path, DataHandler.getNetSaveName(epoch))
  os.remove(filepath)
  filepath = paths.concat(path, DataHandler.getStateSaveName(epoch))
  os.remove(filepath)
end

function DataHandler.findBestConf(dir)
  -- return 'main.conf' or first from sorted filename

  local confName = ""
  local filepaths = paths.dir(dir)
  local confpaths = tablex.filter(filepaths, function(f) return string.find(f,'.conf$') ~= nil end)
  if confpaths ~= nil and #confpaths > 0 then
    local mainConfIx = tablex.find(confpaths, 'main.conf')
    if mainConfIx ~= nil then
      confName = confpaths[mainConfIx]
    else
      for _,v in tablex.sortv(confpaths) do confName = v break end
    end
  end
  return confName
end

function DataHandler.findBestNet(dir, opt)

  local netName = ""
  local filepaths = paths.dir(dir)
  local netpaths = tablex.filter(filepaths, function(f) return string.find(f,'.net$') ~= nil end)
  if netpaths ~= nil and #netpaths > 0 then

    -- try exact match on last saved epoch
    local lastNetIx = nil
    if opt.main.ModelTrainer ~= nil and opt.main.ModelTrainer.lastSaveEpoch ~= nil then
      lastNetIx = tablex.find(netpaths, paths.basename(opt.main.ModelTrainer.lastSaveEpoch))
    end

    if lastNetIx ~= nil then
      netName = netpaths[lastNetIx]
    else
      for _,v in tablex.sortv(netpaths, DataHandler.compareNames) do netName = v end -- iterate til last
    end
  else
    Log.write("<DataHandler.findBestNet> failed to find .net in [".. dir .."]")
  end

  return netName
end

function DataHandler.compareNames(a,b)
  -- return true if a 'less than' b
  -- 'less than' -> shorter in legth, then by alphanumeric order
  if string.len(a) < string.len(b) then return true end
  return a < b
end

function DataHandler.loadExperiment(path)
  assert(paths.dir(path) == nil, "<DataHandler.loadExperiment> given directory to load as experiment [".. path .."]")
  assert(paths.filep(path), "<DataHandler.loadExperiment> can't find file [".. path .."]")

  xp = torch.load(path)
  return xp
end
function DataHandler.loadModule(path)

  assert(paths.dir(path) == nil, "<DataHandler.loadModule> given directory to load as module [".. path .."]")
  assert(paths.filep(path), "<DataHandler.loadModule> can't find file [".. path .."]")

  xp = torch.load(path)

  if torch.type(xp) == 'dp.Experiment' then
    return xp:model().module
  end

  -- path is either .net or other file

  local moduleFilePath = path
  local stateFilePath = moduleFilePath..".state"

  -- loading module

  local f = io.open(moduleFilePath,"r")
  if f==nil then Log.write("<DataHandler.loadModule> failed to open/load module file") return false end

  local loadModule = torch.load(moduleFilePath)

  -- loading module states

  local f = io.open(stateFilePath,"r")
  if f==nil then return loadModule end -- continue without state file, most likely never existed

  local loadState = torch.load(stateFilePath)

  local bnModules = loadModule:findModules('nn.BatchNormalization')
  for k = 1, #bnModules do
    Log.write("Loading BatchNormalization layer: ["..k.."]")
    bnModules[k].weight = loadState[k]["weight"]
    bnModules[k].bias = loadState[k]["bias"]
    bnModules[k].running_mean = loadState[k]["running_mean"]
    bnModules[k].running_var = loadState[k]["running_var"]
  end

  collectgarbage()

  return loadModule
end

function DataHandler.calcDataSize(tensor)
  if tensor:type() == 'torch.ByteTensor' then return 1*tensor:nElement() end
  if tensor:type() == 'torch.CharTensor' then return 1*tensor:nElement() end
  if tensor:type() == 'torch.ShortTensor' then return 2*tensor:nElement() end
  if tensor:type() == 'torch.IntTensor' then return 4*tensor:nElement() end
  if tensor:type() == 'torch.LongTensor' then return 4*tensor:nElement() end
  if tensor:type() == 'torch.FloatTensor' then return 4*tensor:nElement() end
  if tensor:type() == 'torch.DoubleTensor' then return 8*tensor:nElement() end
end

function DataHandler.writeFloats(tensor, filepath)
  require("struct")
  local fmt = ">!1f"
  local out = assert(io.open(filepath,'ab+'))
  t = tensor:reshape(tensor:nElement()):float()
  for i=1,t:size(1) do
    out:write(struct.pack(fmt, t[i]))
  end
  out:flush()
  out:close()
  out = nil
  collectgarbage()
end

function DataHandler.writeSignedChars(tensor, filepath)
  require("struct")
  local fmt = ">b"
  local out = assert(io.open(filepath,'ab+'))
  t = tensor:reshape(tensor:nElement()):type('torch.CharTensor')
  for i=1,t:size(1) do
    out:write(struct.pack(fmt, t[i]))
  end
  out:flush()
  out:close()
  out = nil
  collectgarbage()
end


function DataHandler.overwrite2DTensorToCSV(tensor, filepath)
  local out = assert(io.open(filepath,'w'))
  for i=1,tensor:size(1) do
    for j=1,tensor:size(2) do
      out:write(tostring(tensor[i][j]))
      if j~= tensor:size(2) then out:write(',') end
    end
    out:write('\n')
    out:flush()
  end

end

function DataHandler.write2DTensorToCSV(tensor, filepath)

  local byteLimit = 1000*1000*1000
  local reOpenFileBreak = math.floor(byteLimit / ( DataHandler.calcDataSize(tensor[1]) + tensor[1]:nElement())) -- adding 1 byte for ','
print("limit set at "..tostring(byteLimit).."b [~"..reOpenFileBreak.." rows]")
  local out = assert(io.open(filepath,'a+'))

  for i=1,tensor:size(1) do
    for j=1,tensor:size(2) do
      out:write(tostring(tensor[i][j]))
      if j~= tensor:size(2) then out:write(',') end
    end
    out:write('\n')
    out:flush()
    if i%reOpenFileBreak==0 then -- to prevent memory issues
      print("reached limit "..reOpenFileBreak)
      out:close()
      out = nil
      out = assert(io.open(filepath,'a+'))
    end
  end

end
function DataHandler.write1DTensorToCSV(tensor, filepath)

  local byteLimit = 1000*1000*1000
  local reOpenFileBreak = math.floor(byteLimit / ( DataHandler.calcDataSize(tensor) + tensor:nElement())) -- adding 1 byte for ','
print("limit set at "..tostring(byteLimit).."b [~"..reOpenFileBreak.." rows]")
  local out = assert(io.open(filepath,'a+'))

  for i=1,tensor:size(1) do
    out:write(tostring(tensor[i]))
    out:write('\n')
    out:flush()
    if i%reOpenFileBreak==0 then -- to prevent memory issues
      print("reached limit "..reOpenFileBreak)
      out:close()
      out = nil
      out = assert(io.open(filepath,'a+'))
    end
  end

end
