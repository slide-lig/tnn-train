torch = require 'torch'
pl = require 'pl'

ConfHandler={
  _valueGenLimit=1000,
  _confGenLimit=1000,
  _supports = {
    optim = {
      fn = {'sgd','asgd','adam','adamax','adadelta','adagrad','rmsprop','cmaes'}
    }
  }
}

function ConfHandler.GetNetFile(paths)
  for _,filepath in ipairs(paths) do
      if string.find(filepath,'.net$') ~= nil then
        return filepath
      end
  end
  return nil
end
-- given conf composite file, generate set of full confs
-- given conf parts files, generate full conf
function ConfHandler.GetFullConfs(paths)
  --
  -- given file set
  -- for all full confs
  --    categorize as flat or composite
  -- join any remainign parts to try make full conf
  --    verify is full
  --        categorize as flat or composite
  -- for each full composite
  --    decompose and mark as flats
  --
  -- return flats
  --

  local fullFlats = {}
  local fullComposites = {}
  local partConfs = {}

  -- categorize files into above sets, full (composite or not) or part configuration
  for i = 1,#paths do
    local filepath = paths[i]
    if string.find(filepath,'.conf$') ~= nil then
      local conf = ConfHandler.read(filepath)
      local confCompleteStatus = ConfHandler.ConfFileCompleteStatus(conf)

      if ConfHandler.isConfFull(confCompleteStatus) > 0 then
        if ConfHandler.isConfComposite(confCompleteStatus) > 0 then
          table.insert(fullComposites, conf)
        else
          table.insert(fullFlats, conf)
        end
      elseif ConfHandler.isConfPart(confCompleteStatus) > 0 then
        table.insert(partConfs, conf)
      end-- else not a conf
    end
  end

  -- join part confs, verify is full and categorize as composite or not
  if #partConfs > 0 then
    print("<ConfHandler.GetFullConfs> attempting to join [".. #partConfs .."] part configurations")
    local confFromParts = {}
    for i = 1,#partConfs do
      local partConf = partConfs[i]
      confFromParts = ConfHandler.mergeTbl(confFromParts,partConf)
    end
    local confCompleteStatus = ConfHandler.ConfFileCompleteStatus(confFromParts)
    if ConfHandler.isConfFull(confCompleteStatus) > 0 then
      if ConfHandler.isConfComposite(confCompleteStatus) > 0 then
        table.insert(fullComposites, confFromParts)
      else
        table.insert(fullFlats, confFromParts)
      end
    else
      -- parts don't make up a full conf, so use to overide all others
      print("<ConfHandler.GetFullConfs> warning, over-riding all configuration with part configuration")
      for i = 1,#fullFlats do
        fullFlats[i] = ConfHandler.mergeTbl(fullFlats[i],confFromParts)
      end
      for i = 1,#fullComposites do
        fullComposites[i] = ConfHandler.mergeTbl(fullComposites[i],confFromParts)
      end
    end
  end

  print("<ConfHandler.GetFullConfs> have found [".. #fullComposites .."] full composite confs, [".. #fullFlats .."] full flat confs")

  -- decompose composite confs
  for _,confComposite in pairs(fullComposites) do
    local flats = ConfHandler.decomposeConf(confComposite)
    print("<ConfHandler.GetFullConfs> composite conf has become ["..#flats.."] flat confs")
    for _,flat in pairs(flats) do
      table.insert(fullFlats,flat)
    end
  end

  print("<ConfHandler.GetFullConfs> have found [".. #fullFlats .."] configurations in total")


  for i = 1,#fullFlats do
    fullFlats[i] =  DataHandler.genConfHash(fullFlats[i]) -- generate uuid for conf
    fullFlats[i] =  ConfHandler.compileDstDirValue(fullFlats[i]) -- compile dstDir value
  end

  return fullFlats
end

function ConfHandler.compileDstDirValue(conf)

  if conf.main and conf.main.dstDir then
    local codedDir = conf.main.dstDir
    for keychain in string.gmatch(codedDir, "%%(.-)%%") do
      local replaceVal = ConfHandler.tryGetVal(conf,keychain) or keychain -- default just put keychain back in without %%
      conf.main.dstDir = string.gsub(conf.main.dstDir, "%%"..keychain.."%%", tostring(replaceVal))
    end
  end

  return conf
end

function ConfHandler.decomposeConf(conf)
  local flatConfs = {}
  local compositeConfs = {}
  table.insert(compositeConfs, conf)

  while #compositeConfs > 0 and #flatConfs < ConfHandler._confGenLimit do

    for compConfKey,compConfVal in pairs(compositeConfs) do

      local kc,vs = ConfHandler.getFirstCompositeKeyValue(compConfVal)
      if kc ~= nil then
        -- duplicate conf #vs times
        -- set key value according to set
        -- if one of set is composite
        --      put set into compositeConfs
        --      else put set into flatConfs
        for _,v in pairs(vs) do
          local copy = ConfHandler.copy(compConfVal)
          copy = ConfHandler.write(copy,kc,v)
          if ConfHandler.getFirstCompositeKeyValue(copy) ~= nil then
            table.insert(compositeConfs, copy) -- put back in list
          else
            table.insert(flatConfs, copy)
          end
        end

      else -- not a composite, in the wrong set
        table.insert(flatConfs, compConfVal)
      end

      compositeConfs[compConfKey] = nil -- take out of list

    end
  end

  return flatConfs
end

function ConfHandler.ConfFileCompleteStatus(conf)
  -- file can declare
  -- arch  [1000]
  -- input [0100]
  -- run   [0010]
  -- main  [0001] optional
  -- confCompleteStatus indicates conf exists (1) and if it contains composite values (2)
  --
  local confCompleteStatus = {0,0,0,0}
  if type(conf) ~= 'table' then return confCompleteStatus end

  -- arch conf
  if conf.arch ~= nil then
    if type(conf.arch) == 'table' then
      if conf.arch.modelArch ~= nil then
        confCompleteStatus[1] = 1 -- [1000]
        if ConfHandler.getFirstCompositeKeyValue(conf.arch) ~= nil then
          confCompleteStatus[1] = 2 -- [2000]
        end
      end

    end
  end
  -- input conf
  if conf.input ~= nil then
    if type(conf.input) == 'table' then
        if conf.input.dataset ~= nil then
          confCompleteStatus[2] = 1 -- [0100]
          if ConfHandler.getFirstCompositeKeyValue(conf.input) ~= nil then
            confCompleteStatus[2] = 2 -- [0200]
          end
        end
    end
  end
  -- run conf
  if conf.run ~= nil then
    if type(conf.run) == 'table' then
      if conf.run.randseed ~= nil or conf.run.optim.fn ~= nil then
        confCompleteStatus[3] = 1 -- [0010]
        if ConfHandler.getFirstCompositeKeyValue(conf.run) ~= nil then
          confCompleteStatus[3] = 2 -- [0020]
        end
      end
    end
  end
  -- main conf
  if conf.main ~= nil then
    if type(conf.main) == 'table' then
      if conf.main.dstDir ~= nil then
        confCompleteStatus[4] = 1 -- [0001]
        if ConfHandler.getFirstCompositeKeyValue(conf.main) ~= nil then
          confCompleteStatus[4] = 2 -- [0002]
        end
      end
    end
  end
  return confCompleteStatus
end

function ConfHandler.isConfComposite(status)
  if (status[1]>1) or (status[2]>1) or (status[3]>1) then return 1 end
  return 0
end
function ConfHandler.isConfFull(status)
  if status[1]>0 and status[2]>0 and status[3]>0 then return 1 end -- arch,input&run are required to be considered full
  return 0
end
function ConfHandler.isConfPart(status)
  if status[1]>0 or status[2]>0 or status[3]>0 or status[4]>0 then return 1 end-- only one of arch,input,run&main is required to be considered part
  return 0
end

function ConfHandler.getFirstCompositeKeyValue(conf)
  --
  -- for all key values
  --    if value is composite
  --        return key and value set
  -- return nil
  --
  for k,v in pairs(conf) do
    if type(v) == 'table' then -- need to recurse
      local ck,cv = ConfHandler.getFirstCompositeKeyValue(v)
      if ck ~= nil then table.insert(ck,1,k) return ck,cv end
    else
      local isComposite, valueSet = ConfHandler.isValueComposite(v)
      if isComposite > 0 then local ck={} table.insert(ck,k) return ck,valueSet end
    end
  end
  return nil
end

function ConfHandler.isValueComposite(value)
  -- key=value, where value has the form {'val1','val2',...,'valn'} or {'val1':'step':'valn'}
  local isComposite = 0
  local valueSet = {}
  -- is composite?
  if type(value) ~= 'string' or string.match(value, "{(.+)}") == nil then return isComposite, valueSet end
  isComposite = 1
  -- what type?
  local val1, step, valn = string.match(value, "{(.+):(.+):(.+)}")
  if val1 ~= nil then
    -- generator => {'val1':'step':'valn'}
    val1 = tonumber(val1)
    step = tonumber(step)
    valn = tonumber(valn)
    if val1 == nil or step == nil or valn == nil then error("<ConfHandler.isValueComposite> integer conversion failed for composite [".. value .."]") end
    local curVal = val1

    if val1 > valn then
      while curVal >= valn and ConfHandler._valueGenLimit >= #valueSet do
        table.insert(valueSet,curVal)
        curVal = curVal - step
      end
    else
      while curVal <= valn and ConfHandler._valueGenLimit >= #valueSet  do
        table.insert(valueSet,curVal)
        curVal = curVal + step
      end
    end
  else
    -- array => {'val1','val2',...,'valn'} where valx is alphanumeric
    for val in string.gmatch(value, "%w+%.?%w*") do
      val = tonumber(val) or val
      table.insert(valueSet,val)
    end
  end

  return isComposite, valueSet
end

function ConfHandler.isLineSkip(str)
  if string.len(str) <= 0 then return true end
  if  string.find(str,'^#') ~= nil then return true end
  return false
end

function ConfHandler.trim(s)
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end
function ConfHandler.removeComment(str)
  local i,j = string.find(str, ".*#")
  if i ~= nil and j~= nil then return ConfHandler.trim(string.sub(str,i,j-1)) end
  return ConfHandler.trim(str)
end
function ConfHandler.read(file)
  local tbl = {}
  local f = file
  if type(file) == "string" then
    f = assert(io.open(file,"r"))
  end

  for count = 1, math.huge do
    local line = f:read("*line")
    if line == nil then break end
    while ConfHandler.isLineSkip(line) do
      line = f:read("*line")
      if line == nil then break end
    end
    if line == nil then break end
    line = ConfHandler.removeComment(line)
    local kv = ConfHandler.decodeConf(line)
    if kv ~= nil then
      tbl = ConfHandler.mergeTbl(tbl, kv)
    end
  end

  f:close()

  return tbl
end

function ConfHandler.tryGetVal(conf,keychain) -- keychain must have sequential integer keys

  if type(keychain) == 'string' then
    local keysStr = keychain
    keychain = {}
    while string.find(keysStr, "%.") ~= nil do
      local dotIx = string.find(keysStr, "%.")
      keychain[#keychain+1] = string.sub(keysStr, 1, dotIx-1)
      keysStr = string.sub(keysStr, dotIx+1)
    end
    keychain[#keychain+1] = tonumber(keysStr) or keysStr
  end
  local pt = conf
  for _,v in ipairs(keychain) do
    --if pt[v] == nil then return nil end
    if type(pt[v]) == 'table' then
      pt = pt[v]
    else
      return pt[v]
    end
  end
  return nil
end

function ConfHandler.write(conf,keychain,value) -- keychain must have sequential integer keys
  local pt = conf
  for _,v in ipairs(keychain) do
    if pt[v] == nil then error("<ConfHandler.write> failed to find ["..v.."] link using keychain: "..tostring(keychain)) end
    if type(pt[v]) == 'table' then
      pt = pt[v]
    else
      pt[v] = value
    end
  end
  return conf
end

-- write a 'scrubbed' version of conf as CSV to filepath|conf.main.filepaths.conf
function ConfHandler.scrubCSVWriteKeys(conf, filepath)
  local filepath = filepath
  local f = assert(io.open(filepath,"a"), "<ConfHandler.scrubCSVWrite> failed to open [".. filepath .."] with append access")
  f:write(ConfHandler.encodeConf(ConfHandler.scrubTbl(conf),"",ConfHandler.extractKey,','))
  f:write("\n")
  f:flush()
  f:close()
end
-- write a 'scrubbed' version of conf as CSV to filepath|conf.main.filepaths.conf
function ConfHandler.scrubCSVWriteValues(conf, filepath)
  local filepath = filepath
  local f = assert(io.open(filepath,"a"), "<ConfHandler.scrubCSVWrite> failed to open [".. filepath .."] with append access")
  f:write(ConfHandler.encodeConf(ConfHandler.scrubTbl(conf),"",ConfHandler.extractValue,','))
  f:write("\n")
  f:flush()
  f:close()
end

function ConfHandler.extractKey(keyValue)
  return string.match(keyValue, "^(.+)=")
end
function ConfHandler.extractValue(keyValue)
  return string.match(keyValue, "=(.+)") or ""
end
-- write a 'scrubbed' version of conf to filepath|conf.main.filepaths.conf
function ConfHandler.scrubWrite(conf, filepath)
  local filepath = filepath or conf.main.filepaths.mainconf
  local f = assert(io.open(filepath,"w"), "<ConfHandler.scrubWrite> failed to open [".. filepath .."] with write access")
  f:write(ConfHandler.encodeConf(ConfHandler.scrubTbl(conf)))
  f:flush()
  f:close()
end

function ConfHandler.mergeTbl(tbl,kv)

  for k,v in pairs(kv) do

    if type(tbl[k]) ~= 'table' then
      tbl[k] = v -- dumping value stored in tbl
    else
      if type(v) ~= 'table' then
        table.insert(tbl[k],v) -- keyless entry :(
      else
        tbl[k] = ConfHandler.mergeTbl(tbl[k],v)
      end
    end

  end

  return tbl
end

function ConfHandler.TryBoolean(val)
  if type(val) == 'string' and tablex.find({'TRUE','FALSE'},val:upper()) ~= nil then
    return ('TRUE'==val:upper())
  end
  return val
end

function ConfHandler.decodeConf(str) -- product conf table
  local kvSplit = string.find(str, "=")

  local keysStr = string.sub(str, 1, kvSplit-1)-- split up to first '='
  if keysStr == nil then print("<ConfHandler.decodeConfKey> failed to decode conf key [".. str .."]") return end

  local keys = {}
  while string.find(keysStr, "%.") ~= nil do
    local dotIx = string.find(keysStr, "%.")
    keys[#keys+1] = string.sub(keysStr, 1, dotIx-1)
    keysStr = string.sub(keysStr, dotIx+1)
  end
  keys[#keys+1] = tonumber(keysStr) or keysStr

  local valueStr = string.sub(str,kvSplit+1)
  local value = tonumber(valueStr) or valueStr

  local kv = {}
  local current = kv
  local prev = kv
  for k,v in pairs(keys) do
    current[v] = {}
    prev = current
    current = current[v]
  end
  prev[keys[#keys]] = value

  return kv
end
function ConfHandler.encodeConf(conf, _prefix, _postProcessFn, _delim) -- produce string

  local resultStr = ""
  local nextStr = ""
  local postProcessFn = _postProcessFn or nil
  local delim = _delim or '\n'
  for k,v in tablex.sort(conf) do

    local prefix = _prefix or ""

    if type(v) == 'table' then
      prefix = prefix .. k .. '.'
      nextStr = ConfHandler.encodeConf(v,prefix,postProcessFn,delim)
    else
      nextStr = prefix .. k .. '=' .. v
      if postProcessFn ~= nil then
          nextStr = postProcessFn(nextStr)
      end
      nextStr = nextStr .. delim
    end

    resultStr = resultStr .. nextStr

  end

  return resultStr

end

-- replica without underscored keys
function ConfHandler.scrubTbl(tbl)
  local cleanTbl = {}
  for k,v in pairs(tbl) do

    if string.find(k,'^_') == nil then

      if type(v) == 'table' then
        cleanTbl[k] = ConfHandler.scrubTbl(v)
      else
        cleanTbl[k] = v
      end

    end

  end
  return cleanTbl
end

function ConfHandler.copy(obj)
  --https://gist.github.com/tylerneylon/81333721109155b2d244
  if type(obj) ~= 'table' then return obj end
  local res = setmetatable({}, getmetatable(obj))
  for k, v in pairs(obj) do res[ConfHandler.copy(k)] = ConfHandler.copy(v) end
  return res
end

function ConfHandler.IsOnlineDistortEn(opt)
  return opt.input.distort.count == 0 and (math.abs(opt.input.distort.translate) + math.abs(opt.input.distort.rotate) + math.abs(opt.input.distort.scale)) > 0
end

function ConfHandler.upgradeLegacyConf(conf)

  if conf.main.codehash == "77855987bb1ff507cfe580688cd4a77e1e403b0f" then
    local upgradeConf = conf

    -- deletes

    upgradeConf.arch.pauseLayers = nil

    -- inserts

    upgradeConf.run.randseed = upgradeConf.run.randseed or 1
    upgradeConf.input.scale = upgradeConf.input.scale or 0

    -- modifications/relocations

    if upgradeConf.arch.modelArch == "PROG" or upgradeConf.arch.modelArch == "SPROG" or upgradeConf.arch.modelArch == "SimpleProgressiveBin" then
      upgradeConf.arch.modelArch = "ProgressiveBinarization"
    end

    upgradeConf.arch.layerCount = (upgradeConf.arch.modelWidth+1) or 2
    upgradeConf.arch.neuronPerLayerCount = upgradeConf.arch.modelHeight or 250

    upgradeConf.main.verbose = upgradeConf.run.verbose or 3
    upgradeConf.main.stopEarly = upgradeConf.run.stopEarly or 0
    upgradeConf.main.threads = upgradeConf.run.threads or 4
    upgradeConf.main.cuda = upgradeConf.run.cuda or 0
    upgradeConf.main.debug = upgradeConf.run.debug or 0
    upgradeConf.main.dstDir = upgradeConf.run.dstDir or "results_%arch.modelArch%_%main.confhash%"
    upgradeConf.main.binPath = upgradeConf.run.binPath or ""
    upgradeConf.main.epochCount = upgradeConf.run.epochCount or 1000
    upgradeConf.main.logFileSize = upgradeConf.run.logFileSize or 100000000

    upgradeConf.run.verbose = nil
    upgradeConf.run.stopEarly = nil
    upgradeConf.run.threads = nil
    upgradeConf.run.cuda = nil
    upgradeConf.run.debug = nil
    upgradeConf.run.dstDir = nil
    upgradeConf.run.binPath = nil
    upgradeConf.run.epochCount = nil
    upgradeConf.run.logFileSize = nil

    return upgradeConf
  end

  return nil

end
