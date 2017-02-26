
local CacheLoader = torch.class("dp.CacheLoader")

function CacheLoader:__init()
  self.cacheDir = paths.concat(dp.DATA_DIR,"datacache")
  assert(paths.dir(self.cacheDir) or paths.mkdir(self.cacheDir),"<CacheLoader> failed to make datacache [".. self.cacheDir .."]")
end

function CacheLoader:load(dsName, dsConf, cacheOnMiss)
  cacheOnMiss = cacheOnMiss or 0
  local ds
  local cacheName = CacheLoader:genCacheName(dsName, dsConf)
  local cacheFilePath = paths.concat(self.cacheDir, cacheName)

  if paths.filep(cacheFilePath) then
    print("<CacheLoader> loading "..dsName.." from disk cache at ["..cacheFilePath.."]")
    return torch.load(cacheFilePath)
  else
    print("<CacheLoader> missed "..dsName.." in disk cache")
    ds = dp[dsName](dsConf)
    if ds and cacheOnMiss > 0 then
      print("<CacheLoader> saving "..dsName.." to disk cache")
      torch.save(cacheFilePath, ds)
    end
  end

  return ds
end

function CacheLoader:genCacheName(dsName, dsConf)
  local cacheName = dsName .. "_prep"
  tablex.map(function(x) cacheName = cacheName .. tostring(x)  end, dsConf.input_preprocess)
  cacheName = cacheName .."_fract".. tostring(dsConf.fracture_data)
  cacheName = cacheName .."_valid".. tostring(dsConf.valid_ratio)
  cacheName = cacheName .. ".ds"
  return cacheName
end
