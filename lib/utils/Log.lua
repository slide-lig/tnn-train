require 'utils.List'

Log = {
  file=nil,
  filepath="",
  filename="",
  maxSize=10000,
  rotating=false,
  buffer = List.new(1000)
}

function Log.write(data)
  print(data)
  if Log.file == nil then List.pushlast(Log.buffer, data) return  end
  if Log.needRotate() and not Log.rotating then Log.rotate() end

  while List.count(Log.buffer) > 0 do
      Log.file:write(tostring(List.popfirst(Log.buffer)).."\n")
  end

  Log.file:write(tostring(data).."\n")
  Log.file:flush()
end

function Log.start(filepath, opt)
  local pathTable = string.split(filepath,"[\\/]")
  local filename = pathTable[#pathTable]
  local dirpath = string.match(filepath, "[\\/].*[\\/$]")

  Log.filepath = filepath
  Log.filename = filename
  Log.dirpath = dirpath
  Log.maxSize = opt.main.logFileSize or 10000000 -- 10MB

  Log.file = assert(io.open(tostring(Log.filepath),'a+'))
  Log.write("<Log.start> starting new log [".. os.date("%Y-%m-%d %H:%M:%S") .."]")
end

function Log.finish()
  Log.write("<Log.finish> end of logfile")
  Log.file:flush()
  Log.file:close()
end

function Log.needRotate()
  return Log.size() > Log.maxSize
end

function Log.rotate()
  Log.rotating = true
  Log.write("<Log.rotate> rotating logfile...")
  Log.finish()
  -- TODO propagate existing logs (including just saved file)
  Log.start()
  Log.rotating = false
end

function Log.size()
  local current = Log.file:seek()      -- get current position
  local size = Log.file:seek("end")    -- get file size
  Log.file:seek("set", current)        -- restore position
  return size
end
