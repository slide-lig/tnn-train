-- plot given json file

dkjson = require 'dkjson'
torch = require 'torch'
gnuplot = require 'gnuplot'

DataPlotter = {}

local plotData = {}
local dirName = ""
local dataClass = "Accuracy"
local dataPeriod = "Epoch"
local dirPath = ""

for i,v in ipairs(arg) do
  local filePath = v
  local pathTable = string.split(filePath,"[\\/]")
  local filename = pathTable[#pathTable]
  if dirPath == "" then dirPath = string.match(filePath, "[\\/].*[\\/$]") end
  if dirName == "" and #pathTable > 1 then dirName = pathTable[#pathTable-1] end

  io.input(v)

  local rawData = io.read("*all")
  local jsonData = dkjson.decode(rawData)

  local trainData = {}
  local validData = {}


  for k=1,#jsonData do
    table.insert(trainData,jsonData[k]["results"]["train"])
    table.insert(validData,jsonData[k]["results"]["valid"])
  end

  local dataSeriesLbl = string.gsub(pathTable[#pathTable-1],"[_-]"," ") -- filename -- TODO need to identify what is unique about a data series
  if #jsonData > 1 then table.insert(plotData,{dataSeriesLbl.."--train",torch.Tensor(trainData), '-'}) end
  if #jsonData <= 1 then table.insert(plotData,{dataSeriesLbl.."--train",torch.Tensor(trainData), '+'}) end
  if #jsonData > 1 then table.insert(plotData,{dataSeriesLbl.."--valid",torch.Tensor(validData), '-'}) end
  if #jsonData <= 1 then table.insert(plotData,{dataSeriesLbl.."--valid",torch.Tensor(validData), '+'}) end
end


gnuplot.figure(1)
gnuplot.xlabel(dataPeriod)
gnuplot.ylabel(dataClass)
gnuplot.title(string.gsub(dirName,"[_-]"," "))

gnuplot.plot(plotData)

function DataPlotter.exportPlot(dirPath, fileType, dataPeriod, dataClass, plotTitle, plotData)

  fileName = paths.concat(dirPath,"plot" .. os.date("_%Y%m%d_%H%M%S"))

  if      fileType == "eps" then gnuplot.epsfigure(fileName..".eps")
  elseif  fileType == "pdf" then gnuplot.pdffigure(fileName..".pdf")
  elseif  fileType == "fig" then gnuplot.figprint(fileName..".fig")
  elseif  fileType == "svg" then gnuplot.svgfigure(fileName..".svg")
  else                           gnuplot.pngfigure(fileName..".png") -- PNG default
  end

  print(fileName)

  gnuplot.xlabel(dataPeriod)
  gnuplot.ylabel(dataClass)
  gnuplot.title(plotTitle)
  gnuplot.plot(plotData)

  gnuplot.plotflush()

end

DataPlotter.exportPlot(dirPath,"",dataPeriod,dataClass,string.gsub(dirName,"[_-]"," "),plotData)
