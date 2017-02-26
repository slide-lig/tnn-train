------------------------------------------------------------------------
--[[ EpochLogger ]]--
-- Interface, Observer
-- Simple logger that prints a report every epoch
------------------------------------------------------------------------
local EpochLogger, parent = torch.class("dp.EpochLogger", "dp.Logger")
EpochLogger.isEpochLogger = true

function EpochLogger:__init(save_dir)
   parent.__init(self, {"doneEpoch"})
end

function EpochLogger:setup(config)
   parent.setup(self, config)
end
function EpochLogger:doneEpoch(report)
  if report.validator ~= nil and report.validator.feedback.confusion.cm ~= nil then
    Log.write("Validation Confusion Matrix")
    Log.write(tostring(report.validator.feedback.confusion.cm))
  end
end
