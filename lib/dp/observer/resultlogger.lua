------------------------------------------------------------------------
--[[ ResultLogger ]]--
-- Interface, Observer
-- Simple logger that prints a report every epoch
------------------------------------------------------------------------
local ResultLogger, parent = torch.class("dp.ResultLogger", "dp.Observer")
ResultLogger.isResultLogger = true

function ResultLogger:__init(save_dir)
   self._save_dir = save_dir or dp.SAVE_DIR
   parent.__init(self, {"optimizer:doneEpoch", "doneEpoch"}, {"doneOptimizerEpoch", "doneEpoch"})
end

function ResultLogger:setup(config)
   parent.setup(self, config)
   --concatenate save directory with subject id
   --local subject_path = self._subject:id():toPath()
   --self._save_dir = paths.concat(self._save_dir, subject_path)
   --self._log_dir = paths.concat(self._save_dir, 'log')
   --creates directories if required
   --paths.mkdir(self._log_dir)
   --assert(paths.dirp(self._log_dir), "Log wasn't created : "..self._log_dir)
   --dp.vprint(self._verbose, "ResultLogger: log will be written to " .. self._log_dir)
end

function ResultLogger:doneOptimizerEpoch(report)
  local filePath = paths.concat(self._save_dir, 'results.json')
  local tableKey = "epoch"
  local tableValue = report.epoch + 1 -- report cycles following doneBatchs but before doneEpoch
  local recordKey = "batches"
  local recordValue = {loss=report.tester.feedback.batch.loss,
                        learningRate=report.tester.feedback.batch.learningRate}

  if report.tester.feedback.batch ~= nil and tablex.size(report.tester.feedback.batch) > 0 then
    DataHandler.logJsonRecord(filePath, tableKey, tableValue, recordKey, recordValue)
  end
end

function ResultLogger:doneEpoch(report)
  local filePath = paths.concat(self._save_dir, 'results.json')
  local tableKey = "epoch"
  local tableValue = report.epoch
  local recordKey = "results"
  local recordValue = {train=report.optimizer.feedback.confusion.accuracy,
                      valid=report.validator.feedback.confusion.accuracy,
                      test=report.tester.feedback.confusion.accuracy}

  DataHandler.logJsonRecord(filePath, tableKey, tableValue, recordKey, recordValue)
end
