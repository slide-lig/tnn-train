------------------------------------------------------------------------
--[[ Custom ]]--
------------------------------------------------------------------------
local Custom, DataSource = torch.class("dp.Custom", "dp.DataSource")
Custom.isCustom = true

Custom._name = 'custom'
Custom._classes = _.range(0,31)
Custom._image_size = {2040, 1}
Custom._image_axes = 'bf'
Custom._feature_size = 2040

function Custom:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1],
      "Constructor requires key-value arguments")
   local args, load_all, input_preprocess, target_preprocess
   args, self._fracture_data, self._valid_ratio, self._train_file, self._test_file,
         self._data_path, self._download_url, load_all, input_preprocess,
         target_preprocess
      = xlua.unpack(
      {config},
      'Custom',
      'Custom dataset.' ..
      'Note: Train and valid sets are already shuffled.',
      {arg='fracture_data', type='number', default=1,
       help='proportion of data set to use.'},
      {arg='valid_ratio', type='number', default=1/6,
       help='proportion of training set to use for cross-validation.'},
      {arg='train_file', type='string', default='train.th7',
       help='name of training file'},
      {arg='test_file', type='string', default='test.th7',
       help='name of test file'},
      {arg='data_path', type='string', default=dp.DATA_DIR,
       help='path to data repository'},
      {arg='download_url', type='string',
       default='https://some/online/train.th7',
       help='URL from which to download dataset if not found on disk.'},
      {arg='load_all', type='boolean',
       help='Load all datasets : train, valid, test.', default=true},
      {arg='input_preprocess', type='table | dp.Preprocess',
       help='to be performed on set inputs, measuring statistics ' ..
       '(fitting) on the train_set only, and reusing these to ' ..
       'preprocess the valid_set and test_set.'},
      {arg='target_preprocess', type='table | dp.Preprocess',
       help='to be performed on set targets, measuring statistics ' ..
       '(fitting) on the train_set only, and reusing these to ' ..
       'preprocess the valid_set and test_set.'}
   )

   if load_all then
      self:loadTrainValid()
      self:loadTest()
   end
   DataSource.__init(self, {
      train_set=self:trainSet(), valid_set=self:validSet(),
      test_set=self:testSet(), input_preprocess=input_preprocess,
      target_preprocess=target_preprocess
   })
end

function Custom:loadTrainValid()
   --Data will contain a tensor where each row is an example, and where
   --the last column contains the target class.
   local data = self:loadData(self._train_file, self._download_url)
   local dataSize = math.floor(data[1]:size(1) * self._fracture_data)
   -- train
   local start = 1
   local size = math.floor(dataSize*(1-self._valid_ratio))
   self:trainSet(
      self:createDataSet(
         data[1]:narrow(1, start, size), data[2]:narrow(1, start, size),
         'train'
      )
   )
   -- valid
   if self._valid_ratio == 0 then
      print"Warning : No Valid Set due to valid_ratio == 0"
      return
   end
   start = size+1
   size = dataSize-start+1
   self:validSet(
      self:createDataSet(
         data[1]:narrow(1, start, size), data[2]:narrow(1, start, size),
         'valid'
      )
   )
   return self:trainSet(), self:validSet()
end

function Custom:loadTest()
   local test_data = self:loadData(self._test_file, self._download_url)
   local dataSize = math.floor(test_data[1]:size(1) * self._fracture_data)
   self:testSet(
      self:createDataSet(
        test_data[1]:narrow(1, 1, dataSize), test_data[2]:narrow(1, 1, dataSize),
        'test'
      )
   )
   return self:testSet()
end

--Creates an Custom Dataset out of inputs, targets and which_set
function Custom:createDataSet(inputs, targets, which_set)

   -- class 0 will have index 1, class 1 index 2, and so on.
   targets:add(1)
   -- construct inputs and targets dp.Views
   local input_v, target_v = dp.DataView(), dp.ClassView()
   input_v:forward('bf', inputs)
   target_v:forward('b', targets)
   target_v:setClasses(self._classes)
   -- construct dataset
   local ds = dp.DataSet{inputs=input_v,targets=target_v,which_set=which_set}
   ds:ioShapes('bf', 'b')
   return ds
end

function Custom:loadData(file_name, download_url)
   local path = DataSource.getDataPath{
      name=self._name, url=download_url,
      decompress_file=file_name,
      data_dir=self._data_path
   }
   -- backwards compatible with old binary format
   local status, data = pcall(function() return torch.load(path, "ascii") end)
   if not status then
      return torch.load(path, "binary")
   end
   return data
end
