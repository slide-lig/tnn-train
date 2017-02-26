------------------------------------------------------------------------
--[[ SvhnStd ]]--
-- A color-image set of 10 digits.
------------------------------------------------------------------------
local SvhnStd, DataSource = torch.class("dp.SvhnStd", "dp.DataSource")
SvhnStd.isSvhnStd = true

SvhnStd._name = 'svhnstd'
SvhnStd._image_size = {3, 32, 32}
SvhnStd._image_axes = 'bchw'
SvhnStd._feature_size = 3*32*32
SvhnStd._classes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

function SvhnStd:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1],
      "Constructor requires key-value arguments")
   local args, load_all, input_preprocess, target_preprocess
   args, self._fracture_data, self._valid_ratio, self._train_file, self._test_file,
         self._data_path, self._scale, self._binarize, self._shuffle,
         self._download_url, load_all, input_preprocess,
         target_preprocess
      = xlua.unpack(
      {config},
      'SvhnStd',
      'Handwritten digit classification problem.' ..
      'Note: Train and valid sets are already shuffled.',
      {arg='fracture_data', type='number', default=1,
       help='proportion of data set to use.'},
      {arg='valid_ratio', type='number', default=1/6,
       help='proportion of training set to use for cross-validation.'},
      {arg='train_file', type='string', default='housenumbers/train_32x32.t7',
       help='name of train_file'},
      {arg='test_file', type='string', default='housenumbers/test_32x32.t7',
       help='name of test_file'},
      {arg='data_path', type='string', default=dp.DATA_DIR,
       help='path to data repository'},
      {arg='scale', type='table',
       help='bounds to scale the values between. [Default={0,1}]'},
      {arg='binarize', type='boolean',
       help='binarize the inputs (0s and 1s)', default=false},
      {arg='shuffle', type='boolean',
       help='shuffle different sets', default=true},
      {arg='download_url', type='string',
       default='http://torch7.s3-website-us-east-1.amazonaws.com/data/svhn.t7.tgz',
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
   if (self._scale == nil) then
      self._scale = {0,1}
   end
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

function SvhnStd:loadTrainValid()
   --Data will contain a tensor where each row is an example, and where
   --the last column contains the target class.
   local data = self:loadData(self._train_file, self._download_url)
   local dataSize = math.floor(data.X:size(1) * self._fracture_data)

   if self._shuffle then
     local indices = torch.randperm(data.X:size(1)):long()
     data.X = data.X:index(1, indices) -- shuffle inputs
     data.y = data.y:index(1, indices) -- shuffle targets
   end

   -- train
   local start = 1
   local size = math.floor(dataSize*(1-self._valid_ratio))
   self:trainSet(
      self:createDataSet(
         data.X:narrow(1, start, size), data.y:narrow(1, start, size),
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
         data.X:narrow(1, start, size), data.y:narrow(1, start, size),
         'valid'
      )
   )
   return self:trainSet(), self:validSet()
end

function SvhnStd:loadTest()
   local test_data = self:loadData(self._test_file, self._download_url)
   local dataSize = math.floor(test_data.X:size(1) * self._fracture_data)
   self:testSet(
      self:createDataSet(test_data.X:narrow(1, 1, dataSize), test_data.y:narrow(1, 1, dataSize), 'test')
   )
   return self:testSet()
end

--Creates an SvhnStd Dataset out of inputs, targets and which_set
function SvhnStd:createDataSet(inputs, targets, which_set)
   if self._binarize then
      DataSource.binarize(inputs, 128)
   end
   if self._scale and not self._binarize then
      DataSource.rescale(inputs, self._scale[1], self._scale[2])
   end
   -- construct inputs and targets dp.Views
   local input_v, target_v = dp.ImageView(), dp.ClassView()
   input_v:forward(self._image_axes, inputs)
   target_v:forward('b', targets)
   target_v:setClasses(self._classes)
   -- construct dataset
   local ds = dp.DataSet{inputs=input_v,targets=target_v,which_set=which_set}
   ds:ioShapes(SvhnStd._image_axes, 'b')
   return ds
end

function SvhnStd:loadData(file_name, download_url)
   local path = DataSource.getDataPath{
      name=self._name, url=download_url,
      decompress_file=file_name,
      data_dir=self._data_path
   }
   -- backwards compatible with old binary format
   local status, data = pcall(function() return torch.load(path, "ascii") end)
   if not status then
      data = torch.load(path, "binary")
   end
   data.X = data.X:float() -- was ByteTensor
   data.y = data.y:squeeze():int() -- was DoubleTensor
   return data
end
