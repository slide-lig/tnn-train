-------------------------------------------------
--[[ Gtsrb ]]--
-- http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
-- Image set with over 50k images in total, of over 40 classes.
-- Images range from 15x15 to 250x250 in size and contain one traffic sign each
-------------------------------------------------
local pl = (require 'pl.import_into')()

local Gtsrb, parent = torch.class("dp.Gtsrb", "dp.DataSource")
Gtsrb.isGtsrb = true
Gtsrb._name = 'gtsrb'
Gtsrb._image_axes = 'bchw'
Gtsrb._classes = _.range(1,43)

-- Private function declaration
local generate_dataset
local relocate_gtsrb_test_file

function Gtsrb:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1],
      "Constructor requires key-value arguments")
   local load_all, input_preprocess, target_preprocess

   self.args, self._fracture_data, self._valid_ratio, self._data_path,
   self._scale, self._img_size, self._download_train_url, self._download_test_url, self._download_test_gt_url, load_all,
   self._shuffle,self._balance,self._crop,
   input_preprocess, target_preprocess
      = xlua.unpack(
      {config},
      'gtsrb', nil,
      {arg='fracture_data', type='number', default=1,
       help='proportion of data set to use.'},
      {arg='valid_ratio', type='number', default=1/5,
        help='proportion of training set to use for cross-validation.'},
      {arg='data_path', type='string', default=dp.DATA_DIR,
        help='path to data repository'},
      {arg='scale', type='table',
        help='bounds to scale the values between'},
      {arg='img_size', type='table', default=32,
        help='height/width of the image following conversion to tensors'},
      {arg='download_train_url', type='string',
        default='http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip',
        help='URL from which to download training dataset if not found on disk.'},
      {arg='download_test_url', type='string',
          default='http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip',
          help='URL from which to download test dataset if not found on disk.'},
      {arg='download_test_gt_url', type='string',
          default='http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip',
          help='URL from which to download test dataset if not found on disk.'},
      {arg='load_all', type='boolean',
        help='Load all datasets : train, valid, test.', default=true},
      {arg='shuffle', type='boolean',
        help='shuffle train/valid data', default=false},
      {arg='balance', type='boolean',
        help='balance the class sample counts (will cause big reduction in sample count)', default=false},
      {arg='crop', type='boolean',
        help='crop datasets using ROI annotations', default=true},
      {arg='input_preprocess', type='table | dp.Preprocess',
        help='to be performed on set inputs, measuring statistics ' ..
        '(fitting) on the train_set only, and reusing these to ' ..
        'preprocess the valid_set and test_set.'},
      {arg='target_preprocess', type='table | dp.Preprocess',
        help='to be performed on set targets, measuring statistics ' ..
        '(fitting) on the train_set only, and reusing these to ' ..
        'preprocess the valid_set and test_set.'}
   )
   --TODO resolve these
   self._fracture_data=1
   self._valid_ratio=1/5
   ------

   if (self._scale == nil) then
      self._scale = {0,1}
   end
   self._image_size = {3, self._img_size, self._img_size}
   self._feature_size = 3*self._img_size*self._img_size
   if load_all then
      self:loadTrainValid()
      --self:loadTrain()
      --self:loadValid()
      self:loadTest()
   end
   parent.__init(self, {train_set=self:trainSet(),
                        valid_set=self:validSet(),
                        test_set=self:testSet(),
                        input_preprocess=input_preprocess,
                        target_preprocess=target_preprocess})
end

function Gtsrb:loadTrainValid()
  local train_data, valid_data = self:loadData(self._download_train_url, 'train')

  if self._shuffle then
    local indices = torch.randperm(train_data:size(1)):long()
    train_data = train_data:index(1, indices)
    indices = torch.randperm(valid_data:size(1)):long()
    valid_data = valid_data:index(1, indices)
  end

  self:trainSet(self:createDataSet(train_data, 'train'))
  self:validSet(self:createDataSet(valid_data, 'valid'))
  return self:trainSet(), self:validSet()
end

function Gtsrb:loadTrain()
   --Data will contain a tensor where each row is an example, and where
   --the last column contains the target class.
   local data = self:loadData(self._download_train_url, 'train')
   local dataSize = math.floor(data:size(1) * self._fracture_data)
   local size = math.floor(dataSize*(1-self._valid_ratio))
   local train_data = data:narrow(1, 1, size)
   self:trainSet(self:createDataSet(train_data, 'train'))
   return self:trainSet()
end

function Gtsrb:loadValid()
   data = self:loadData(self._download_train_url, 'train')
   local dataSize = math.floor(data:size(1) * self._fracture_data)
   if self._valid_ratio == 0 then
      print"Warning : No Valid Set due to valid_ratio == 0"
      return
   end
   local start = math.ceil(dataSize*(1-self._valid_ratio))
   local size = dataSize-start
   local valid_data = data:narrow(1, start, size)
   self:validSet(self:createDataSet(valid_data, 'valid'))
   return self:validSet()
end

function Gtsrb:loadTest()
   local test_data = self:loadData({self._download_test_url,self._download_test_gt_url}, 'test')
   local dataSize = math.floor(test_data:size(1) * self._fracture_data)
   self:testSet(self:createDataSet(test_data:narrow(1,1,dataSize), 'test'))
   return self:testSet()
end

function Gtsrb:createDataSet(data, which_set)
   local inputs = data:narrow(2, 1, self._feature_size):clone()
   inputs = inputs:type('torch.DoubleTensor')
   inputs:resize(inputs:size(1), unpack(self._image_size))
   if self._scale then
      parent.rescale(inputs, self._scale[1], self._scale[2])
   end
   --inputs:resize(inputs:size(1), unpack(self._image_size))
   local targets = data:select(2, self._feature_size+1):clone()
   -- class 0 will have index 1, class 1 index 2, and so on.
   targets:add(1)
   targets = targets:type('torch.DoubleTensor')
   -- construct inputs and targets dp.Views
   local input_v, target_v = dp.ImageView(), dp.ClassView()
   input_v:forward(self._image_axes, inputs)
   target_v:forward('b', targets)
   target_v:setClasses(self._classes)
   -- construct dataset
   local ds = dp.DataSet{inputs=input_v,targets=target_v,which_set=which_set}
   ds:ioShapes('bchw', 'b')
   return ds
end

function Gtsrb:loadData(download_url, which_set)

  local ppm_parent_path = which_set=="train" and "GTSRB/Final_Training/Images" or "GTSRB/Final_Test/Images"
  local expected_n_example =  which_set=="train" and 39209 or 12630
  local expected_n_feature = self._image_size[1]*self._image_size[2]*self._image_size[3]
  local t7_file = which_set=="train" and "train.t7" or "test.t7"
  local t7_file_valid = "valid.t7"
  t7_file = self._image_size[2] .. "x" .. self._image_size[3] .. "_" .. t7_file
  t7_file_valid = self._image_size[2] .. "x" .. self._image_size[3] .. "_" .. t7_file_valid
  local t7_path = paths.concat(self._data_path,self._name,t7_file)
  local t7_path_valid = paths.concat(self._data_path,self._name,t7_file_valid)


  if not pl.path.isfile(t7_path) then

    local path = nil
    if type(download_url) == 'table' then
      for _,dl_url in ipairs(download_url) do
        dp.DataSource.getDataPath{
          name=self._name, url=dl_url, data_dir=self._data_path,
          decompress_file="/cant/find/me/dummy"
        }
      end
    else
      dp.DataSource.getDataPath{
        name=self._name, url=download_url, data_dir=self._data_path,
        decompress_file="/cant/find/me/dummy"
      }
    end

    if which_set=="test" then
      dp.do_with_cwd(paths.concat(self._data_path,self._name),relocate_gtsrb_test_file)
      local data_set = generate_dataset(paths.concat(self._data_path,self._name,ppm_parent_path), self._image_size[2]) -- generate .t7 from images downloaded
      torch.save(t7_path, data_set)
    else
      local data_set_train, data_set_valid = generate_dataset_train_valid(paths.concat(self._data_path,self._name,ppm_parent_path), self._image_size[2]) -- generate .t7 from images downloaded
      torch.save(t7_path, data_set_train)
      torch.save(t7_path_valid, data_set_valid)
    end

    data_set = nil
    collectgarbage()

    if which_set=="test" then
      dp.do_with_cwd(paths.concat(self._data_path,self._name),function()
        if paths.dirp('GTSRB') then
          os.execute('rm -r GTSRB')
        end
      end)
    end


  end

  if which_set=="train" then
    local t_train = torch.load(t7_path)
    local n_example = t_train.data:size(1)
    local n_feature = t_train.data:size(2) * t_train.data:size(3) * t_train.data:size(4)
    assert(n_feature == expected_n_feature)

    local train_tensor = torch.Tensor(n_example, expected_n_feature+1) -- extra col for labels
    train_tensor[{{1, n_example},{1, n_feature}}] = t_train.data
    train_tensor[{{1, n_example},{n_feature+1}}] = t_train.labels

    local t_valid = torch.load(t7_path_valid)
    local n_example = t_valid.data:size(1)
    local n_feature = t_valid.data:size(2) * t_valid.data:size(3) * t_valid.data:size(4)
    assert(n_feature == expected_n_feature)

    local valid_tensor = torch.Tensor(n_example, expected_n_feature+1) -- extra col for labels
    valid_tensor[{{1, n_example},{1, n_feature}}] = t_valid.data
    valid_tensor[{{1, n_example},{n_feature+1}}] = t_valid.labels

    return train_tensor, valid_tensor
  end


  local tensor = torch.Tensor(expected_n_example, expected_n_feature+1) -- extra col for labels

  local t = torch.load(t7_path)
  local n_example = t.data:size(1)
  local n_feature = t.data:size(2) * t.data:size(3) * t.data:size(4)
  assert(n_feature == expected_n_feature)
  assert(n_example == expected_n_example)
  tensor[{{1, n_example},{1, n_feature}}] = t.data
  tensor[{{1, n_example},{n_feature+1}}] = t.labels

  return tensor

end

local function Gtsrbtest(num_images)
   local c = dp.Gtsrb()
   require 'image'
   local dt = c:trainSet():inputs(1)
   for idx = 1,num_images do
      local img = dt:image():select(1,idx):transpose(1,3)
      image.savePNG('Gtsrbimage'..idx..'.png', img)
   end
   dt:feature()
   for idx = 1,num_images do
      img = dt:image():select(1,idx):transpose(1,3)
      image.savePNG('Gtsrbfeature'..idx..'.png', img)
   end
   c:inputPreprocess(dp.LeCunLCN())
   c:preprocess()
   for idx = 1,num_images do
      img = dt:image():select(1,idx):transpose(1,3)
      print(dt:image():select(1,idx):size())

      image.savePNG('Gtsrblecun'..idx..'.png', img)
   end
end


-- This will generate a dataset as torch tensor from a directory of images
-- parent_path is a string of the path containing all the images
-- use validation allows to generate a validation set
generate_dataset_train_valid = function(parent_path, img_scaled_size)
  assert(parent_path, "A parent path is needed to generate the dataset")

  local nbr_elements = 0

  local images_directories = pl.dir.getdirectories(parent_path)
  table.sort(images_directories)

  local class_set = {} -- 43 tables of tracks
  for image_directory_id, image_directory in ipairs(images_directories) do
    local csv_file_name = 'GT-' .. pl.path.basename(image_directory) .. '.csv'
    local csv_file_path = pl.path.join(image_directory, csv_file_name)
    local csv_content = pl.data.read(csv_file_path)

    local filename_index = csv_content.fieldnames:index('Filename')
    local class_id_index = csv_content.fieldnames:index('ClassId')
    local Roi_X1 = csv_content.fieldnames:index('Roi_X1')
    local Roi_Y1 = csv_content.fieldnames:index('Roi_Y1')
    local Roi_X2 = csv_content.fieldnames:index('Roi_X2')
    local Roi_Y2 = csv_content.fieldnames:index('Roi_Y2')

    local track_set = {} -- tables of n track
    local img_set = {} -- 30 imgs
    local prev_track_nbr = -1
    for image_index, image_metadata in ipairs(csv_content) do
      local track_nbr = tonumber(pl.utils.split(image_metadata[filename_index], '_')[1])
      local image_path = pl.path.join(image_directory, image_metadata[filename_index])
      local image_data = image.load(image_path)
      -- cropping for regions of interest

      --if self._crop then
        image_data = image.crop(image_data,
                              tonumber(image_metadata[Roi_X1]),
                              tonumber(image_metadata[Roi_Y1]),
                              tonumber(image_metadata[Roi_X2]),
                              tonumber(image_metadata[Roi_Y2]))
      --end
      -- rescaling so all the images have the same size
      image_data = image.scale(image_data, img_scaled_size, img_scaled_size)
      local label = torch.Tensor{image_metadata[class_id_index]}

      if prev_track_nbr ~= track_nbr and prev_track_nbr ~= -1 then
        table.insert(track_set, img_set)
        img_set = {}
        prev_track_nbr = track_nbr
      end

      table.insert(img_set, {image_data, label})
      prev_track_nbr = track_nbr

      nbr_elements = nbr_elements + 1

      if image_index % 50 == 0 then
        collectgarbage()
      end
    end

    table.insert(track_set, img_set)
    table.insert(class_set,track_set)
  end

  local valid_data = torch.Tensor(1290,3,img_scaled_size, img_scaled_size) -- a random track of 30 imgs from each class... 43 * 30 = 1290 (30 is max)
  local valid_label = torch.Tensor(1290,1)

  local valid_count = 0
  local valid_tracks = {}
  local min_track_count = #class_set[1]
  for i=1,#class_set do
    local rand_track = torch.randperm(#class_set[i])[1]
    table.insert(valid_tracks, rand_track)
    min_track_count = math.min(min_track_count, #class_set[i])
    for j=1,#class_set[i] do
      if j==rand_track then
        print("Valid - adding class/track: "..i.."/"..j)
      end
      for k=1,#class_set[i][j] do
        if j==rand_track then
          valid_count = valid_count + 1
          valid_data[valid_count]:copy(class_set[i][j][k][1])
          valid_label[valid_count]:copy(class_set[i][j][k][2])
        end
      end
    end
  end

  --cut valid data
  valid_data = valid_data[{{1,valid_count}}] -- possible that track length is less than 30
  valid_label = valid_label[{{1,valid_count}}]

  local train_data = torch.Tensor(nbr_elements-valid_count, 3, img_scaled_size, img_scaled_size)
  local train_label = torch.Tensor(nbr_elements-valid_count, 1)

  local train_count = 0
  for i=1,#class_set do
    local train_track_count = self._balance and min_track_count or #class_set[i] -- set to min_track_count to
    if valid_tracks[i] > train_track_count then train_track_count = train_track_count - 1 end
    for j=1,train_track_count do
      if j ~= valid_tracks[i] then
        print("Train - adding class/track: "..i.."/"..j)
      end
      for k=1,#class_set[i][j] do
        if j ~= valid_tracks[i] then
          train_count = train_count + 1
          train_data[train_count]:copy(class_set[i][j][k][1])
          train_label[train_count]:copy(class_set[i][j][k][2])
        end
      end
    end
  end

  --cut train data
  train_data = train_data[{{1,train_count}}] -- possible that track length is less than 30
  train_label = train_label[{{1,train_count}}]
  print("train_count: "..train_count)
  print("valid_count: "..valid_count)


  train_dataset = {}
  train_dataset.data = train_data
  train_dataset.labels = train_label
  valid_dataset = {}
  valid_dataset.data = valid_data
  valid_dataset.labels = valid_label

  return train_dataset, valid_dataset
end

generate_dataset = function(parent_path, img_scaled_size)
  assert(parent_path, "A parent path is needed to generate the dataset")

  local main_dataset = {}
  main_dataset.nbr_elements = 0

  local images_directories = pl.dir.getdirectories(parent_path)
  table.sort(images_directories)

  for image_directory_id, image_directory in ipairs(images_directories) do
    local csv_file_name = 'GT-' .. pl.path.basename(image_directory) .. '.csv'
    local csv_file_path = pl.path.join(image_directory, csv_file_name)
    local csv_content = pl.data.read(csv_file_path)

    local filename_index = csv_content.fieldnames:index('Filename')
    local class_id_index = csv_content.fieldnames:index('ClassId')
    local Roi_X1 = csv_content.fieldnames:index('Roi_X1')
    local Roi_Y1 = csv_content.fieldnames:index('Roi_Y1')
    local Roi_X2 = csv_content.fieldnames:index('Roi_X2')
    local Roi_Y2 = csv_content.fieldnames:index('Roi_Y2')

    for image_index, image_metadata in ipairs(csv_content) do
      local track_nbr = tonumber(pl.utils.split(image_metadata[filename_index], '_')[1])
      local image_path = pl.path.join(image_directory, image_metadata[filename_index])
      local image_data = image.load(image_path)

      -- cropping for regions of interest
      --if self._crop then
        image_data = image.crop(image_data,
                              tonumber(image_metadata[Roi_X1]),
                              tonumber(image_metadata[Roi_Y1]),
                              tonumber(image_metadata[Roi_X2]),
                              tonumber(image_metadata[Roi_Y2]))
      --end
      -- rescaling so all the images have the same size
      image_data = image.scale(image_data, img_scaled_size, img_scaled_size)
      local label = torch.Tensor{image_metadata[class_id_index]}

      main_dataset.nbr_elements = main_dataset.nbr_elements + 1
      main_dataset[main_dataset.nbr_elements] = {image_data, label}

      if image_index % 50 == 0 then
        collectgarbage()
      end
    end
  end

  -- Store everything as proper torch Tensor now that we know the total size
  local main_data = torch.Tensor(main_dataset.nbr_elements, 3, img_scaled_size, img_scaled_size)
  local main_label = torch.Tensor(main_dataset.nbr_elements, 1)
  for i,pair in ipairs(main_dataset) do
    main_data[i]:copy(main_dataset[i][1])
    main_label[i]:copy(main_dataset[i][2])
  end
  main_dataset = {}
  main_dataset.data = main_data
  main_dataset.labels = main_label

  return main_dataset
end


relocate_gtsrb_test_file = function()
  os.execute('mkdir GTSRB/Final_Test/Images/final_test; ' ..
             -- too many arguments for a plain mv...
             [[find GTSRB/Final_Test/Images -maxdepth 1 -name '*.ppm' -exec sh -c 'mv "$@" "$0"' GTSRB/Final_Test/Images/final_test/ {} +;]] ..
             'rm GTSRB/Final_Test/Images/GT-final_test.test.csv')
  os.execute('mv GT-final_test.csv GTSRB/Final_Test/Images/final_test/GT-final_test.csv')
  return
end


-- Return the module
return dataset
