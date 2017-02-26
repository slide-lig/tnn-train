This repository is released for reproducing the results in the following publication

Hande Alemdar, Vincent Leroy, Adrien Prost-Boucle, and Frederic Petrot. “Ternary Neural Networks for Resource- Efficient AI Applications”. In: International Joint Conference on Neural Networks (IJCNN). 2017.

This repository provides the training code of the student network.
The ternarization code is provided in the following repository.

https://github.com/slide-lig/tnn_convert


Installation
============

Requirements
------------
The following python packages are required:
  ```
git clone https://github.com/caldweln/distro.git ~/torch --recursive
cd ~/torch;TORCH_LUA_VERSION=LUA51 ./install.sh;source install/bin/torch-activate;
git clone https://github.com/caldweln/dp.git ~/dp
cd ~/dp; luarocks make rocks/dp-scm-1.rockspec
git clone https://github.com/caldweln/nninit.git ~/nninit
cd ~/nninit; luarocks make rocks/nninit-scm-1.rockspec
luarocks install dkjson
  ```
  - Optional
    - CUDA
    ```
    luarocks install cunnx
    ```
      - cuDNN (https://developer.nvidia.com/cudnn) (ensure libcudnn.so.5 location in $LD_LIBRARY_PATH)
      - CUDA Driver/Toolkit (ensure nvcc location in $PATH)
    - Other
    ```
    luarocks install matio
    luarocks install cephes
    luarocks install csvigo
    ```

Usage
======

train.lua
--------
Use for training and testing the suite of neural network implementations.

Navigate to lsr/bin and execute:
```
~$ th train.lua <files>
```

<files> a variable length .conf file list providing architecture, input and run configuration.

Configurations may be a list of part and/or full configurations.
If giving a configuration in parts, the combined set must add up to a full configuration.
A full configuration consists of 'arch', 'input' and 'run' tables.

A composite configuration contains key-values that may cause the configuration to split into two or more separate configuration. Examples of such key-values are detailed below 'Configuration options'.

An additional 'main' configuration table may be provided to resume an execution.

Example configurations can be found in the /etc directory.

Example configuration in full execution:
```
~$ th train.lua ~/tnn_train/etc/full/simple-mlp.conf
```

Example configuration in parts execution:
```
~$ th train.lua ~/tnn_train/etc/arch/mlp-1x100.conf ~/tnn_train/etc/input/MNIST.conf ~/tnn_train/etc/run/100_epochs_10_batch.conf
```

Example resuming existing training:
```
~$ th train.lua ~/tnn_train/log/MLP/results/main.conf
```

Configuration options
---------------------
```
#
# arch
#

arch.modelArch=MLP                          # MLP | LENET | ProgressiveBinarization | BinaryConnect
arch.neuronPerLayerCount.1=1000             # number of neurons for hidden layer 1
arch.neuronPerLayerCount.2=1000             # similarly for hidden layer 2, add/remove layers as desired
arch.dropout_in=0                           # Probability of an input unit being DROPPED
arch.dropout_conv=0                         # Probability of an convolutional unit being DROPPED
arch.dropout_hidden=0                       # Probability of a hidden unit being DROPPED
arch.batchnorm.epsilon=0.0001               # BinaryConnect, BatchNormalization argument
arch.convChannelSize.1=6                    # Filter/Channel size for output of first convolutional layer
arch.convChannelSize.2=16                   # Filter/Channel size for output of second convolutional layer
arch.convKernelSize=5                       # Kernel/Receptor size for convolutional layer
arch.convPoolSize=2                         # Pooling window size for MaxPooling layer
arch.convPadding=2                          # Padding size for convolutional layer
arch.activationFn=ReLU                      # Activation/non-linearity function to be applied
arch.stochFireDuringTraining=1              # Enable/Disable stochastic firing through training
arch.criterion=ClassNLLCriterion            # SqrHingeEmbeddingCriterion + more at https://github.com/torch/nn/blob/master/doc/criterion.md
arch.criterionArg=                          # rarely used criterion arguments, e.g. 2 for square margin loss with MarginCriterion
arch.paramInit.x=weight                     # weight | bias | weight,bias
arch.paramInit.fn=uniform                   # initialization fn, full list at https://github.com/Kaixhin/nninit
arch.paramInit.args.1=-1                    # arguments list/table for paramInit.fn
arch.paramInit.args.2=1                     # arguments list/table for paramInit.fn

#
# arch prep                                 # A 'perparation' model that precedes the main model defined above
#

arch.prep.modelArch=PREPCONV                # architecture to load just as above, loads a model from the Models dir
arch.prep.convChannelSize.1=3               # arch.prep table options for those specific to this preparation model configuration
arch.prep.convKernelSize=3
arch.prep.convPadding=1
arch.prep.batchNorm=1
arch.prep.stcFlag=1
arch.prep.outputTernary=1

#
# input
#

input.narize.bitCount=1                     # quantize input data to this number of bits
input.narize.signed=0                       # quantize as 2s complement when this is enabled (=1)
input.fractureData=1                        # fracture off to use only part of the data: 1 | 0.1 | etc
input.distort.count=0                       # Number of additional train sets
input.distort.rotate=0                      # Rotate train data within +/- <rotate> [°]
input.distort.scale=0                       # Scaled train data within +/- <scale> [%]
input.distort.translate=0                   # Translate train data within +/- <translate> [px]
input.dataset=mnist                         # input datasource, mnist | notmnist | cifar10 | cifar100 | gtsrb32 | gtsrb48
input.validRatio=0.1                        # ratio to split train/valid set, will use dataset default if empty
input.normalize=0                           # normalize input data
input.scale=0                               # scale input data
input.zca=1                                 # Performs Zero Component Analysis Whitening on input
input.gcn=1                                 # Performs Global Contrast Normalization on input
input.lecunlcn=0                            # Implements the Local Contrast Normalization Layer at page 3 in http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf
#
# run
#

run.batchSize=200                           # mini-batch size
run.randseed=1
run.shuffleIndices=1                        # shuffle indices to training data on each epoch
run.optim.fn=sgd                            # optim fn, run.optim.fn & run.optim.conf described at https://github.com/torch/optim/blob/master/doc/index.md
run.optim.conf.learningRate=0.1             # learning rate
run.optim.conf.learningRateDecay=0.001      # learning rate decay, supports schedule table or string formula for adam, adamax and rmsprop (see example lsr/etc/optim/adam.conf)
run.optim.conf.momentum=0                   # momentum
run.optim.conf.weightDecay=0                # weight decay
run.optim.conf.dampening=0                  # dampening for momentum
run.optim.conf.nesterov=false               # enables Nesterov momentum

#
# main
#

main.verbose=3                               # Level of verbosity for output/logging
main.stopEarly=30                            # Stop training early if determined to have reached best performance
main.threads=16                              # torch thread count
main.cuda=0                                  # attempt use of CUDA api for GPU usage
main.cudnn=0                                 # use cuDNN modules for faster but non-deterministic training
main.nGPU=1                                  # specify how many GPUs to use
main.debug=0
main.dstDir=results_%main.confhash%          # friendly name for output
main.binPath=
main.epochCount=100                          # max number of epcohs to run
main.logFileSize=100000000                   # max log file size
main.trace.loss=0                            # set to 1 to enable tracing of training loss
main.trace.learningRate=0                    # set to 1 to enable tracing of learning rate
main.hist.params=0                           # set to 1 to enable plotting histograms of model parameters before each epoch
```

- Composite Configuration

Composite configurations may contain key-values that allow a set of values be described.
This will cause the train script to split the configuration and run the config with each unique value.
Composite configurations will continue to split until all key-values have been reduced down.

 -- Explicit value set
  Use {} to enclose a comma separated list of explicit values
    e.g. arch.modelArch={'BC','EBP'}    # will cause the initial conf to split into two each with a unique value of arch.modelArch, namely 'BC' and 'EBP'

 -- Range value set
  Use {} to enclose 3 values ':' separated, to give the initial value, the increment and final value.  
    e.g. arch.modelWidth={3:1:5}      # will cause the initial conf to split into three each with a unique value of arch.modelWidth, namely 3, 4 and 5

If both examples given here are provided in the same initial configuration, the result will be SIX separate configurations giving a matrix of permutations.

plot.lua
--------
For plotting the json data results from the train.lua script.

Navigate to lsr/bin and execute
```
~$ th plot.lua <json file path(s)>
```
Example:
```
~$ th plot.lua /patt/to/yourfiles/*.json
```
This will initiate qt to display the plot, and generate an image on the filesystem alongside the json files.

util.lua
--------
For misc functions like preparing datasets, you can use utility script.
This is useful as train.lua expects train.t7, valid.t7, test.t7 and classlist.t7 to exist in the <dataSrc>/<srcDir> directory.
All functions leave source data intact (non-destructive).

data.mat -> data.t7

To convert a matlab dataset and save the torch file to /path/to/data.t7:
```
~$ th util.lua --run mat2t7 --data /path/to/data.mat
```

data.t7 -> train.t7 & valid.t7

To split data.t7 and save to /path/to/train.t7 & /path/to/valid.t7 with a ratio of 60:40, while maintaining class distribution:
```
~$ th util.lua --run splitData --data /path/to/data.t7 --fractureData 0.6
```

data.t7 -> classlist.t7

To generate a table of unique classes (saved to /path/to/classlist.t7) that exist in the 'labels' tensor in the data.t7:
```
~$ th util.lua --run genClasslist --data /path/to/data.t7
```

epoch.net -> [ epoch.net.weight_layer1.csv, epoch.net.bias_layer1.csv ]

To generate a set of CSV files of weights and biases (for each layer) from a saved network:
```
~$ th util.lua --run net2csv --data /path/to/epoch.net
```

[ epoch.net, epoch.net.weight_layer1.csv, epoch.net.bias_layer1.csv ] -> epoch.net

To load CSV files of weights and biases (for each layer) to a saved network:
```
~$ th util.lua --run csv2net --data /path/to/epoch.net
```

License
============

Copyright 2017 Université Grenoble Alpes

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
