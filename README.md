<!-- <div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</div> -->

# TensorGate

-----------------

| **`Version`** | **`License`** | **`Linux CPU`** | **`Linux GPU`** | **`Windows CPU`** | **`Windows GPU`** | **`Mac GPU`** |
|---------------|---------------|-----------------|-----------------|-------------------|-------------------|---------------|
|![][version]   |![][license]   |![][linux-cpu]   |![][linux-gpu]   |![][win-cpu]       |![][win-gpu]       |![][mac-gpu]   |

[version]: https://img.shields.io/badge/TensorGate-v4.2-brightgreen.svg
[license]: https://img.shields.io/badge/license-Apache--2.0-blue.svg
[linux-cpu]: https://img.shields.io/badge/build-passed-brightgreen.svg
[linux-gpu]: https://img.shields.io/badge/build-failed-brightred.svg
[win-cpu]: https://img.shields.io/badge/build-passed-brightgreen.svg
[win-gpu]: https://img.shields.io/badge/build-running-blue.svg
[mac-gpu]: https://img.shields.io/badge/build-unsupport-lightgrey.svg


**TensorGate** is an open-source framework based on [TensorFlow](https://github.com/tensorflow/tensorflow) deep learning open source library. It offers a set of upper layer demo for a variety of deep learning, such as image classification, recognition, segmentation, GAN network, etc. Also, it offeres a set of analyizer tools to parse the log data.

## Requirements
- python > 3.6
- tensorflow >= 1.7.0
- CUDA == 9.0
- cuDNN == 7.0
- python-opencv
- json

## Usage
```shell
# running to train mnist dataset on GPU-0
$ python main.py 0 -dataset=mnist
# load config file to train mnist on GPU-0
$ python main.py 0 -dataset=mnist -extra=demo.json
```

## Directory
- **\<asserts\>** used by example code
- **\<demo\>** pre-setting config file
- **\<gate\>** provide critical functions for running gate framework
  - **\<config\>** provide config file for a variety of datasets
    - **\<dataset\>** a series of dataset config to index
      - **[mnist.py](#)** MNIST classification
    - **[config_base.py](#)** config base class
    - **[config_params.py](#)** config params to set parameters
    - **[config_factory.py](#)** config factory to index the dataset
  - **\<data\>** data index and prefetech method
    - **\<tfqueue\>** using tensorflow queue and batch prefetch method
    - **\<custom\>** customed data model by using placeholder
    - **[data_utils.py](#)** a unified tools for all data model
    - **[data_factory.py](#)** data factory to index the data model
  - **\<layer\>** customed loss/net/ops assemble
  - **\<net\>** collect a variety of network models
    - **\<custom\>** customed network for specific task
    - **\<deepfuse\>** multi-layer weight shared network model
    - **\<nets\>** slim net model
    - **\<vae\>** varational auto-encoder model zoos
    - **[net_factory.py](#)** net factory to index the model
    - **[net_model.py](#)** model parameter config assemble
  - **\<solver\>** offer the training tools and snapshot
    - **[learning_rate.py](#)** provide different learning rate decay method
    - **[optimizer.py](#)** provide various optimizer
    - **[snapshot.py](#)** dumping the ckpt in according to a certain of iterations
    - **[summary.py](#)** record the running info
    - **[updater.py](#)** updater gradient to weights
  - **\<util\>** system utils
    - **[checkpoint.py](#)** load and operate tensorflow ckpt
    - **[devicequery.py](#)** help to search/show device info
    - **[filesystem.py](#)** folder/file operation
    - **[logger.py](#)** event helper
    - **[profiler.py](#)** detecter of network model
    - **[similarity.py](#)** a cosine-metric method
    - **[string.py](#)** string/tensor-string operation
    - **[variable.py](#)** search/print/select variables in the network
    - **[heatmap.py](#)** generate heatmap image for visual data
  - **\<processing\>** data processing method
    - **\<slim\>** including slim processing method and corresponding interface
    - **[processing_vision.py](#)** vision data processing
    - **[processing_audio.py](#)** audio data processing
    - **[processing_text.py](#)** text type data processing
  - **[context.py](#)** the running context to manage the app
  - **[env.py](#)** a global output control center
- **\<samples\>** offer some examples for current deep learning tasks
  - **\<vision\>** several common vision tasks
    - **[classification.py](#)**
    - **[regression.py](#)** 
    - **[mask_rcnn.py](#)**
    - **[vanilla_gan.py](#)**
    - **[vanilla_vae.py](#)**
  - **\<avec2014\>** research on AVEC2014 (multi-modal) depressive recognition tasks
  - **\<kinship\>** research on Kinship recognition tasks
  - **\<trafficflow\>** research on TrafficFlow dataset tasks
- **\<tools\>** some external tools to analyze the data and log event files
  - **[drawer.py](#)** provide a set of drawing tools by using log file
  - **[dataset.py](#)** provide tool to generate train/val/test file for specific data folder
- **[main.py](#)** start and initialize the system
- **[pipline.py](#)** execute multi-task at once
- **[compile.py](#)** packege `gate` into a fold in `.bin` or `.py` without debug info

## To-Do
- [x] (05/15/18) Merge drawer into gate
- [x] (05/12/18) Resume pipline & fix a bug of config-base re-write
- [x] (05/12/18) Merge drawer in
- [x] (05/08/18) Review Kinface related issues
- [x] (05/08/18) Review AVEC2014-CNN/HEATMAP/FLOW/BICNN/Audio-NET
- [ ] Reconstruct gate framework to make more flexible
- [ ] Add mask-rcnn trainig module
- [ ] Add mask-rcnn inference module
- [ ] Add mask-rcnn visualization method
- [x] (04/08/18) Add batchnorm params to update collections (fixed BN un-trained)
- [x] (03/08/18) Add GradCAM, guidedCAM, guided backpropagation
- [x] (03/08/18) Add heatmap for AVEC2014-Image
- [x] (03/08/18) Package heatmap as a class
- [x] Update slim model to TensorGate
- [x] Update net factory logic: argscope in the head of net model
- [x] Package functions with class
- [x] Learning rate: add cosine, linear cosine, noisy linear cosine, inverse time
- [x] Optimizer: add proximal, proximal adagrad
- [x] Env: summary, logger, compiler
- [x] Move classical model to ./example
- [x] Move project model to ./issue
- [x] Separate preprocessing method by input format
- [x] Re-construct data layer
- [x] Move a part of classical method to example folder

## License
[Apache License 2.0](https://github.com/atranitell/TensorGate/blob/v6/LICENSE)