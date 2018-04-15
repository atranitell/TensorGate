# TensorGate [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]() [![tg](https://img.shields.io/badge/TensorGate-v4-brightgreen.svg)]()
**TensorGate** is an framework based on [Tensorflow v1.6](https://github.com/tensorflow/tensorflow) deep learning open source library. It offers a set of upper layer demo for a variety of deep learning, such as image classification, recognition, GAN network, etc.

### Update 4.1.1 [3/6/2018]
- add batchnorm params to update collections
- add GradCAM, guidedCAM, guided backpropagation
- add heatmap for AVEC2014-Image
- package heatmap as a class
- add imagenet dataset

### Update 4.1 [3/4/2018]
- add AVEC2014 Image Issue [untest]
- add AVEC2014 Audio Issue [untest]
- update slim model to TensorGate
- update net factory logic: argscope in the head of net model

### Update 4.0 [3/2/2018]
- package functions with class.
- offer a global constant file.
- learning rate: add cosine, linear cosine, noisy linear cosine, inverse time.
- optimizer: add proximal, proximal adagrad.
- env: summary, logger, compiler.
- move classical model to ./example.
- move project model to ./issue.
- separate preprocessing method by input format.
- re-construct data layer.
- move a part of classical method to example folder.

### Directory
- **./config** provide config file for dataset.
  - **./dataset** provide a series of dataset configuration.
  - **factory.py** a factory to allocate different dataset.
- **./core** provide critical functions for running gate framework.
  - **./data** data abstract layer to process data format and pipline.
    - **data_entry.py** parse the textline data record.
    - **data_params.py** offer a guideline to setting hyperparameters.
    - **data_prefetch.py** make data pipline become a batch.
    - **data_utils.py** used for data loader.
    - **database.py** a base class to offer a set of database format.
    - **factory.py** assemble of data loader.
  - **./loss** a convenient tools to compute loss and error.
  - **./network** collect a variety of network models.
  - **./solver** offer the training tools and snapshot.
    - **context.py** the running context to manage the app.
    - **learning_rate.py** provide different learning rate decay method.
    - **optimizer.py** provide various optimizer.
    - **snapshot.py** dumping the ckpt in according to a certain of iterations.
    - **summary.py** record the running info.
    - **updater.py** updater gradient to weights.
  - **./utils** system utils.
    - **checkpoint.py** load and operate tensorflow ckpt.
    - **devicequery.py** help to search/show device info.
    - **filesystem.py** folder/file operation.
    - **image.py** offer image save, read, etc.
    - **logger.py** event helper.
    - **path.py** path string operation.
    - **profiler.py** detecter of network model.
    - **similarity.py** a cosine-metric method.
    - **string.py** string/tensor-string operation.
    - **variables.py** search/print/select variables in the network.
  - **./tools** computational tools.
  - **env.py** the environment variables.
- **./draw** provide a set of drawing tools by using log file.
  - **_config.py** to generate multi config file used for draw-tools.
  - **_line.py** drawing line curve.
  - **_parse.py** parse log file.
  - **_roc.py** drawing roc curve.
  - **_trend.py** drawing trend curve.
  - **main.py** allocate to different tasks.
  - **utils.py** file handle and numerical treatment.
- **./example** provide a series of demo.
- **./tools** provide common tools.
  - **dataset.py** make train/test file from a folder.
  - **checkpoint.py** parse checkpoint file of tensorflow.
- **compile.py** compile source to binary format.
- **gate.py** the gate task allocator.
- **pipeline.py** pipline to processing tasks.