# TensorGate [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]() [![tg](https://img.shields.io/badge/TensorGate-v4.0-brightgreen.svg)]()
**TensorGate** is an framework based on [Tensorflow v1.5](https://github.com/tensorflow/tensorflow) deep learning open source library. It offers a set of upper layer demo for a variety of deep learning, such as image classification, recognition, GAN network, etc.

### Update
- package functions with class.
- offer a global constant file.
- learning rate: add cosine, linear cosine, noisy linear cosine, inverse time.
- optimizer: add proximal, proximal adagrad

### Directory
- **./config** provide config file for dataset.
- **./core** provide critical functions for running gate framework.
  - **./utils** system utils.
    - **filesystem.py** folder/file operation
    - **path.py** path string operation
    - **string.py** string/tensor-string operation
    - **profiler.py** detecter of network model
    - **devicequery.py** help to search/show device info
    - **logger.py** event helper
  - **./tools** computational tools.
  - **constant.py** the environment variables.
- **./draw** provide a set of drawing tools by using log file.
- **./example** provide a series of demo.
- **./tools** provide common tools.
  - **dataset.py** make train/test file from a folder.
  - **checkpoint.py** parse checkpoint file of tensorflow.
- **compile.py** compile source to binary format
- **gate.py**
- **pipeline.py**