# TensorGate [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]() [![tg](https://img.shields.io/badge/TensorGate-v1.0-brightgreen.svg)]()
**TensorGate** is an framework based on [Tensorflow](https://github.com/tensorflow/tensorflow) deep learning open source library. It offers a set of upper layer demo for a variety of deep learning, such as image classification, recognition, GAN network, etc.

<!--
[![tg](https://img.shields.io/badge/status-passing-green.svg)]()
[![tg](https://img.shields.io/badge/status-checking-yellow.svg)]()
[![tg](https://img.shields.io/badge/status-pending-orange.svg)]()
[![tg](https://img.shields.io/badge/status-unrealized-lightgrey.svg)]()
[![tg](https://img.shields.io/badge/status-failed-red.svg)]()
-->

### Gate Module
| Group | Component                | Status                                                                  | Details |
|-------|--------------------------|-------------------------------------------------------------------------|---------|
| Image | Classification           | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |         |
|       | Regression               | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |         |
|       | Regression Fuse          | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |         |
|       | Regression Fuse Share    | [![tg](https://img.shields.io/badge/status-checking-yellow.svg)]()      |         |
| GAN   | Conditional GAN          | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |         |
| Audio | Audio Recognition - LSTM | [![tg](https://img.shields.io/badge/status-unrealized-lightgrey.svg)]() |         |

### Gate Library

| Group         | Component                             | Status                                                                  | Details                                            |
|---------------|---------------------------------------|-------------------------------------------------------------------------|----------------------------------------------------|
| Data Loader   | load_image_from_text                  | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        | Load Image from list.txt with path and labels.     |
|               | load_pair_image_from_text             | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        | Considering two input with different feature data. |
|               | load_image_from_memory                | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        | Put all images into memory and load in the system. |
|               | load_block_random_video_from_text     | [![tg](https://img.shields.io/badge/status-checking-yellow.svg)]()      |                                                    |
|               | load_block_continuous_video_from_text | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | load_pair_block_succ_video_from_text  | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | tfrecord                              | [![tg](https://img.shields.io/badge/status-pending-orange.svg)]()       |                                                    |
| Dataset       | cifar10                               | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | cifar100                              | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | imagenet                              | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | mnist                                 | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
| Net           | alexnet                               | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | cifarnet                              | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | inception_resnet_v2                   | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | lenet                                 | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | resnet_50/101/152/200                 | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | vgg_11/16/19                          | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
| Preprocessing | cifarnet                              | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | inception                             | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | lenet                                 | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | mnist                                 | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | minst_for_gan                         | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | vgg                                   | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
| Optimizer     | SGD                                   | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | momentum                              | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | adadelta                              | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | adagrad                               | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | adam                                  | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | ftrl                                  | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | rmsprop                               | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
| Updator       | layerwise learning rate               | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | moving average decay                  | [![tg](https://img.shields.io/badge/status-unrealized-lightgrey.svg)]() |                                                    |
| Snapshot      | Summary Hooks                         | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |
|               | Saver Hooks                           | [![tg](https://img.shields.io/badge/status-passing-green.svg)]()        |                                                    |

## Contact
If you have any issues, please contact: [atranitell@outlook.com]. We glad to welcome more people to join us to perfect the program.