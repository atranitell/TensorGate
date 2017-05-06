# TensorGate [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]() [![tg](https://img.shields.io/badge/TensorGate-v1.0-brightgreen.svg)]()
**TensorGate** is an framework based on [Tensorflow](https://github.com/tensorflow/tensorflow) deep learning open source library. It offers a set of upper layer demo for a variety of deep learning, such as image classification, recognition, GAN network, etc.

### Gate Module
| Group | Component                | Status                                                                  | Details |
|-------|--------------------------|-------------------------------------------------------------------------|---------|
| Image | Classification           | passed        |         |
|       | Regression               | passed        |         |
|       | Regression Fuse          | passed        |         |
|       | Regression Fuse Share    | checking      |         |
| GAN   | Conditional GAN          | passed        |         |
| Audio | Audio Recognition - LSTM | unrealized |         |

### Gate Library

| Group         | Component                             | Status                                                                  | Details                                            |
|---------------|---------------------------------------|-------------------------------------------------------------------------|----------------------------------------------------|
| Data Loader   | load_image_from_text                  | passed        | Load Image from list.txt with path and labels.     |
|               | load_pair_image_from_text             | passed        | Considering two input with different feature data. |
|               | load_image_from_memory                | passed        | Put all images into memory and load in the system. |
|               | load_block_random_video_from_text     | checking      |                                                    |
|               | load_block_continuous_video_from_text | passed        |                                                    |
|               | load_pair_block_succ_video_from_text  | passed        |                                                    |
|               | tfrecord                              | pending       |                                                    |
| Dataset       | cifar10                               | passed        |                                                    |
|               | cifar100                              | passed        |                                                    |
|               | imagenet                              | passed        |                                                    |
|               | mnist                                 | passed        |                                                    |
| Net           | alexnet                               | passed        |                                                    |
|               | cifarnet                              | passed        |                                                    |
|               | inception_resnet_v2                   | passed        |                                                    |
|               | lenet                                 | passed        |                                                    |
|               | resnet_50/101/152/200                 | passed        |                                                    |
|               | vgg_11/16/19                          | passed        |                                                    |
| Preprocessing | cifarnet                              | passed        |                                                    |
|               | inception                             | passed        |                                                    |
|               | lenet                                 | passed        |                                                    |
|               | mnist                                 | passed        |                                                    |
|               | minst_for_gan                         | passed        |                                                    |
|               | vgg                                   | passed        |                                                    |
| Optimizer     | SGD                                   | passed        |                                                    |
|               | momentum                              | passed        |                                                    |
|               | adadelta                              | passed        |                                                    |
|               | adagrad                               | passed        |                                                    |
|               | adam                                  | passed        |                                                    |
|               | ftrl                                  | passed        |                                                    |
|               | rmsprop                               | passed        |                                                    |
| Updator       | layerwise learning rate               | passed        |                                                    |
|               | moving average decay                  | unrealized |                                                    |
| Snapshot      | Summary Hooks                         | passed        |                                                    |
|               | Saver Hooks                           | passed        |                                                    |

## Contact
If you have any issues, please contact: [atranitell@outlook.com]. We glad to welcome more people to join us to perfect the program.