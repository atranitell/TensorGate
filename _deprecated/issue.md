
# 当前版本存在的问题

## 数据层
### 数据流分层应该更明显
- 读取层：从text中/ubyte文件/npy文件
- 贮存层：决定是全部贮存在内存还是从文件中动态读取[根据系统memory判断]
- 采样层：不仅仅是简单的随机采样，应可以定义多种采样方式
- 预处理层：在这一层数据会进行各种预处理方式[包括增加噪声等等]
- 批封装层：这一层应该更robust一点，现在需要对每一种格式都定义一次，太繁琐。考虑几种范式（数据-标签，数据-数据，数据），每一种范式自动解析格式和参数数目。

### 增加sampler方法
- 自定义的active sampler method

## 模型层
- 在系统每次运行时，应该有一个class贮存全局变量
- 考虑到后续CNN/RNN/LSTM/ML算法，如何进行有效的融合互不干涉，参数独立
    - 每个net在入口处需要检查所需的超参
- net factory的API需要考虑到一个更通用的接口（支持gan等）
- *slim model* 由于slim不断在更新，为保持更好的兼容性，是否可以独立于slim文件单独提供yi'ge

## 功能层
### 分布式运行
- 考虑protoconfig函数以及模型的分布式运行[通过多个GPU]

### 数据库
- 为更好的可视化等方便后续的web操作，应该加入一个数据库系统。前端可直接进行读取，或动态监控模型变化。

### logger
- logger应该可以直接在cmd中指定文件夹并自动保存到文件夹内部
- 支持每一步的速度分析

### 外部接口
- 考虑mtcnn提供的人脸分割工具入口
- 考虑ssd提供的图像分割工具的入口

### API重构
- 考虑issue层的接口重新调整，比如有些val=test=val-train
- 考虑全局参数和变量在一个类中，由它配置所有的设置


# 未来支持的issue
## CNN
- single-image classification/regression
- multi-image regression/classification
- audio data classification/regression
- video data classification/regression
- object segmentation
- tracking
## GAN
- cgan/wgan/...
## VAE
- ..
## RNN/LSTM
- text translator
- speech translator
## RL
- a simple demo

# top-layer
## 模型自动化训练
- *配置自动生成* 需要将配置写入文本中而不是py中

## 深度学习可视化工具
- tensorflow summary 单独封装起来，成为一个类，直接添加即可
- 考虑deep tracker的设计
- 考虑直接导入数据库的设计（与前端交互）

## tools
- checkpoint工具


### PS
- 命名风格统一化！！！
