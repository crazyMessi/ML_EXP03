[TOC]



### 参考资料

[tensorflow处理不平衡数据集的官方教程](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)

### 使用说明

* 配置好tensorflow的环境之后，在根目录下输入命令

  `python nettcr.py -tr data/model_input/train_set.csv -te data/model_input/gig_test.csv`

  即可运行代码。不想打印在控制台上可以重定向到文件里，详见python教程。



### 工作流

#### 预处理

* 分成三类

* 打乱顺序

* 对氨基酸序列进行编码、填充

  nettcr用的是blosum50_20a，可以尝试其他的

* 改进数据集不平衡的问题

  * 过采样、欠采样
  
  * SMOTE采样
  
    SMOTE采样因为要根据编码方式来进行 所以不太好用csv格式

**输出**：

* \~_pep.npy :(len,9,20)
* \~_a.npy:(len,30,20)
* \~_b.npy:(len,30,20)
* ~binder.npy(len,1)

总共8个numpy数组（测试、训练集各四个）,~表示train或test，len表示长度。数组以文件的方式存在硬盘上



#### 模型训练

**输入：train_set.npk、test_set.npk**

* 避免过拟合
  * 划分验证集和训练集
  * 正则化

* 局部最优$\rightarrow$全局最优

  用多组不同的值初始化参数

* 训练模型
  * 衰减轮
* 调参

**输出**：

* 模型在测试集上的表现。输出文件以使用的超参来命名，格式如下

  **lr<learnrate>bs<batchsize>ep<epoch>.csv**	

  例如，使用学习率为0.001，batch_size为128，ep为128，则输出文件的名称为

  lr0001bs128ep128.csv

* 模型训练过程

  **lr<learnrate>bs<batchsize>ep<epoch>.log**

  其中包含log和在验证集上的AUC，需要根据这个表现来进行调参



#### 模型评估

**输入**

* lr<learnrate>bs<batchsize>ep<epoch>.csv

* lr<learnrate>bs<batchsize>ep<epoch>.log

**输出**

* AUC、ROC曲线、P、R、PR曲线等
* 损失函数变化曲线、**验证集**AUC变化曲线

根据模型评估调整参数，反复进行模型训练以及模型评估



### 题目解析

#### 背景

TCR有两个链，CDR α和 CDR β链，每个链分别有三个环，分别称为CDR-1/2/3，其中与多肽的结合能力主要由3链反映，也就是CDR3链。

α链又主要由V-、J **基因**重组而成；β主要由V-、D-和J **基因**组成，因而更多样。

[DeepTCR is a deep learning framework…](参考资料)

#### TCR链特征提取

[AI-MHC: an allele-integrated deep learning framework for improving Class I & Class II HLA-binding predictions](https://www.biorxiv.org/content/10.1101/318881v1.full.pdf):

解决问题：将不同长度的基因序列传入输入层

类比图像识别，其做到的是：在保留**局部特征**的情况下，最小化进入网络的图像

但在统一输入向量时存在困难：

* 不等长的序列不可以像图像一样通过放缩统一。
* 统一的长度应该设为多长？





**池化（Pooling）**

池化过程在一般卷积过程后。池化（pooling） 的本质，其实就是采样。Pooling 对于输入的 Feature Map，选择某种方式对其进行降维压缩，以加快运算速度。

主要作用：

* 保留主要特征的同时减少参数和计算量，防止过拟合
* invariance(不变性)，这种不变性包括translation(平移)，rotation(旋转)，scale(尺度)





#### NetTCR

NetTCR指出大多数数据生成研究只关注CDR3-β链，但配对预测会提高精度。

* **The amino acids were encoded using the
  BLOSUM50 matrix**,That is, each amino acid is represented as the score for
  substituting the amino acid with all the 20 amino acids.



**Model**

* 构建：
  * 输入：CDR3a、CDR3ab、peptide**三个序列**
  * 编码：每一个序列$\color{red}使用零填充$至等长，然后使用BLOSUM50变成(30,20)/(9,20)的向量
  * 每一个序列，$\color{red}分别通过16个大小为 1、3、5、7、9的卷积核$；对三者池化,每一个序列得到长为80的特征向量，三个特征向量拼接（concatenated）得到一个长为240的特征向量
  *  特征向量进入有32个隐藏层的神经网络，输出一个预测值。

* 训练（$\color{red}主要修改fit函数中的参数$）
  * 五折交叉验证
  
    需要设置
  
  * epoch = 300
  
  * lr = 0.001($\color{red}首先调整$)
  
  * batch_size = 128
  
    $\color{red}可以衰减;在lr后调整$
  
  * loss function:binary_crossentropy交叉熵损失函数（适用于二分类）
  
  * $\color{red}drop函数（正则化）$
  
* 输出

  * 



### 源码使用

训练一个周期作为测试：

>python nettcr.py -tr data /model_input/train_set.csv -te data/model_input/gig_test.csv >data/model_output/test/test.log --e 1





### 训练结果





