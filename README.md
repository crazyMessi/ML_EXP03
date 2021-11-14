### 参考资料

[tensorflow处理不平衡数据集的官方教程](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)



### 工作流

#### 预处理

* 分成三类

* 打乱顺序

* 对氨基酸序列进行编码、填充

  nettcr用的是blosum50_20a，可以尝试其他的

* 改进数据集不平衡的问题

  * 过采样、欠采样
  * SMOTE采样

**输出**：

* \~_pep.npy :(len,9,20)
* \~_a.npy:(len,30,20)
* \~_b.npy:(len,30,20)
* ~binder.npy(len,1)

总共8个numpy数组（测试、训练集各四个）,~表示train或test，len表示长度。数组以文件的方式存在硬盘上



#### 模型训练

**输入：train_set.npk、test_set.npk**

* 划分验证集和训练集

* 训练模型
* 调参

**输出**：

* 模型在测试集上的表现。输出文件以使用的超参来命名，格式如下

  **lr<learnrate>bs<batchsize>ep<epoch>.csv**	

  例如，使用学习率为0.001，batch_size为128，ep为128，则输出文件的名称为

  lr0001bs128ep128.csv

* 模型训练过程

  **lr<learnrate>bs<batchsize>ep<epoch>.lop**

  其中包含log和在验证集上的AUC，需要根据这个表现来进行调参



#### 模型评估

**输入**

* lr<learnrate>bs<batchsize>ep<epoch>.csv

* lr<learnrate>bs<batchsize>ep<epoch>.lop

**输出**

* AUC、ROC曲线、P、R、PR曲线等
* 损失函数变化曲线、**验证集**AUC变化曲线

根据模型评估调整参数，反复进行模型训练以及模型评估







