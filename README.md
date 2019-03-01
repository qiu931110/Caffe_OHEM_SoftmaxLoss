#### 0.前言
在解决一个分类问题时，遇到样本不平衡问题。查找CSDN后，以及知乎后，发现网上有很多类似于欠采样 ，重复采样，换模型等等宏观的概念，并没有太多可实际应用（代码）的策略。经过一番查找和调试和修改，最终找到3个相对靠谱的策略，故总结此文给有需要同志，策略均来自网络，本人只是进行了部分代码修改和可用性测试。以下将简单介绍各个策略的机制以及对应代码（亲测能跑通）。
>NOTE：下述代码均是基于caffe的，而且实现策略都是通过新增自定义层。主要流程大致为：修改caffe.proto-->导入hpp/cpp/cu-->重新编译-->执行。
#### 1.带权重的softmaxLoss

[softmaxLoss代码——github传送门](https://github.com/qiu931110/Weighted_Softmax_Loss)
在样本不均衡分类问题中，样本量大的类别往往会主导训练过程，因为其累积loss会比较大。带权重的softmaxloss函数通过加权来决定主导训练的类别。在具体的实现过程中增加pos_mult字段（指定某类的权重乘子）和pos_cid字段（指定的某类的类别编号）两个参数来确定类别和当前类别的系数，通过系数来控制当前类别在反向传播中的重要性。（若pos_mult=0.5，就表示当然类别重要度减半）。


（1）修改caffe.proto文件
编辑src/caffe/proto/caffe.proto文件，主要是在原有的SoftmaxParameter字段上添加了pos_mul和pos_cid字段。
```
  optional float pos_mult = 3 [default = 1];
  optional int32 pos_cid = 4 [default = 1];
```

（2）导入hpp/cpp/cu文件
将```weighted_softmax_loss_layer.hpp```文件添加到include/caffe/layers/文件夹下。
将```weighted_softmax_loss_layer.cpp```文件添加到src/caffe/layers/文件夹下。
将```weighted_softmax_loss_layer.cu```文件添加到src/caffe/layers/文件夹下。

（3）编译
返回到caffe的根目录，使用make指令(下面几个都可以，任选一个)，即可。
```
 make
 make -j
 make -j16
 make -j32    // 这里j后面的数字与电脑配置有关系，可以加速编译
```
（4）使用方法
```
layer {
  name: "loss"
  type: "WeightedSoftmaxWithLoss"
  bottom: "fc_end"
  bottom: "label"
  top: "loss"
  softmax_param {
    pos_cid: 1
    pos_mult: 0.5
  }
}
```
需要注意的是pos_cid也是从0开始的，和caffe中标签的定义从0开始对应。若指定为0表示pos_mult的参数将乘到对应的类别中，简而言之就是和标签对应，对应代码如下。
```
 Dtype w = (label_value == pos_cid_) ? pos_mult_ : 1;
```

#### 2.OHEMLoss

[OHEM论文地址](https://arxiv.org/abs/1604.03540)
[OHEM代码——github传送门](https://github.com/qiu931110/Caffe_OHEM_SoftmaxLosss)
OHEM被称为难例挖掘，主要针对模型训练过程中导致损失值很大的一些样本进行特殊处理，重新训练它们。因此OHEM也能在一定程度上解决样本分布不均的问题。它会维护一个错误分类样本池, 把每个batch训练数据中的出错率很大的样本放入该样本池中，将这些样本放回网络重新训练。通俗的讲OHEM就是加强loss大样本的训练。

（1）修改caffe.proto文件
编辑src/caffe/proto/caffe.proto文件，主要是在原有的SoftmaxParameter字段上添加如下三个字段。
```
  optional bool use_hard_mining = 3 [default = false];
  optional int32 batch_size = 4 [default = 1];
  optional float hard_ratio = 5 [default = 1];
```
（2）导入hpp/cpp/cu文件
将```softmax_loss_ohem_layer.hpp```文件添加到include/caffe/layers/文件夹下。
将```softmax_loss_ohem_layer.cpp```文件添加到src/caffe/layers/文件夹下。
将```softmax_loss_ohem_layer.cu```文件添加到src/caffe/layers/文件夹下。
（3）编译
返回到caffe的根目录，使用make指令(下面几个都可以，任选一个)，即可。
```
 make
 make -j
 make -j16
 make -j32    // 这里j后面的数字与电脑配置有关系，可以加速编译
```
（4）使用方法
```
layer {
  name: "loss"
  type: "SoftmaxWithLossOHEM"
  bottom: "fc_end"
  bottom: "label"
  top: "loss"
   softmax_param {
    use_hard_mining : 1
    hard_ratio : 0.5
  }
}
```
使用过程中，```use_hard_mining ```字段表示是否使用ohem。```hard_ratio ```字段表示有多少样本被定义为难例样本，如上述所示有百分之50的难例样本。
#### 3.FocalLoss
[FocalLoss论文代码](https://arxiv.org/abs/1708.02002)
[FocalLoss代码——github传送门](https://github.com/qiu931110/Focal-Loss)
FocalLoss就是在带权重的Loss的基础上作出了改进，更好的解决样本不平衡问题，总体思想和带权重的有点类似，FocalLoss首先解决的就是样本不平衡的问题，类似于weihgtsoftmaxloss。即在CE上加权重，当class为1的时候，乘以权重alpha，当class为0的时候，乘以权重1-alpha，正如第一个策略所示，这是最基本的解决样本不平衡的方法，也就是在loss计算时乘以权重。
![](https://upload-images.jianshu.io/upload_images/5529997-996992d32b701b27.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
在此基础上，focalloss的核心就是在CE的前面乘上了（1-pt）的gama次方。pt就是准确率，因此该公式表示的含义为：准确率越高 ，整个loss值就越小。所以我们把参数gama称为衰减系数，准确率越高的类衰减的越厉害。这就使得准确率低的类能够占据loss的大部分，并主导训练。
![](https://upload-images.jianshu.io/upload_images/5529997-146da207a59e05a5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
正如上述讲到的第二种方法OHEM是让loss大的进行主导。而FocalLoss是让准确率低的类进行主导。两者在这个机制上殊途同归。但OHEM的缺点是其只取一部分难例样本进行loss计算来实现上述功能，而FocalLoss则作用于所有样本，因此从原理上讲FocalLoss会更加有效。最终FocalLoss的公式如下：
![](https://upload-images.jianshu.io/upload_images/5529997-b59763913cea6206.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
（1）修改caffe.proto文件
由于FocalLoss改动较多，因此重新建立一个参数字段，而不是在SoftmaxParamter字段的基础上修改。
编辑src/caffe/proto/caffe.proto文件，添加一个新的FocalLossParameter 字段，并且为其定义功能字段。
```
optional FocalLossParameter focal_loss_param = 145;//需要和自己的protobuf序列号对应，不能产生冲突

message FocalLossParameter {
  enum Type {
    ORIGIN = 0;
     // FL(p_t)  = -(1 - p_t) ^ gama * log(p_t), where p_t = p if y == 1 else 1 - p, whre p = sigmoid(x)
    LINEAR = 1; 
    // FL*(p_t) = -log(p_t) / gama, where p_t = sigmoid(gama * x_t + beta), where x_t = x * y, y is the ground truth label {-1, 1}
  }
  optional Type type   = 1 [default = ORIGIN]; 
  optional float gamma = 2 [default = 2];
  // cross-categories weights to solve the imbalance problem
  optional float alpha = 3 [default = 0.25]; 
  optional float beta  = 4 [default = 1.0];
}
```
（2）导入hpp/cpp/cu文件
将```focal_loss_layer.hpp```文件添加到include/caffe/layers/文件夹下。
将```focal_loss_layer.cpp```文件添加到src/caffe/layers/文件夹下。
将```focal_loss_layer.cu```文件添加到src/caffe/layers/文件夹下。

（3）编译
返回到caffe的根目录，使用make指令(下面几个都可以，任选一个)，即可。
```
 make
 make -j
 make -j16
 make -j32    // 这里j后面的数字与电脑配置有关系，可以加速编译
```
（4）使用方法
```
layer {
  name: "loss_cls"
  type: "FocalLoss"
  bottom: "cls_score"
  bottom: "labels"
  propagate_down: 1
  propagate_down: 0
  top: "loss_cls"
  include { phase: TRAIN }
  loss_weight: 1
  loss_param { ignore_label: -1 normalize: true }
  focal_loss_param { alpha: 0.25 gamma: 2 }
}
```
