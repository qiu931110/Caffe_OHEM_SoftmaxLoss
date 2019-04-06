> 商汤提出的一种特征交织机制，用来提升目标检测网络对小目标的学习能力，感觉大佬的思想好有创造力啊！！！

论文地址：https://arxiv.org/pdf/1903.11851.pdf
论文源码：https://github.com/hli2020/feature_intertwiner

本文主要介绍下特征交织技术的核心思想，反正我看完论文感觉真的很有创造力！

作者认为我们在进行目标检测任务时，整体的特征空间可以被分组为两组。 一组更可靠，而另一组则不太可靠。 众所周知，在目标检测任务中，由于在网络前向传递（例如，RoI操作）期间会丢失部分信息，低分辨率的目标将更难检测。因此，在目标检测任务中，作者认为将高分辨率的目标得到的特征视为可靠的特征集合，将低分辨率的目标得到的特征视为不太可靠的特征集合。可靠的集合可以指导不太可靠的集合的特征学习，具体而言，本文提出的特征交织策略旨在最小化两组特征之间的分布差异，使得特征更加紧凑，使得低分辨率的目标能够摒弃其空间属性的不足，去学习高分辨率特征的强分辨能力。

在本文中，作者的方法是在FasterRCNN的基础上进行修改。因此，我们先用一段话来简单描述下Faster RCNN是如何实现目标检测的。在Faster RCNN中，输入图像首先被送到骨干网络用以提取特征; 得到的特征分两路走（两路的特征是一样的，相当于骨干网络提取的特征被用于两个部分的输入），其中一路输入区域建议网络（RPN）用以生成潜在的区域建议框，这些框是可能包含对象的几个候选矩形框的集合， 这些矩形框的大小各不相同，RPN网络输出的是几个坐标值；另外一路结合RPN网络输出的坐标值得到区域特征，并将其扭曲成相同的空间大小（通过RoI Pooling）。 最后，利用浅层CNN来对目标进行定位和分类。


接下来，我们从宏观的角度来讲下如何将特征交织的概念应用到目标检测框架中。 下图为本文提出的InterNet的网络结构。
![](https://upload-images.jianshu.io/upload_images/5529997-6a97bf66182f40eb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

因为本文用到的骨干网络是结合了特征金字塔改进的ResNet模型，因此该模型会输出多尺度特征进行后续检测。而我们都知道特征图的大小和网络的深度成反比，在浅层区域特征图大，在深层区域特征图小，因此基于特征映射的空间我们可以将网络划分为若干级别。对于每个级别L，我们都会得到对应于这个级别L的区域建议框，并且我们将区域建议框分为两类：一类是可靠区域建议框，其大小大于RoI池化层的输出大小，另一类是不可靠区域特征。这两组分别对应于可靠特征和不太可靠特征（如何获取请看文章最后注解）。

如上图所示，表示的是其中一个层级L的训练过程。对于当前层PL我们首先将不可靠建议框输入到ROI中并经过make-up层进行修饰，该层用于补充其在ROI过程中的信息丢失，该过程是通过上图黄色框中的特征交织器实现的。其中红色框表示的是最优散度分布，用来对齐不同层级间的信息。Pm|L表示的是当前层级L中的可靠建议框层。m表示的是比层L更高的所有层（具体获取请看注解）。

特征交织器本质上是用于评估可靠特征和不可靠特征之间数据分布的差异。对于可靠集合，特征交织器的输入直接是可靠对象特征Pm|L层经过RoI层的结果，其对应于更高级别/分辨率的样本。 对于不太可靠集合，输入是PL层经过ROI后，又通过make-up补给层的输出。 两个输出都被输入到critic模块，该模块包含两个卷积，其作用是将特征转换为更大的channel空间，并将特征尺寸减小到一个，因此不考虑空间信息，并用L2损失来衡量两者之间的差异。这种方式就可以使得不太可靠集合中的小目标能够脱离空间信息的本质，去学习到更高层的特征表示！（听起来好有道理是不是！）最终的网络loss是由各个层级之间的L2特征交织loss和原始检测网络中的loss之和。


本文还对各种最优散度分布策略进行了介绍，本文就不多做分析了，对于其中的各种实现细节在没有看源码之前我也很难理顺！所以感兴趣的同学可以自行阅读论文和源码~

>注解：
这边来讲下，对于每个Level来说，如何对大，小两个集的分配。
本文中，采用基于特征金字塔改进的ResNet模型提取特征，它生成五种不同大小的特征图，以作为后续RPN和检测分支的输入。将这五个级别的特征图的索引表示为L = {1，2，3，4，5}，且对应的特征标志位PL。特征层级越浅特征中物体的目标信息丰富，小目标的信息也很丰富。因此层级越小的层我们获取尽可能多的区域建议框。下图展示了不同层级的区域建议框个数，以及将区域建议框分为大框和小框的阈值，下面会详细介绍下框中各个数值的含义。
![](https://upload-images.jianshu.io/upload_images/5529997-a536b257d22073cc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
proposal行：如上图所示，四个层级总共有302+36+54+8=400个区域建议框，一般而言我们会对一个512分辨率的输入图片取200个区域建议框，而一般batch设为2，因此一个batch中一般会有400个区域建议框（这就是这400的由来了）。
threshold行：那么我们已经知道了每个层级所拥有的区域建议框数量后，我们就要来将每一个层级的区域建议框分成可靠的和不可靠的，也就是大区域框和小区域框。本文的区分策略也很简单，确定一个面积阈值，若当前区域框面积大于阈值的为大框，反之则为小框。那么上图中的阈值是如何得到的呢？对于512分辨率的输入图片，最终经过ROI层的输出得到的特征图尺寸为边长为14的正方形。而各层级特征图尺寸的大小依次如下：
层级2：128    阈值=（14/128）\*(14/128)=0.012
层级3：64 阈值=（14/64）\*(14/64)=0.0479
层级4：32 阈值=（14/32）\*(14/32)=0.1914
层级5：16 阈值=（14/16）\*(14/16)=0.7657
所以直观来说若当前区域建议框在各自层级的面积占比小于最终ROI后的输出，那么我们认为是小框，反之则为大框。因为如果占比小的话，该框最终会通过上采样才能达到ROI的输出，会使得信息丢失，所以小框需要帮助！
below# / above#：这一行的数据就是基于上述得到的阈值对每个层级的大小框的划分，其中below部分是小框，above部分是大框。
intertwiner small/intertwiner large行：对于网络中的L曾，我们将小交织框定义为已经分配给L的L框，将大建议定义为高于L层的所有层中的框的总和。
说实话，这部分我是没有很懂的，既然特征交织中的可靠和不可靠是这么分的，那为何还要花篇幅介绍通过阈值区分大小建议框。这一块估计只有后续阅读源码才能真正理解了。
