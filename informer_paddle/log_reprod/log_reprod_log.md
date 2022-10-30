# 论文复现对齐日志

【日志内容相关解释】

该日志参考[论文复现赛指南-NLP方向](https://github.com/PaddlePaddle/models/blob/release/2.3/tutorials/article-implementation/ArticleReproduction_NLP.md#1)
的第三小节进行编写，文档编号直接由3.1开始，实现对应，方便对比学习。原文档有以下缺点：

- 提供了一个良好的复现架构，但文档和代码中的step对应关系较差，不易读者理解。
- 需要额外装库（可能是的没完全理解文章意图，本人参考学习时pip安装了paddlenlp和hugging face 的transformers。）

> 基于上述原因，想参考前人文档来实现一下时间序列模型informer的对齐过程(正好官网上没有，也许时间序列问题比较小众？)
> ，希望一切顺利，下面正文

原文档更新日志：

- 3.1.1 PyTorch-PaddlePaddle API映射表：地址更新

### 3.1 模型结构对齐

对齐模型结构时，一般有3个主要步骤：

- 网络结构代码转换
- 权重转换
- 模型组网正确性验证

#### 3.1.1 网络结构代码转换

**【基本流程】**

由于PyTorch的API和PaddlePaddle的API非常相似，可以参考[PyTorch-PaddlePaddle API映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html#id1)
，组网部分代码直接进行手动转换即可。

**【注意事项】**

如果遇到PaddlePaddle没有的API，可以尝试用多种API来组合，也可以给PaddlePaddle团队提[ISSUE](https://github.com/PaddlePaddle/Paddle/issues)
，获得支持。

**【实战】**

Informer网络结构的PyTorch实现: 2022_Informer_Paddle\informer_pytroch\models\model.py

对应转换后的PaddlePaddle实现: 2022_Informer_Paddle\informer_paddle\models\model.py

#### 3.1.2 权重转换

**【基本流程】**

组网代码转换完成之后，需要对模型权重进行转换，如果PyTorch
repo中已经提供权重，那么可以直接下载并进行后续的转换；如果没有提供，则可以基于PyTorch代码，随机生成一个初始化权重(
定义完model以后，使用`torch.save()` API保存模型权重)，然后进行权重转换。

**【注意事项】**

在权重转换的时候，需要注意`paddle.nn.Linear`以及`paddle.nn.BatchNorm2D`
等API的权重保存格式和名称等与PyTorch稍有diff，具体内容可以参考`4.1章节`。

**【实战】**

Informer中没有预训练，所以不需要下载预训练模型。但是可以通过运行源码获取保存的参数文件，位置为*
informer_pytroch/checkpoints/informer_XXXX（略写）/checkpoint.pth* 。`运行torch2paddle.py代码时，参数文件复制到同级目录下。`

运行完成之后，会在当前目录生成`model_state.pdparams`文件，即为转换后的PaddlePaddle预训练模型。

#### 3.1.3 模型组网正确性验证

**【基本流程】**

1. 定义PyTorch模型，加载权重，固定seed，基于numpy生成随机数，转换为PyTorch可以处理的tensor，送入网络，获取输出，使用reprod_log保存结果。
2. 定义PaddlePaddle模型，加载权重，固定seed，基于numpy生成随机数，转换为PaddlePaddle可以处理的tensor，送入网络，获取输出，使用reprod_log保存结果。
3. 使用reprod_log排查diff，小于阈值，即可完成自测。

**【注意事项】**

- 模型在前向对齐验证时，需要调用`model.eval()`方法，保证组网中的随机量被关闭，比如BatchNorm、Dropout等。
- 给定相同的输入数据，为保证可复现性，如果有随机数生成，固定相关的随机种子。
- 输出diff可以使用`np.mean(np.abs(o1 - o2))`
  进行计算，一般小于1e-6的话，可以认为前向没有问题。如果最终输出结果diff较大，可以使用二分的方法进行排查，比如说BERT，包含1个embdding层、12个transformer-block以及最后的MLM
  head层，那么完成模型组网和权重转换之后，如果模型输出没有对齐，可以尝试输出中间某一个transformer-block的tensor进行对比，如果相同，则向后进行排查；如果不同，则继续向前进行排查，以此类推，直到找到导致没有对齐的操作。

**【实战】**

