# 2022-Informer_Paddle

2022年飞桨论文复现挑战赛（第七期）第46题——2022_PaddleChallenge(7th)-46

## 一、简介

本项目基于paddlepaddle框架复现Informer，Informer是一种新的长时间序列预测模型，基于编码器和解码器的transformer架构。

**[Transformer的3个缺点：](https://zhuanlan.zhihu.com/p/480242779)**

1. self-attention的时间和空间复杂度是$O(L^2)$，L 为序列长度。
2. memory瓶颈。stack of encoder/decoder需要大量memory $O(NL^2)$ ，N是encoder/decoder数量。
3. transformer的decoding是step-by-step的，使得推理(inference)的速度变慢。

**Informer的改进：**

1. 提出`ProbSparse self-attention`机制来替换`inner product self-attention`，使得时间和空间复杂度降为$O(LlogL)$。
2. 提出`self-attention distilling`来突出dominating score，缩短每一层输入的长度，降低空间复杂度到$O((2-\epsilon)LlogL)$ 。
3. 提出`generative decoder`来预测输出，此过程仅需要one forward step，时间复杂夫降为 $O(1)$。

**论文：**

- [[1] Haoyi Zhou, Shanghang Zhang, Jieqi Peng et al. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021, Virtual Conference. (35)12, 11106-11115, 2021](https://arxiv.org/abs/2012.07436)

**参考项目：**

- https://github.com/zhouhaoyi/Informer2020

**项目aistudio地址：**

- 

## 二、复现精度

non
## 三、数据集

数据集来源于论文源码中的所提供的[ETDataset](https://github.com/zhouhaoyi/ETDataset/blob/main/README_CN.md)和[百度云盘](https://pan.baidu.com/s/1wyaGUisUICYHnfkZzWCwyA?_at_=1665205285640#list/path=%2F),password: 6gan.

Weather数据集：WTH.csv

- 数据集大小：数据集维度为（35064.13）
- 数据格式：csv格式，第一列为date，即时间序列特征，后12列为特征，训练时可选取一列或多列作为标签（预测目标）

## 四、环境依赖

略

## 五、快速开始

略

## 六、代码结构与详细说明

关于参数的详细描述如下：

| 参数名称         | 参数描述                                                     |
| :--------------- | :----------------------------------------------------------- |
| model            | 实验中的模型。可设置为 `informer`, `informerstack`, `informerlight(TBD)` |
| data             | 数据集名称                                                   |
| root_path        | 数据文件的根目录                                             |
| data_path        | 数据文件的文件名                                             |
| features         | 预测任务（默认为`M`）。可设置为`M`,`S`,`MS`(M:多变量预测多变量，S:单变量预测单变量，MS：多变量预测单变量) |
| target           | 在S和MS任务中的目标特征（默认为OT）                          |
| freq             | 时间特征编码的频率（默认为h），可设置为s,t,h,d,b,w,m（s:秒,t:分,h:小时,d:天,b:工作日,w:周,m:月)，也可以添加细节，如15min 或者 3h |
| checkpoints      | 模型检查点的位置（默认为`./checkpoints/`）                   |
| seq_len          | Informer编码器中输入序列的长度（默认为96）                   |
| label_len        | Informer解码器中预热的长度（默认为48）                       |
| pred_len         | 预测序列的长度（默认为24）                                   |
| enc_in           | 编码器中输入特征的数量（默认为7）                            |
| dec_in           | 解码器中输入特征的数量（默认为7）                            |
| c_out            | 输出特征的数量（默认为7）                                    |
| d_model          | 模型的维度：`这里还不是很清楚`（默认为512）                  |
| n_heads          | Transformer结构中头的数量（默认为8）                         |
| e_layers         | 编码器中层的数量（默认为2）                                  |
| d_layers         | 解码器中层的数量（默认为1）                                  |
| s_layers         | 堆叠解码器层的数量（默认为3,2,1）                            |
| d_ff             | 全连接层中的维度（默认为2048）                               |
| factor           | Probsparse attn 中的参数因子：这里涉及了QKV筛选的数量（默认为5） |
| padding          | padding的类型（默认为0）                                     |
| distil           | 是否使用模型蒸馏：意思是在编码器中进行类似于降维的操作，例如96=>48（默认为True） |
| dropout          | dropout的比例（默认为0.05）                                  |
| attn             | 编码器中的注意力机制使用方式（默认为`prob`，即作者提出的优化策略），可设置为prob（Informer模型）或full（Transformer） |
| embed            | 时间特征编码（默认为`timeF`），可设置为timeF`, `fixed`, `learned |
| activation       | 激活函数（默认为`gelu`）                                     |
| output_attention | 是否在编码器输出的时候使用注意力机制（默认为False）          |
| do_predict       | 是否预测不可见的未来数据（默认为False）                      |
|                  |                                                              |



## 七、模型信息