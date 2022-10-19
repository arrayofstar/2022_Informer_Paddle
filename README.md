# 2022-Informer_Paddle

2022年飞桨论文复现挑战赛（第七期）第46题——2022_PaddleChallenge(7th)-46

**简体中文** | [English](./README_en.md)

------------------------------------------------------------------------------------------

<p align="center">
  <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
  <a href=""><img src="https://img.shields.io/badge/paddlepaddle-2.3.0+-aff.svg"></a>
</p>


## 一、简介

本项目基于paddlepaddle框架复现Informer，Informer是一种新的长时间序列预测模型，基于编码器、解码器和注意力机制的Transformer架构。

**[Transformer的3个缺点：](https://zhuanlan.zhihu.com/p/480242779)**

1. self-attention的时间和空间复杂度是$O(L^2)$，L 为序列长度。
2. memory瓶颈。堆叠encoder/decoder需要大量内存$O(NL^2)$ ，N是encoder/decoder的数量。
3. transformer的解码过程(decoding)是一步接一步(step-by-step)的，使得推理(inference)的速度变慢。

**Informer的改进：**

1. 提出`ProbSparse self-attention`机制来替换`inner product self-attention`，使得时间和空间复杂度降为$O(LlogL)$。
2. 提出`self-attention distilling`来突出dominating score，缩短每一层输入的长度，降低空间复杂度到$O((2-\epsilon)LlogL)$ 。
3. 提出`generative decoder`来预测输出，此过程仅需要one forward step，时间复杂夫降为 $O(1)$。

**论文：**

- [[1] Haoyi Zhou, Shanghang Zhang, Jieqi Peng et al. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021, Virtual Conference. (35)12, 11106-11115, 2021](https://arxiv.org/abs/2012.07436)

**参考项目：**

- https://github.com/zhouhaoyi/Informer2020

**项目AIstudio地址：**

- https://aistudio.baidu.com/aistudio/projectdetail/4719840

## 二、复现精度

### 多变量预测结果 - Multivariate forecasting results

| 数据集 | 序列长度 | 标签长度 | 预测长度 | paddle-MSE | paddle-MAE |
| ------ | -------- | -------- | -------- | ---------- | ---------- |
| ETTh1  | 48       | 48       | 24       | 0.8075     | 0.6871     |
| ETTh1  | 96       | 48       | 48       | 1.1301     | 0.8598     |
| ETTh1  | 168      | 168      | 168      | 1.1985     | 0.8621     |

### 单变量预测结果 - Univariate forecasting results.

| 数据集 | 序列长度 | 标签长度 | 预测长度 | paddle-MSE | paddle-MAE |
| ------ | -------- | -------- | -------- | ---------- | ---------- |
| ETTh1  | 48       | 48       | 24       | 0.0998     | 0.2529     |
| ETTh1  | 96       | 48       | 48       | 0.1390     | 0.3044     |
| ETTh1  | 168      | 168      | 168      | 0.1307     | 0.2905     |
| ETTh1  | 336      | 168      | 168      | 0.0768     | 0.2158     |
|        |          |          |          |            |            |
## 三、数据集

数据集置于./data/目录之下， 来源于论文源码中的所提供的[ETDataset](https://github.com/zhouhaoyi/ETDataset/blob/main/README_CN.md)和[百度云盘](https://pan.baidu.com/s/1wyaGUisUICYHnfkZzWCwyA?_at_=1665205285640#list/path=%2F),password: 6gan.

ETDataset：ETTh1.csv、ETTh2.csv、ETTm1.csv、ETTm1.csv

- 数据集维度：ETTh1(17420, 8)、ETTh2 (17420, 8)、ETTm1 (69680, 8)、ETTm1(69680, 8)
- 数据格式：csv格式，第一列为时间特征(date)，后7列为数据特征。
- 详细链接：https://github.com/zhouhaoyi/ETDataset

ECL数据集：ECL.csv

- 数据集维度：ECL(26304, 322)
- 数据格式：csv格式，第一列为时间特征(date)，后321列为数据特征。

Weather数据集：WTH.csv

- 数据集维度：WTH(35064.13)
- 数据格式：csv格式，第一列为时间特征(date)，后12列为数据特征

## 四、环境依赖

参考环境：python == 3.7  paddlepaddle == 2.3.2  pandas == 1.3.5  numpy == 1.21.5  matplotlib ==3.5.2  

## 五、快速开始

```
! python main_informer.py
```

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
| d_model          | 模型的维度（默认为512）                                      |
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
| mix              | 是否在生成解码器中使用混合注意力，使用此参数意味着不使用混合注意力（默认为Ture） |
| cols             | 数据文件中作为输入特征的列                                   |
| num_workers      | Dataloader中的num_works参数（默认为0）                       |
| itr              | 实验重复次数（默认为2）                                      |
| train_epochs     | 训练的epochs次数（默认为6）                                  |
| batch_size       | 训练输入数据的batch大小（默认为32）                          |
| patience         | 训练提前终止的参数-Early stopping patience（默认为3）        |
| learning_rate    | 优化器的学习率                                               |
| des              | 实验的描述（默认为test）                                     |
| loss             | 损失函数（默认为mse）                                        |
| lradj            | 学习率调整方法（默认为type1）                                |
| use_amp          | 是否使用自动混合精度训练，使用此参数意味着使用amp（默认为False） |
| inverse          | 是否翻转输出数据，使用此参数意味着输出数据将由归一化数据反转为正常数值区间的数据（默认为False），设置为True，mse会不同 |
| use_gpu          | 使用是否gup（默认为True）                                    |
| gpu              | gpu的编号，用于训练和预测（默认为0）                         |
| use_multi_gpu    | 是否使用多个gpu，使用此参数意味着使用多卡进行计算（默认为False） |
| devices          | 多个gpu时的显卡ID（默认为 0,1,2,3）                          |

## 七、模型信息

模型基于Transformer的架构，新增了ProbSparse Attention.

自注意力得到会形成长尾分布，其中"激活"的q值有较高的得分，“懒惰” 的q值为拖尾。作者设计了ProbSparse Attention来选择"激活"的q值，而忽略“懒惰"的q值，这样就简化了计算，加快了速度。

<p align="center">
<img src=".\img\informer.png" height = "360" alt="" align=center />
<br><br>
<b>Figure 1.</b> The architecture of Informer.
</p>
<p align="center">
<img src=".\img\Informer_encoder.png" height = "230" alt="" align=center />
<br><br>
<b>Figure 2.</b> The single stack in Informer's encoder.
</p>

<p align="center">
<b>Table 1.</b> The Informer network components in details
<br><br>
<img src=".\img\Informer_network_detail.png" height = "300" alt="" align=center />
</p>

## 8、后面的任务

论文复现的要求：

> univariate设定下，在ETTh, ETTm，Weather和ECL数据集上的MAE，在最长区间分别达到 0.431， 0.644， 0.466, 0.608; 
> 在multiplevariate设定下, 在ETTh, ETTm，Weather和ECL数据集上的MAE，在最长区间分别达到 1.473， 0.926， 0.731, 0.548

目前虽然完成了代码的paddle复现，但是还有很多细节没有完全处理好，记录如下，作为后续的改进内容：

- 单卡计算容量问题：seq_len=168  label_len=168  pred_len=336 的多变量计算时显示如下错误：应该是显卡的容量不足，或者是代码复现中部分地方的设置没有调整正确。如果想达到复现的精度要求就必须找到这里的问题。

  > ```
  > Out of memory error on GPU 0. Cannot allocate 15.503906GB memory on GPU 0, 28.770508GB memory has been allocated and available memory is only 10.815613GB.
  > ```

- 多卡计算问题：使用了百度飞桨V100四卡测试代码效果，发现多卡运行的时候会出问题，这个问题需要后续有机会接触到相关的知识再解决。

