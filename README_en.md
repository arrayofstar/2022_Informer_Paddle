# 2022_Informer_Paddle

[简体中文](./README_cn.md) |  **English**

2022_PaddleChallenge(7th)-46

## reference papers --《Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting》

## 1、Introduction

This repostiory replicates Informer based on the PaddlePaddle framework. Informer is a new long time series prediction model, which is based on the transformer architecture of  encoder, decoder and attention mechanism.

**[Disadvantages of Transformer:]((https://zhuanlan.zhihu.com/p/480242779))**

1. The time and space complexity of self attention is $O (L ^ 2) $, and L is the sequence length.
2. Memory bottleneck. Stack of encoder/decoder requires a large amount of memory $O (NL ^ 2) $. N is the number of encoder/decoder.
3. The transformer's decoding is step by step, which slows down the speed of inference.

**Informer improvements:**：

1. The `ProbSparse self attention` mechanism be proposed to replace `inner product self attention`, reducing the time and space complexity to $O (LlogL) $.
2. A `self attention dumping` is proposed to highlight the dominating score, shorten the input length of each layer, and reduce the space complexity to $O((2-\epsilon)LlogL)$ .
3. A `generating encoder` is proposed to predict the output. This process only requires one forward step, and the time complexity can be reduced to $O (1) $.

paper: [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)

Reference github: https://github.com/zhouhaoyi/Informer2020

AI Studio Link: https://aistudio.baidu.com/aistudio/projectdetail/4719840

## 2、Retrieval accuracy

### Multivariate forecasting results

| Datasets | seq_len | label_len | pred_len | paddle-MSE          | paddle-MAE |
| -------- | ------- | --------- | -------- |---------------------| ---------- |
| ETTh1  | 48       | 48       | 24       | 0.7087              | 0.5966     |
| ETTh1  | 96       | 48       | 48       | 1.1204              | 0.8598     |
| ETTh1  | 168      | 168      | 168      | 1.1150              | 0.8304     |
| ETTh1  | 336      | 168      | 168      | 1.2381              | 0.8738     |
| ETTh1  | 336      | 336      | 720      | Insufficient memory |            |

### Univariate forecasting results.

| Datasets | seq_len | label_len | pred_len | paddle-MSE | paddle-MAE |
| -------- | ------- | --------- | -------- | ---------- | ---------- |
| ETTh1  | 48       | 48       | 24       | 0.0998     | 0.2529     |
| ETTh1  | 96       | 48       | 48       | 0.1390     | 0.3044     |
| ETTh1  | 168      | 168      | 168      | 0.1404     | 0.3029     |
| ETTh1  | 336      | 168      | 168      | 0.0873     | 0.2278     |
| ETTh1  | 336      | 336      | 720      | Insufficient memory   |            |

## 3、Datasets

The data set is placed in/ Under the data/directory, it comes from the [ETDataset](https://github.com/zhouhaoyi/ETDataset/blob/main/README_CN.md) and [Baidu Cloud Disk](https://pan.baidu.com/s/1wyaGUisUICYHnfkZzWCwyA?_at_=1665205285640#list/path=%2F) (password: 6gan.) provided in the paper source code.

ETDataset：ETTh1.csv、ETTh2.csv、ETTm1.csv、ETTm1.csv

- shape：ETTh1(17420, 8)、ETTh2 (17420, 8)、ETTm1 (69680, 8)、ETTm1(69680, 8)
- data format：csv，The first column is date，other 7 is features data。
- more detial：https://github.com/zhouhaoyi/ETDataset

ECL Datasets：ECL.csv

- shape：ECL(26304, 322)
- data format：csv，The first column is date，other 321 is features data。

Weather Datasets：WTH.csv

- shape：WTH(35064.13)
- data format：csv，The first column is date，other is features data。

## 4、Environment

python == 3.7  paddlepaddle == 2.3.2  pandas == 1.3.5  numpy == 1.21.5  matplotlib ==3.5.2 

## 5、Quick start

```bash
!python main_informer.py
```

## 6、Structure and Detail

The detailed descriptions about the arguments are as following:

| Parameter name   | Description of parameter                                     |
| :--------------- | :----------------------------------------------------------- |
| model            | The model of experiment. This can be set to `informer`, `informerstack`, `informerlight(TBD)` |
| data             | The dataset name                                             |
| root_path        | The root path of the data file (defaults to `./data/`)       |
| data_path        | The data file name (defaults to `ETTh1.csv`)                 |
| features         | The forecasting task (defaults to `M`). This can be set to `M`,`S`,`MS` (M : multivariate predict multivariate, S : univariate predict univariate, MS : multivariate predict univariate) |
| target           | Target feature in S or MS task (defaults to `OT`)            |
| freq             | Freq for time features encoding (defaults to `h`). This can be set to `s`,`t`,`h`,`d`,`b`,`w`,`m` (s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly,  m:monthly).You can also use more detailed freq like 15min or 3h |
| checkpoints      | Location of model checkpoints (defaults to `./checkpoints/`) |
| seq_len          | Input sequence length of Informer encoder (defaults to 96)   |
| label_len        | Start token length of Informer decoder (defaults to 48)      |
| pred_len         | Prediction sequence length (defaults to 24)                  |
| enc_in           | Encoder input size (defaults to 7)                           |
| dec_in           | Decoder input size (defaults to 7)                           |
| c_out            | Output size (defaults to 7)                                  |
| d_model          | Dimension of model (defaults to 512)                         |
| n_heads          | Num of heads (defaults to 8)                                 |
| e_layers         | Num of encoder layers (defaults to 2)                        |
| d_layers         | Num of decoder layers (defaults to 1)                        |
| s_layers         | Num of stack encoder layers (defaults to `3,2,1`)            |
| d_ff             | Dimension of fcn (defaults to 2048)                          |
| factor           | Probsparse attn factor (defaults to 5)                       |
| padding          | Padding type(defaults to 0).                                 |
| distil           | Whether to use distilling in encoder, using this argument means not using distilling (defaults to `True`) |
| dropout          | The probability of dropout (defaults to 0.05)                |
| attn             | Attention used in encoder (defaults to `prob`). This can be set to `prob` (informer), `full` (transformer) |
| embed            | Time features encoding (defaults to `timeF`). This can be set to `timeF`, `fixed`, `learned` |
| activation       | Activation function (defaults to `gelu`)                     |
| output_attention | Whether to output attention in encoder, using this argument means outputing attention (defaults to `False`) |
| do_predict       | Whether to predict unseen future data, using this argument means making predictions (defaults to `False`) |
| mix              | Whether to use mix attention in generative decoder, using this argument means not using mix attention (defaults to `True`) |
| cols             | Certain cols from the data files as the input features       |
| num_workers      | The num_works of Data loader (defaults to 0)                 |
| itr              | Experiments times (defaults to 2)                            |
| train_epochs     | Train epochs (defaults to 6)                                 |
| batch_size       | The batch size of training input data (defaults to 32)       |
| patience         | Early stopping patience (defaults to 3)                      |
| learning_rate    | Optimizer learning rate (defaults to 0.0001)                 |
| des              | Experiment description (defaults to `test`)                  |
| loss             | Loss function (defaults to `mse`)                            |
| lradj            | Ways to adjust the learning rate (defaults to `type1`)       |
| use_amp          | Whether to use automatic mixed precision training, using this argument means using amp (defaults to `False`) |
| inverse          | Whether to inverse output data, using this argument means inversing output data (defaults to `False`) |
| use_gpu          | Whether to use gpu (defaults to `True`)                      |
| gpu              | The gpu no, used for training and inference (defaults to 0)  |
| use_multi_gpu    | Whether to use multiple gpus, using this argument means using mulitple gpus (defaults to `False`) |
| devices          | Device ids of multile gpus (defaults to `0,1,2,3`)           |

## 7、Model information

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
<img src=".\img\Informer_network_detail.png" height = "300" alt="" align=center />
<br><br>

## 8、TODO

At present, although the code has been duplicated in the pad, many details have not been completely handled. The following is recorded as the content of subsequent improvement:

- Single gpu computing capacity: seq_ len=168  label_ len=168  pred_=336  The multivariable calculation shows the following error，the capacity of the graphics card is insufficient, or the settings in some parts of the code reproduction are not adjusted correctly. If you want to achieve the accuracy requirements of reproduction, you must find the problem here.

  > Out of memory error on GPU 0. Cannot allocate 15.503906GB memory on GPU 0, 28.770508GB memory has been allocated and available memory is only 10.815613GB.

- Multi gpu computing problem: AI studio V100 four card is used to test the code, but not work. This problem needs to be solved after the opportunity to contact relevant knowledge.