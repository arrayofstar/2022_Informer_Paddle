import math

import paddle
import paddle.nn as nn


class PositionalEmbedding(nn.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = paddle.zeros([max_len, d_model], dtype='float32')  # mf-这里不确定
        pe.stop_gradient = True  # mf-这里不确定

        position = paddle.arange(0, max_len, dtype='float32').unsqueeze(1)
        div_term = (paddle.arange(0, d_model, 2, dtype='float32') * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        para = x.shape[1]
        return self.pe[:, :x.shape[1], :]  # mf-为什么要取最后96个pe？


class TokenEmbedding(nn.Layer):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if paddle.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1D(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias_attr=True,
                                   weight_attr=nn.initializer.KaimingNormal())

    def forward(self, x):
        x = x
        x = paddle.transpose(x, [0, 2, 1])  # mf-这里还没有找到对应的函数 x.permute(0, 2, 1)
        x = self.tokenConv(x)
        x = paddle.transpose(x, [0, 2, 1])
        return x


class FixedEmbedding(nn.Layer):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = paddle.zeros([c_in, d_model])
        w.stop_gradient = True

        position = paddle.arange(0, c_in, dtype='float32').unsqueeze(1)
        div_term = (paddle.arange(0, d_model, 2, dtype='float32') * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = paddle.sin(position * div_term)
        w[:, 1::2] = paddle.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = paddle.create_parameter(shape=w.shape, dtype=str(w.numpy().dtype))

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Layer):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Layer):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)  # mf-感觉有问题

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Layer):
    def __init__(self, c_in, d_model, embed_type='timeF', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x1 = self.value_embedding(x)  # 数值编码
        x2 = self.position_embedding(x)  # 位置编码
        x3 = self.temporal_embedding(x_mark)  # 时间编码
        x = x1 + x2 + x3

        return self.dropout(x)
