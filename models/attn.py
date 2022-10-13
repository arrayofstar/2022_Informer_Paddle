import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask, masked_fill

class FullAttention(nn.Layer):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = paddle.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(F.softmax(scale * scores, axis=-1))
        V = paddle.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V, A)
        else:
            return (V, None)

class ProbAttention(nn.Layer):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3)
        K_expand = K_expand.broadcast_to([B, H, L_Q, L_K, E])  # mf-先增加一个维度，相当于复制，再扩充
        # print('K_expand.shape', K_expand.shape)
        index_sample = paddle.randint(high=L_K, shape=[L_Q, sample_k]) # real U = U_part(factor*ln(L_k))*L_q
        index_sample = paddle.tile(index_sample[None, None, :, :, None], repeat_times=[B, H, 1, 1, E])
        # 构建25的随机数来随机选25个K来计算Q的分布大小
        # K_sample = K_expand[:, :, paddle.arange(L_Q).unsqueeze(1), index_sample, :]  # torch的切片方式与paddle不一样
        # mf-这里的索引采样未必合适
        K_sample = paddle.take_along_axis(K_expand, index_sample, axis=3)
        # print('K_sample', K_sample.shape)
        Q_K_sample = paddle.matmul(Q.unsqueeze(-2), K_sample.transpose([0, 1, 2, 4, 3])).squeeze(-2)
        # print('Q_K_sample', Q_K_sample.shape)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1) - Q_K_sample.sum(-1)/L_K  # 96个Q和25个K之间的关系
        # print('Q_K_sample.max(-1)[0].shape', Q_K_sample.max(-1)[0].shape)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        # Q_reduce = Q[:,:,M_top, :] # factor*ln(L_q)
        Q_reduce = paddle.take_along_axis(Q, M_top[:, :, :, None], axis=-2)
        Q_K = paddle.matmul(Q_reduce, K.transpose([0,1,3,2])) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(axis=-2)   # 对V求均值-交给之前没有选中的V
            contex = V_sum.unsqueeze(-2).broadcast_to([B, H, L_Q, V_sum.shape[-1]]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(axis=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.place)  # 这里还没有改
            # scores.masked_fill_(attn_mask.mask, -np.inf)  # paddle中没有masked_fill_函数因此在ProbMask添加了函数实现
            scores = masked_fill(scores, attn_mask.mask, -np.inf)

        attn = F.softmax(scores, axis=-1) # nn.Softmax(dim=-1)(scores)
        index_in = paddle.tile(index[:, :, :, None],[1, 1, 1, D])
        context_in = paddle.put_along_axis(context_in, index_in, paddle.matmul(attn, V), axis=2)  # mf-这里的替换不太确定

        if self.output_attention:
            attn_one = paddle.ones([B, H, L_V, L_V])/L_V
            index_attn = paddle.tile(index[:, :, :, None], [1, 1, 1, L_V])
            # attns[paddle.arange(B)[:, None, None], paddle.arange(H)[None, :, None], index, :] = attn
            attns = paddle.put_along_axis(attn_one, index_attn, attn, axis=-2)
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = paddle.transpose(queries, (0, 2, 1, 3))
        keys = paddle.transpose(keys, (0, 2, 1, 3))
        values = paddle.transpose(values, (0, 2, 1, 3))

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k) mf-Key里要选的个数
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)   # mf-重点

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        context = paddle.transpose(context, [0, 2, 1, 3])
        return context, attn


class AttentionLayer(nn.Layer):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).reshape([B, L, H, -1])
        keys = self.key_projection(keys).reshape([B, S, H, -1])
        values = self.value_projection(values).reshape([B, S, H, -1])

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = paddle.transpose(out, [0, 2, 1, 3])
        out = out.reshape([B, L, -1])

        return self.out_projection(out), attn
