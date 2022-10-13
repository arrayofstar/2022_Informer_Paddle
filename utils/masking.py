import paddle

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with paddle.no_grad():
            self._mask = paddle.triu(paddle.ones(mask_shape), diagonal=1)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        # _mask = paddle.triu(paddle.ones([L, scores.shape[-1]], dtype='bool'), diagonal=1)  # mf-后面不支持bool类型
        _mask = paddle.triu(paddle.ones([L, scores.shape[-1]]), diagonal=1)
        _mask_ex = _mask[None, None, :].broadcast_to([B, H, L, scores.shape[-1]])
        # mf-方法paddle.take_along_axis不支持bool数据类型，所以这里用的是 0,1
        indicator = paddle.take_along_axis(_mask_ex, index[:, :, :, None], axis=-2)
        self._mask = indicator.reshape(scores.shape)  # mf-删除了to(device)，reshape替换view

    @property
    def mask(self):
        return self._mask.astype('bool')

def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    out = paddle.where(mask, y, x)
    return out