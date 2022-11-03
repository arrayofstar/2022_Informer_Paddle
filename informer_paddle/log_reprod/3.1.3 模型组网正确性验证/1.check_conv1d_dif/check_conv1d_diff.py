from reprod_log import ReprodDiffHelper
import random
import numpy as np
import paddle
import torch
import math

from reprod_log import ReprodLogger

class TokenEmbedding_pytorch(torch.nn.Module):
    def __init__(self, c_in, d_model, reprod=''):
        super(TokenEmbedding_pytorch, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = torch.nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=1, padding_mode='circular')

        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                torch.nn.init.constant_(m.bias, 0.0)
        # for m in self.modules():  # 当把权重都置为0.5时，可通过check
        #     if isinstance(m, torch.nn.Conv1d):
        #         torch.nn.init.constant_(m.weight,0.5)
        #         torch.nn.init.constant_(m.bias,0.0)

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class TokenEmbedding_paddle(paddle.nn.Layer):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding_paddle, self).__init__()
        _weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingNormal(nonlinearity='leaky_relu'))
        # 当把权重都置为0.5时，可通过check
        # _weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(paddle.full([512,7,3],0.5)))
        _bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(paddle.full([512],0)))
        self.tokenConv = paddle.nn.Conv1D(in_channels=c_in, out_channels=d_model,
                                          kernel_size=3, padding=1, padding_mode='circular',
                                          weight_attr=_weight_attr, bias_attr=_bias_attr)
        a = 1

    def forward(self, x):
        out = paddle.transpose(x, [0, 2, 1])
        out = self.tokenConv(out)
        out = paddle.transpose(out, [0, 2, 1])
        return out


if __name__ == "__main__":
    np.random.seed(7)
    random.seed(7)
    # init logger
    reprod_logger = ReprodLogger()

    # paddle
    fake_x = np.load("../fake_data/fake_x.npy")
    fake_x = paddle.to_tensor(fake_x).astype('float32')
    model_pd = TokenEmbedding_paddle(7, 512)
    model_pd.eval()
    out = model_pd(fake_x)
    reprod_logger.clear()
    reprod_logger.add("forward-check model", out.cpu().detach().numpy())
    reprod_logger.save("../1.check_conv1d_dif/TokenEmbedding_paddle_out.npy")


    # pytorch
    fake_x = np.load("../fake_data/fake_x.npy")
    fake_x = torch.from_numpy(fake_x).float()
    model_pt = TokenEmbedding_pytorch(7, 512)
    model_pt.eval()
    out = model_pt(fake_x)
    reprod_logger.clear()
    reprod_logger.add("forward-check model", out.cpu().detach().numpy())
    reprod_logger.save("../1.check_conv1d_dif/TokenEmbedding_pytorch_out.npy")

    # init ReprodDiffHelper
    diff_helper = ReprodDiffHelper()

    # forward-check enc_embedding-TokenEmbedding
    torch_info = diff_helper.load_info("./TokenEmbedding_paddle_out.npy")
    paddle_info = diff_helper.load_info("./TokenEmbedding_pytorch_out.npy")

    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="forward_diff.log")


