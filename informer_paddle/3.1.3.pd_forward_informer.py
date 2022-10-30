'''复制到informer_paddle文件目录下运行'''

import numpy as np
import paddle
from models.model import Informer
from reprod_log import ReprodLogger

if __name__ == "__main__":
    paddle.set_device("gpu")

    # def logger
    reprod_logger = ReprodLogger()

    # init model
    model = Informer(enc_in=7, dec_in=7, c_out=1, seq_len=96, label_len=48, out_len=24, embed='timeF')

    # read or gen fake data
    fake_x = np.load("../informer_paddle/log_reprod/3.1.3 模型组网正确性验证/fake_data/fake_x.npy")
    fake_x = paddle.to_tensor(fake_x)
    fake_x_mark = np.load("../informer_paddle/log_reprod/3.1.3 模型组网正确性验证/fake_data/fake_x_mark.npy")
    fake_x_mark = paddle.to_tensor(fake_x_mark).astype('float32')
    fake_y = np.load("../informer_paddle/log_reprod/3.1.3 模型组网正确性验证/fake_data/fake_y.npy")
    fake_y = paddle.to_tensor(fake_y)
    fake_y_mark = np.load("../informer_paddle/log_reprod/3.1.3 模型组网正确性验证/fake_data/fake_y_mark.npy")
    fake_y_mark = paddle.to_tensor(fake_y_mark).astype('float32')

    # forward
    model.eval()
    out = model(fake_x, fake_x_mark, fake_y, fake_y_mark)

    # reprod
    reprod_logger.add("logits", out.cpu().detach().numpy())
    reprod_logger.save("../informer_paddle/log_reprod/3.1.3 模型组网正确性验证/forward_paddle.npy")
    print("运行结束")