'''复制到informer_paddle文件目录下运行'''

import random
import numpy as np
import paddle

from models.model import Informer
from models.embed import DataEmbedding, TokenEmbedding
from reprod_log import ReprodLogger

if __name__ == "__main__":
    np.random.seed(7)
    random.seed(7)

    # def logger
    reprod_logger = ReprodLogger()

    # init model
    model = Informer(enc_in=7, dec_in=7, c_out=1, seq_len=96, label_len=48, out_len=24, embed='timeF')

    # read or gen fake data
    fake_x = np.load("../informer_paddle/log_reprod/3.1.3 模型组网正确性验证/fake_data/fake_x.npy")
    fake_x = paddle.to_tensor(fake_x).astype('float32')
    fake_x_mark = np.load("../informer_paddle/log_reprod/3.1.3 模型组网正确性验证/fake_data/fake_x_mark.npy")
    fake_x_mark = paddle.to_tensor(fake_x_mark).astype('float32')
    fake_y = np.load("../informer_paddle/log_reprod/3.1.3 模型组网正确性验证/fake_data/fake_y.npy")
    fake_y = paddle.to_tensor(fake_y).astype('float32')
    fake_y_mark = np.load("../informer_paddle/log_reprod/3.1.3 模型组网正确性验证/fake_data/fake_y_mark.npy")
    fake_y_mark = paddle.to_tensor(fake_y_mark).astype('float32')

    model.eval()
    # # forward-check model
    # out = model(fake_x, fake_x_mark, fake_y, fake_y_mark)
    # reprod_logger.clear()
    # reprod_logger.add("forward-check model", out.cpu().detach().numpy())
    # reprod_logger.save("../informer_paddle/log_reprod/3.1.3 模型组网正确性验证/forward_out_paddle.npy")


    # # forward-check enc_embedding
    # enc_embedding = DataEmbedding(c_in=7, d_model=512, embed_type='timeF', freq='h', dropout=0.0)
    # enc_out = enc_embedding(fake_x, fake_x_mark)
    # reprod_logger.clear()
    # reprod_logger.add("forward-check enc_embedding model", enc_out.cpu().detach().numpy())
    # reprod_logger.save("../informer_paddle/log_reprod/3.1.3 模型组网正确性验证/forward_enc_out_paddle.npy")
    #
    # # forward-check enc_embedding
    # enc_embedding = DataEmbedding(c_in=7, d_model=512, embed_type='timeF', freq='h', dropout=0.0)
    # enc_out = enc_embedding(fake_x, fake_x_mark)
    # reprod_logger.clear()
    # reprod_logger.add("forward-check enc_embedding model", enc_out.cpu().detach().numpy())
    # reprod_logger.save("../informer_paddle/log_reprod/3.1.3 模型组网正确性验证/forward_enc_out_paddle.npy")

    # forward-check enc_embedding-TokenEmbedding
    enc_tokenembedding = TokenEmbedding(c_in=7, d_model=512)
    enc_tokenembedding_out = enc_tokenembedding(fake_x)
    reprod_logger.clear()
    reprod_logger.add("forward-check enc_embedding-TokenEmbedding", enc_tokenembedding_out.cpu().detach().numpy())
    reprod_logger.save(
        "../informer_paddle/log_reprod/3.1.3 模型组网正确性验证/forward_enc_tokenembedding_out_paddle.npy")

    # reprod reference
    # reprod_logger.add("logits", enc_out.cpu().detach().numpy())
    # reprod_logger.save("../informer_paddle/log_reprod/3.1.3 模型组网正确性验证/forward_paddle.npy")
    print("运行结束")