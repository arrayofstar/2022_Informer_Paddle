import numpy as np
import pandas as pd
from informer_paddle.utils import time_features

def gen_fake_data(seed):
    # 以MS任务为例，输入特征为4列，最后一列为标签
    np.random.seed(seed)
    # 构造特征数据，标签数据，时间特征数据
    data_x = np.random.random(size=(1, 1000, 7)).astype(np.float32)
    data_y = data_x
    df_stamp = pd.DataFrame(
        {
            'date': pd.date_range('2021-01-01', periods=500, freq='1h')
        }
    )
    data_stamp = time_features(df_stamp, timeenc=1, freq='h')

    # 直接从第一个索引开始切片
    fake_x = []
    fake_y = []
    fake_x_mark = []
    fake_y_mark = []
    batch_size = 32
    seq_len = 96
    label_len = 48
    pred_len = 48
    for index in range(batch_size):
        s_begin = index
        s_end = s_begin + seq_len
        r_begin = s_end - label_len
        r_end = r_begin + label_len + pred_len
        x = data_x[:, s_begin:s_end, :]
        y = data_y[:, r_begin:r_end]
        x_mark = data_stamp[s_begin:s_end]
        y_mark = data_stamp[r_begin:r_end]
        fake_x.append(x)
        fake_y.append(y)
        fake_x_mark.append(np.expand_dims(x_mark, axis=0))
        fake_y_mark.append(np.expand_dims(y_mark, axis=0))
    fake_x = np.vstack(fake_x)
    fake_y = np.vstack(fake_y)
    fake_x_mark = np.vstack(fake_x_mark)
    fake_y_mark = np.vstack(fake_y_mark)

    np.save("fake_x.npy", fake_x)
    np.save("fake_x_mark.npy", fake_x_mark)
    np.save("fake_y.npy", fake_y)
    np.save("fake_y_mark.npy", fake_y_mark)
    print(fake_x.shape,fake_x_mark.shape,fake_y.shape,fake_y_mark.shape)

if __name__ == "__main__":
    gen_fake_data(seed=7)
