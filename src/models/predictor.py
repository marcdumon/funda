import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# def predict(model: nn.Module, cont_x: np.array, cat_x: np.array, device: th.device = th.device('cpu'), labels: bool = True) -> np.array:
#     """
#
#     :param model: trained model
#     :param x: x values
#     :param device: cpu or cuda:0 or cuda:1
#     :param labels: if true, execute argmax on y
#     :return: predicted y's. If labels=true then y = indices
#     """
#     cont_x = th.Tensor(cont_x).to(device)
#     cat_x = th.Tensor(cat_x).long().to(device)  # Todo: why .long() here and not during training ???? Anyway, this works
#     model = model.to(device)
#     y = model(cont_x, cat_x)
#     y = F.softmax(y, dim=1)
#     if labels: y = th.argmax(y, dim=1)
#     y = y.detach().numpy()
#     return y


def predict(model, path, df, criterion):
    model = th.load(path + 'model.pth').to('cpu')
    model.eval()

    skip_features = ['m_start', 'm_end', 'q_start', 'sector', 'industry', 'INF_scale', 'label']
    cols = [c for c in df.columns if c not in skip_features]
    data = df[cols].values
    data = th.tensor(data, dtype=th.float)
    with th.no_grad():  # doesn't calculate grads
        result = model(data)
    print(data)
    print(result)
    print(result.shape)
    loss = criterion(data[135], result[135])
    print(data[0], '\n', result[0])
    print(loss)
    df[cols] = result
    df['loss'] = [criterion(data[i], result[i]).item() for i in range(len(data))]
    print(df.head())
    df.to_csv('xxx.csv')

    if __name__ == '__main__':
        pass
