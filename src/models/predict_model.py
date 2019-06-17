import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def predict(model: nn.Module, cont_x: np.array, cat_x: np.array, device: th.device = th.device('cpu'), labels: bool = True) -> np.array:
    """

    :param model: trained model
    :param x: x values
    :param device: cpu or cuda:0 or cuda:1
    :param labels: if true, execute argmax on y
    :return: predicted y's. If labels=true then y = indices
    """
    cont_x = th.Tensor(cont_x).to(device)
    cat_x = th.Tensor(cat_x).long().to(device)  # Todo: why .long() here and not during training ???? Anyway, this works
    model = model.to(device)
    y = model(cont_x, cat_x)
    y = F.softmax(y, dim=1)
    if labels: y = th.argmax(y, dim=1)
    y = y.detach().numpy()
    return y
