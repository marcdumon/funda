# --------------------------------------------------------------------------------------------------------
# 2019/04/18
# 0_ml_project_template - model.py
# md
# --------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Todo: See: https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/
class Linear3Layers(nn.Module):
    """
    Simple 3 layer linear neural network.
        Tensors:
            layer1:
                * In: [n_features]
                * W: [n_features, n_layer1]
                * B: [n_layer1]
                * Out: [n_layer1]
            layer2:
                * In: [n_layer1]
                * W: [n_layer1, n_layer2]
                * B: [n_layer2]
                * Out: [n_layer2]
            layer3:
                * In: [n_layer2]
                * W: [n_layer2, n_classes]
                * B: [n_classes]
                * Out: [n_classes]
    """

    def __init__(self, n_features, n_layer1, n_layer2, n_classes, activation='relu'):
        super(Linear3Layers, self).__init__()

        self.fc1 = nn.Linear(n_features, n_layer1)
        self.fc2 = nn.Linear(n_layer1, n_layer2)
        self.fc3 = nn.Linear(n_layer2, n_classes)

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        else:
            print('Unknown activation!')
            exit()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x
