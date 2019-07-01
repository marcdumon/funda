# --------------------------------------------------------------------------------------------------------
# 2019/04/18
# 0_ml_project_template - model.py
# md
# --------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FeedForwardNN(nn.Module):
    # Inspired by https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/

    def __init__(self, emb_dims, no_of_cont, lin_layer_sizes, output_size, emb_dropout, lin_layer_dropouts):

        """
        Parameters
        ----------

        emb_dims: List of two element tuples
          This list will contain a two element tuple for each
          categorical feature. The first element of a tuple will
          denote the number of unique values of the categorical
          feature. The second element will denote the embedding
          dimension to be used for that feature.

        no_of_cont: Integer
          The number of continuous features in the data.

        lin_layer_sizes: List of integers.
          The size of each linear layer. The length will be equal
          to the total number
          of linear layers in the network.

        output_size: Integer
          The size of the final output.

        emb_dropout: Float
          The dropout to be used after the embedding layers.

        lin_layer_dropouts: List of floats
          The dropouts to be used after each linear layer.
        """

        super().__init__()

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont, lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList([first_lin_layer] +
                                        [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1]) for i in range(len(lin_layer_sizes) - 1)])

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])

    def forward(self, cont_data, cat_data):
        if self.no_of_embs != 0:
            x = [emb_layer(cat_data[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)

        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(cont_data)
            if self.no_of_embs != 0:
                x = torch.cat([x, normalized_cont_data], 1)
            else:
                x = normalized_cont_data

        for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers, self.droput_layers, self.bn_layers):
            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        x = self.output_layer(x)

        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(116, 115)
        self.fc2 = nn.Linear(115, 114)
        # self.fc3 = nn.Linear(114, 113)
        # self.fc4 = nn.Linear(113, 112)
        # self.fc5 = nn.Linear(112, 111)
        # self.fc6 = nn.Linear(111, 110)
        # self.fc7 = nn.Linear(110, 111)
        # self.fc8 = nn.Linear(111, 112)
        # self.fc9 = nn.Linear(112, 113)
        # self.fc10 = nn.Linear(113, 114)
        self.fc11 = nn.Linear(114, 115)
        self.fc12 = nn.Linear(115, 116)
        # self.fc13 = nn.Linear(110, 115)
        # self.fc14 = nn.Linear(115, 116)
        self.relu = nn.ReLU()
        # self.tanh=nn.Tanh()
        # nn.init.kaiming_normal_(self.fc1.weight.data)
        # nn.init.kaiming_normal_(self.fc6.weight.data)

    def forward(self, cont_data):
        # x = self.relu(self.fc1(cont_data))
        x = self.fc1(cont_data)
        x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        # x = self.relu(self.fc4(x))
        # x = self.relu(self.fc5(x))
        # x = self.relu(self.fc6(x))
        # x = self.relu(self.fc7(x))
        # x = self.relu(self.fc8(x))
        # x = self.relu(self.fc9(x))
        # x = self.relu(self.fc10(x))
        x = self.relu(self.fc11(x))
        x = self.fc12(x)
        # x = self.relu(self.fc13(x))
        # x = self.fc14(x)
        return x
