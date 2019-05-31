# --------------------------------------------------------------------------------------------------------
# 2019/04/18
# 0_ml_project_template - synthetic_data.py
# md
# --------------------------------------------------------------------------------------------------------
import torch as th
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons

"""
Collection of methods returning a pandas DataFrame containing synthetic data
"""


class SyntheticDataset(Dataset):
    """
    Custom Dataset for Synthetic data
    """
    def __init__(self, Xy: pd.DataFrame):
        self.X = Xy.drop(columns=['label']).values
        self.y = Xy['label'].values
        self.id = id

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]
        x = th.Tensor(x)
        return x, y  # X: Tensor, y_true: float


def make_double_spiral(n_samples: int, shuffle: bool = False, noise: float = 0.) -> pd.DataFrame:
    """
    Makes a DataFrame containing double spiral

    :param n_samples: Nr of samples to generate
    :param shuffle: If True, shuffle the data
    :param noise: Noise factor
    :return: DataFrame with 2 features [a,b] and 1 binary label [label] containing double spiral
    """
    # Todo: This works but we should make it better readable
    # Todo: Noise doesn't seem to be well implemented. noise of 80% still converges
    n = np.sqrt(np.random.rand(n_samples, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_samples, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_samples, 1) * noise
    X = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))
    y = np.hstack((np.zeros(n_samples), np.ones(n_samples)))

    Xy = pd.DataFrame(X, columns=['a', 'b'])
    Xy['label'] = y

    if shuffle: Xy = Xy.sample(frac=1).reset_index(drop=True)  # drop=True deletes old index entries.
    return Xy



# See: https://machinelearningmastery.com/generate-test-datasets-python-scikit-learn/

def make_double_moons(n_samples: int, shuffle: bool = False, noise: float = None) -> pd.DataFrame:
    """
    See: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html

    :param n_samples: number of samples to generate
    :param noise: noise factor
    :return: DataFrame with 2 features [a,b] and 1 binary label [label containing moons
    """
    pass

def make_circles():
    pass