# --------------------------------------------------------------------------------------------------------
# 2019/04/18
# 0_ml_project_template - data_helpers.py
# md
# --------------------------------------------------------------------------------------------------------
import pandas as pd


def split_train_valid(Xy: pd.DataFrame, train_pct: float = 0.1):
    """
    Splits a dataframe into training and validation dataframes
    :param Xy: Dataframe to split
    :param train_pct: percent of Xy to train dataframe. 1-val_pct to valid dataframe
    :return: train dataframe, valid dataframe
    """
    # Todo: replace with split in sklearn ?
    n_train_samples = int(Xy.shape[0] * train_pct)
    train_df = Xy.sample(n_train_samples)
    valid_df = Xy[~Xy.index.isin(train_df.index)]

    return train_df, valid_df


def scale_normalize_data(Xy: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
    """
    Scales the data from a Dataframe. The label column is expected to be named 'label' and will not be normalized

    :param Xy: Dataframe to be normalized
    :param method: Choice between 'standard' (mean=0, std=1) normalization or 'min_max' scaling
    :return: Dataframe with scaled data
    """
    if method == 'standard':
        Xy_features = Xy.loc[:, Xy.columns != 'label']
        Xy_features = (Xy_features - Xy_features.mean()) / Xy_features.std()
        Xy.loc[:, Xy.columns != 'label'] = Xy_features
    elif method == 'min_max':
        Xy_features = Xy.loc[:, Xy.columns != 'label']
        Xy_features = (Xy_features - Xy_features.min()) / (Xy_features.max() - Xy_features.min())
        Xy.loc[:, Xy.columns != 'label'] = Xy_features
    else:
        print('Unknown method for scale_normalize_data()')
        exit()
    return Xy
