# --------------------------------------------------------------------------------------------------------
# 2019/06/01
# fundamental_stock_analysis - make_final_dataset.py
# md
# --------------------------------------------------------------------------------------------------------


import sys
from time import strftime, gmtime
from typing import Any, Union

import talib
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.io.parsers import TextFileReader

# from src.my_tools.my_toolbox import MyOsTools as mos
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder

# Set pandas display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 19)
pd.set_option('display.width', 1500)

base_data_path = '/mnt/Development/My_Projects/fundamental_stock_analysis/data/'
raw_stockpup_path = 'raw/stockpup/'
raw_yahoo_quotes_path = 'raw/yahoo_quotes/'
raw_yahoo_info_path = 'raw/yahoo_info/'
raw_edgar_path = 'raw/edgar/'
raw_economic_indicators = 'raw/economic_indicators/'

interim_data_path = 'interim/'
processed_data_path = 'processed/'


def make_labels(min_success_quarters=4, min_return=.2):
    """
    :arg min_success_quarters: min # of quarters in the future to calculate pct_change-
    :arg min_return: min yearly return
    :return:
    """
    tickers = pd.read_csv(base_data_path + 'interim/tickers.csv', index_col=0)

    for ticker in tickers['ticker'].values:
        print(ticker)
        try:
            ticker_data = pd.read_csv(base_data_path + interim_data_path + 'interim_data/' + ticker + '.csv', index_col=0)
            if ticker_data.empty: continue
            # Todo: Check if returns are correct !!!!
            for r in range(min_success_quarters, 0, -1):
                dji = ticker_data['PR_^DJI_close'].pct_change(periods=r).shift(-r)
                sp500 = ticker_data['PR_^GSPC_close'].pct_change(periods=r).shift(-r)
                nasdaq = ticker_data['PR_^IXIC_close'].pct_change(periods=r).shift(-r)
                indx_mean = (dji + sp500 + nasdaq) / 3
                returns = ticker_data['PR_close'].pct_change(periods=r).shift(-r)
                ticker_data['X_return_' + str(r)] = returns
                excess_gain = pd.DataFrame((returns - indx_mean) > ((1 + min_return) ** (r / 4)) - 1, columns=['X_return_' + str(r)])  # min_return is yearly
                excess_loss = pd.DataFrame(((returns - indx_mean) < -(((1 + min_return) ** (r / 4)) - 1)), columns=['X_return_' + str(r)])  # min_return is yearly
                # Bool to int
                excess_gain = excess_gain.astype(int)
                excess_loss = excess_loss.astype(int)

                ticker_data['X_gain_lbl_' + str(r)] = excess_gain
                ticker_data['X_loss_lbl_' + str(r)] = excess_loss

            # Make label
            def f(x):
                """
                Count the # consecutive 1's starting going from ricght to left.
                """
                i = 1
                int_x = int(''.join(map(str, x)), 2) >> 0  # binary to int
                while int_x % 2 != 0:
                    int_x = int(''.join(map(str, x)), 2) >> i  # binary to int
                    i += 1
                return i - 1

            gain_lbl_cols = [c for c in ticker_data.columns if 'X_gain' in c]
            loss_lbl_cols = [c for c in ticker_data.columns if 'X_loss' in c]

            gain = ticker_data[gain_lbl_cols].apply(lambda x: f(x), axis=1)
            loss = ticker_data[loss_lbl_cols].apply(lambda x: f(x), axis=1)
            ticker_data['X_label_gain'] = gain
            ticker_data['X_label_loss'] = loss
            ticker_data.to_csv(base_data_path + processed_data_path + 'ticker_data/' + ticker + '.csv')

        except (KeyError, FileNotFoundError) as e:
            print('============> Error:', ticker)
            print(e)


# def _adjust_inflation(df: DataFrame) -> DataFrame:
#     # Todo: Implement adjust_inflation
#     ticker = 'AAPL'
#     ticker_interim_data = pd.read_csv(base_data_path + interim_data_path + 'interim_data/' + ticker + '.csv', index_col=0)
#     inflation = pd.read_csv(base_data_path + interim_data_path + 'economic_indicators/monthly_inflation_rates.csv', index_col=0)
#     print(type(inflation['Date'][0]))
#     ticker_interim_data['tmp_left_on'] = ticker_interim_data['Filing date'].str.rsplit('-', n=1, expand=True)[0]  # yyyy-mm
#     inflation['tmp_right_on'] = inflation['Date'].str.rsplit('-', n=1, expand=True)[0]  # yyyy-mm
#     # merge dataframe
#     ticker_interim_data = pd.merge(ticker_interim_data, inflation, how='left', left_on='tmp_left_on', right_on='tmp_right_on')
#     # Clean up
#     ticker_interim_data.rename(index=str, columns={'Value': 'Inflation'}, inplace=True)
#     ticker_interim_data.drop(['tmp_left_on', 'tmp_right_on', 'Date'], axis=1, inplace=True)
#     print(ticker_interim_data)


def scale_ticker_data():
    """
    This function scales the balance, income and cash flow columns with adjusted shares. All these statement data is now on a 'per share' base
    :return:
    """
    tickers = pd.read_csv(base_data_path + 'interim/tickers.csv', index_col=0)

    for ticker in tickers['ticker'].values:
        print(ticker)
        try:
            ticker_data = pd.read_csv(base_data_path + processed_data_path + 'ticker_data/' + ticker + '.csv', index_col=0)

            ticker_data['INF_scale'] = np.around(np.log10(ticker_data['INC_revenue'].values))  # Todo: better in interim_factory
            # if 0 revenure => inf scale
            ticker_data.replace(np.inf, np.nan, inplace=True)
            ticker_data.replace(-np.inf, np.nan, inplace=True)

            scale_cols = [c for c in ticker_data.columns if c[:3] in ['BAL', 'INC', 'CF_']]  # statement columns
            ticker_data[scale_cols] = ticker_data[scale_cols].div(ticker_data['INF_shares_split_adjusted'], axis=0)
            ticker_data.to_csv(base_data_path + processed_data_path + 'scaled_ticker_data/' + ticker + '.csv')
        except FileNotFoundError:
            print('=====>', ticker)


def combine_scaled_data():  # Todo: combine with scale_ticker_data
    final_dataset = pd.DataFrame()
    tickers = pd.read_csv(base_data_path + 'interim/tickers.csv', index_col=0)
    for ticker in tickers['ticker'].values:
        print(ticker)
        try:
            df = pd.read_csv(base_data_path + processed_data_path + 'scaled_ticker_data/' + ticker + '.csv', index_col=0)
            final_dataset = pd.concat([final_dataset, df])
        except FileNotFoundError:
            print('=====>', ticker)
    # Reset index
    final_dataset.reset_index(drop=True)
    final_dataset.to_csv(base_data_path + processed_data_path + 'final_dataset.csv')


def split_normalize_final_dataset(n_valid_test_tickers=100, n_valid_test_years=2):
    all_df = pd.read_csv(base_data_path + processed_data_path + 'final_dataset.csv', index_col=0)
    all_df.reset_index(inplace=True)
    all_df.drop(['index'], axis=1, inplace=True)

    # Handle nan
    nan_columns = all_df.columns[all_df.isna().sum() > 0]
    nan_mask = all_df.loc[:, nan_columns].isna()
    # replace nan with 0, the mean of the normalised columns
    all_df = all_df.fillna(0)
    # add mask as feature
    nan_mask.columns = ['MASK_{}'.format(c) for c in nan_mask.columns]
    all_df = pd.concat([all_df, nan_mask.astype(int)], axis=1)

    # Add final label
    gain = all_df['X_label_gain'] > 1
    loss = all_df['X_label_loss'] > 1
    all_df['label'] = gain * 1 - loss + 1 # +1 because Labels must be [0,1,2] and not [-1,0,1] for Pytorch CrossEntropyLoss

    # Remove future features
    all_df.drop([c for c in all_df.columns if c[:2] in ['X_']], axis=1, inplace=True)
    all_df.drop([c for c in all_df.columns if c[:7] in ['MASK_X_']], axis=1, inplace=True)

    # Remove corrolating =1 features
    all_df.drop([c for c in all_df.columns if
                 c in ['MASK_BAL_liabilities', 'MASK_BAL_shareholders_equity', 'MASK_INC_earnings_earnings_available',
                       'MASK_INF_scale', 'MASK_RAT_book_value_of_equity_per_share', 'MASK_RAT_long-term_debt_to_equity_ratio'
                                                                                    'MASK_RAT_current_ratio']], axis=1, inplace=True)

    # Takeout companies
    tickers = pd.DataFrame({'ticker': all_df['ticker'].unique()})

    valid_test_tickers = tickers.sample(n_valid_test_tickers)
    valid_tickers = valid_test_tickers.head(int(n_valid_test_tickers / 2))
    test_tickers = valid_test_tickers.tail(int(n_valid_test_tickers / 2))
    train_tickers = tickers.drop(valid_test_tickers.index)

    train_df = all_df[all_df['ticker'].isin(train_tickers['ticker'])]
    valid_df = all_df[all_df['ticker'].isin(valid_tickers['ticker'])]
    test_df = all_df[all_df['ticker'].isin(test_tickers['ticker'])]

    # Remove years
    valid_year = str(int(train_df['filing_date'].max()[:4]) - n_valid_test_years / 1)
    test_year = str(int(train_df['filing_date'].max()[:4]) - n_valid_test_years / 2)

    valid_mask = (train_df['filing_date'] > valid_year) & (train_df['filing_date'] < test_year)
    test_mask = (train_df['filing_date'] > test_year)
    train_mask = (train_df['filing_date'] < valid_year)

    valid_df = pd.concat([valid_df, train_df[valid_mask]])
    test_df = pd.concat([test_df, train_df[test_mask]])
    train_df = train_df[train_mask]
    print(all_df.shape, test_df.shape[0] + valid_df.shape[0] + train_df.shape[0])

    # Normalize
    non_normalise_cols = ['ticker', 'filing_date', 'quarter_start', 'quarter_end', 'sector', 'industry', 'm_start', 'm_end', 'q_start', 'label']
    non_normalise_cols = non_normalise_cols + [c for c in all_df.columns if c[:2] in ['X_', 'MA']]  # MASK_
    normalize_cols = [c for c in train_df.columns.values if c not in non_normalise_cols]

    mean = train_df[normalize_cols].mean()
    std = train_df[normalize_cols].std()

    train_df[normalize_cols] = (train_df[normalize_cols] - mean) / std
    valid_df[normalize_cols] = (valid_df[normalize_cols] - mean) / std
    test_df[normalize_cols] = (test_df[normalize_cols] - mean) / std

    # Remove metadata
    train_df = train_df.drop(['ticker', 'quarter_start', 'quarter_end', 'filing_date'], axis=1)
    valid_df = valid_df.drop(['ticker', 'quarter_start', 'quarter_end', 'filing_date'], axis=1)
    test_df = test_df.drop(['ticker', 'quarter_start', 'quarter_end', 'filing_date'], axis=1)

    print(train_df.head())
    print(train_df.tail())
    train_df.to_csv(base_data_path + processed_data_path + 'train_dataset.csv')
    valid_df.to_csv(base_data_path + processed_data_path + 'valid_dataset.csv')
    test_df.to_csv(base_data_path + processed_data_path + 'test_dataset.csv')

    # train_df['label'].hist()
    # plt.show()
    # valid_df['label'].hist()
    # plt.show()
    # test_df['label'].hist()
    # plt.show()


if __name__ == '__main__':
    # make_labels()
    # scale_ticker_data()
    # combine_scaled_data()
    split_normalize_final_dataset()
    pass
