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


def make_labels(min_success_quarters=8, min_return=.2):
    """
    :arg min_success_quarters: min # of quarters in the future to calculate pct_change-
    :arg min_return: min yearly return
    :return:
    """
    tickers = pd.read_csv(base_data_path + 'interim/tickers.csv', index_col=0)

    for ticker in tickers['Ticker'].values:
        # if ticker<'ACV': continue
        print(ticker)
        df = pd.DataFrame()
        try:
            ticker_interim_data = pd.read_csv(base_data_path + interim_data_path + 'interim_data/' + ticker + '.csv', index_col=0)
            if ticker_interim_data.empty: continue
            # Remove previously created labels columns (Return, count) if they exist
            return_cols = ticker_interim_data.columns.str.contains('Return|count')
            ticker_interim_data = ticker_interim_data.loc[:, ~return_cols]
            for r in range(min_success_quarters, 0, -1):
                dji = ticker_interim_data['^DJI Close'].pct_change(periods=r).shift(-r)
                sp500 = ticker_interim_data['^GSPC Close'].pct_change(periods=r).shift(-r)
                nasdaq = ticker_interim_data['^IXIC Close'].pct_change(periods=r).shift(-r)
                indx_mean = (dji + sp500 + nasdaq) / 3
                returns = ticker_interim_data['Close'].pct_change(periods=r).shift(-r)
                excess_return = pd.DataFrame((returns - indx_mean) > ((1 + min_return) ** (r / 4)) - 1, columns=['Return_' + str(r)])  # min_return is yearly
                df = pd.concat([df, excess_return], axis=1).astype(int)  # bool -> int

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

            result = df.apply(lambda x: f(x), axis=1)
            ticker_interim_data['Label'] = result
            print(ticker_interim_data)
            ticker_interim_data.to_csv(base_data_path + processed_data_path + 'ticker_data/' + ticker + '.csv')

        except (KeyError, FileNotFoundError):
            print('============> Error:', ticker)
            # raise


def _adjust_inflation(df: DataFrame) -> DataFrame:
    # Todo: Implement adjust_inflation
    ticker = 'AAPL'
    ticker_interim_data = pd.read_csv(base_data_path + interim_data_path + 'interim_data/' + ticker + '.csv', index_col=0)
    inflation = pd.read_csv(base_data_path + interim_data_path + 'economic_indicators/monthly_inflation_rates.csv', index_col=0)
    print(type(inflation['Date'][0]))
    ticker_interim_data['tmp_left_on'] = ticker_interim_data['Filing date'].str.rsplit('-', n=1, expand=True)[0]  # yyyy-mm
    inflation['tmp_right_on'] = inflation['Date'].str.rsplit('-', n=1, expand=True)[0]  # yyyy-mm
    # merge dataframe
    ticker_interim_data = pd.merge(ticker_interim_data, inflation, how='left', left_on='tmp_left_on', right_on='tmp_right_on')
    # Clean up
    ticker_interim_data.rename(index=str, columns={'Value': 'Inflation'}, inplace=True)
    ticker_interim_data.drop(['tmp_left_on', 'tmp_right_on', 'Date'], axis=1, inplace=True)
    print(ticker_interim_data)


def _scale_final_dataset(df: DataFrame) -> DataFrame:
    # df.reset_index(inplace=True)

    # Add scale = magnitude of order of revenue
    df.loc[df['Revenue'] > 0, 'Scale'] = np.around(np.log10(df.loc[df['Revenue'] > 0, 'Revenue'].values))

    col_info = ['Ticker', 'Filing date', 'Quarter start', 'Quarter end', 'Sector', 'Industry', 'Shares', 'Shares split adjusted', 'Split factor']
    # Balance sheet statement
    col_balance = ['Assets', 'Current Assets', 'Liabilities', 'Current Liabilities', 'Shareholders equity',
                   'Non-controlling interest', 'Preferred equity', 'Goodwill & intangibles', 'Long-term debt']
    # Income statement
    col_income = ['Revenue', 'Earnings', 'Earnings available for common stockholders',
                  'EPS basic', 'EPS diluted', 'Dividend per share']
    # Cashflow statement
    col_cf = ['Cash from operating activities', 'Cash from investing activities', 'Cash from financing activities',
              'Cash change during period', 'Cash at end of period', 'Capital expenditures']
    # Ratio's
    col_ratio = ['ROE', 'ROA', 'Book value of equity per share', 'P/B ratio', 'P/E ratio',
                 'Dividend payout ratio', 'Long-term debt to equity ratio',
                 'Equity to assets ratio', 'Net margin', 'Asset turnover', 'Free cash flow per share', 'Current ratio']
    col_quotes = ['Open', 'Close', 'Cl_Max', 'Cl_Mean', 'Cl_Min', 'Cl_Std', 'O_ma252', 'C_ma252',
                  '^DJI C_ma252', '^DJI Cl_Max', '^DJI Cl_Mean', '^DJI Cl_Min', '^DJI Cl_Std', '^DJI Close', '^DJI O_ma252', '^DJI Open',
                  '^FVX C_ma252', '^FVX Cl_Max', '^FVX Cl_Mean', '^FVX Cl_Min', '^FVX Cl_Std', '^FVX Close', '^FVX O_ma252', '^FVX Open',
                  '^GSPC C_ma252', '^GSPC Cl_Max', '^GSPC Cl_Mean', '^GSPC Cl_Min', '^GSPC Cl_Std', '^GSPC Close', '^GSPC O_ma252', '^GSPC Open',
                  '^IRX C_ma252', '^IRX Cl_Max', '^IRX Cl_Mean', '^IRX Cl_Min', '^IRX Cl_Std', '^IRX Close', '^IRX O_ma252', '^IRX Open',
                  '^IXIC C_ma252', '^IXIC Cl_Max', '^IXIC Cl_Mean', '^IXIC Cl_Min', '^IXIC Cl_Std', '^IXIC Close', '^IXIC O_ma252', '^IXIC Open',
                  '^N225 C_ma252', '^N225 Cl_Max', '^N225 Cl_Mean', '^N225 Cl_Min', '^N225 Cl_Std', '^N225 Close', '^N225 O_ma252', '^N225 Open',
                  '^TNX C_ma252', '^TNX Cl_Max', '^TNX Cl_Mean', '^TNX Cl_Min', '^TNX Cl_Std', '^TNX Close', '^TNX O_ma252', '^TNX Open',
                  '^TYX C_ma252', '^TYX Cl_Max', '^TYX Cl_Mean', '^TYX Cl_Min', '^TYX Cl_Std', '^TYX Close', '^TYX O_ma252', '^TYX Open',
                  '^VIX C_ma252', '^VIX Cl_Max', '^VIX Cl_Mean', '^VIX Cl_Min', '^VIX Cl_Std', '^VIX Close', '^VIX O_ma252', '^VIX Open']
    # Removed columns
    # 'Cumulative dividends per share', #  Dividents from 1st reporting quarter until now. Irrelevant
    # 'Price', 'Price high', 'Price low', # Will get prices from yahoo_quotes

    cols = col_info + col_balance + col_income + col_cf + col_ratio + col_quotes + ['Scale'] + ['Label']

    cols_remove = ['Ticker', 'Filing date', 'Quarter start', 'Quarter end', 'Sector', 'Industry',
                   'EPS basic', 'EPS diluted', 'Dividend per share',
                   'Cl_Std', '^DJI C_ma252', '^DJI Cl_Max', '^DJI Cl_Mean', '^DJI Cl_Min', '^DJI Cl_Std', '^DJI Close', '^DJI O_ma252', '^DJI Open',
                   '^FVX C_ma252', '^FVX Cl_Max', '^FVX Cl_Mean', '^FVX Cl_Min', '^FVX Cl_Std', '^FVX Close', '^FVX O_ma252', '^FVX Open',
                   '^GSPC C_ma252', '^GSPC Cl_Max', '^GSPC Cl_Mean', '^GSPC Cl_Min', '^GSPC Cl_Std', '^GSPC Close', '^GSPC O_ma252', '^GSPC Open',
                   '^IRX C_ma252', '^IRX Cl_Max', '^IRX Cl_Mean', '^IRX Cl_Min', '^IRX Cl_Std', '^IRX Close', '^IRX O_ma252', '^IRX Open',
                   '^IXIC C_ma252', '^IXIC Cl_Max', '^IXIC Cl_Mean', '^IXIC Cl_Min', '^IXIC Cl_Std', '^IXIC Close', '^IXIC O_ma252', '^IXIC Open',
                   '^N225 C_ma252', '^N225 Cl_Max', '^N225 Cl_Mean', '^N225 Cl_Min', '^N225 Cl_Std', '^N225 Close', '^N225 O_ma252', '^N225 Open',
                   '^TNX C_ma252', '^TNX Cl_Max', '^TNX Cl_Mean', '^TNX Cl_Min', '^TNX Cl_Std', '^TNX Close', '^TNX O_ma252', '^TNX Open',
                   '^TYX C_ma252', '^TYX Cl_Max', '^TYX Cl_Mean', '^TYX Cl_Min', '^TYX Cl_Std', '^TYX Close', '^TYX O_ma252', '^TYX Open',
                   '^VIX C_ma252', '^VIX Cl_Max', '^VIX Cl_Mean', '^VIX Cl_Min', '^VIX Cl_Std', '^VIX Close', '^VIX O_ma252', '^VIX Open', 'Scale', 'Label'
                   ] + col_ratio
    col_scale = [x for x in cols if x not in cols_remove]
    # scaler = RobustScaler(quantile_range=(.01, .99))
    scaler = MinMaxScaler(feature_range=(-1, 1))

    def f(x):
        if not x.sum(): return x  # column contains only null-values (fi: AAN Current Assets)
        null_index = x.isnull()
        if null_index.sum() > 0:
            x[~null_index] = scaler.fit_transform(x[~null_index].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            x = scaler.fit_transform(x.values.reshape(-1, 1)).reshape(1, -1)[0]
        return x

    df[col_scale] = df[col_scale].apply(f, axis=0)
    return df[cols]


def combine_interim_data():
    final_dataset = pd.DataFrame()
    tickers = pd.read_csv(base_data_path + 'interim/tickers.csv', index_col=0)
    for ticker in tickers['Ticker'].values:
        print(ticker)
        try:
            df = pd.read_csv(base_data_path + processed_data_path + 'scaled_ticker_data/' + ticker + '.csv', index_col=0)
            final_dataset = pd.concat([final_dataset, df])
        except FileNotFoundError:
            print('=====>', ticker)
    final_dataset.to_csv(base_data_path + processed_data_path + 'final_dataset.csv')


def split_dataset():
    tickers = pd.read_csv(base_data_path + 'interim/tickers.csv', index_col=0)
    # Takeout companies
    s = tickers.sample(20)
    print(s)
    print(tickers.drop(s.index))
    # Takeout year
    pass


def split_normalize_final_dataset(all_df, n_valid_test_tickers=100, n_valid_test_years=2):
    all_df.reset_index(inplace=True)
    all_df.drop(['index'], axis=1, inplace=True)

    tickers = pd.read_csv(base_data_path + 'interim/tickers.csv', index_col=0)

    # Takeout companies
    valid_test_tickers = tickers.sample(n_valid_test_tickers)
    valid_tickers = valid_test_tickers.head(int(n_valid_test_tickers / 2))
    test_tickers = valid_test_tickers.tail(int(n_valid_test_tickers / 2))
    train_tickers = tickers.drop(valid_test_tickers.index)

    train_df = all_df[all_df['Ticker'].isin(train_tickers['Ticker'])]
    valid_df = all_df[all_df['Ticker'].isin(valid_tickers['Ticker'])]
    test_df = all_df[all_df['Ticker'].isin(test_tickers['Ticker'])]

    # Remove years
    valid_year = str(int(train_df['Filing date'].max()[:4]) - n_valid_test_years / 1)
    test_year = str(int(train_df['Filing date'].max()[:4]) - n_valid_test_years / 2)

    valid_mask = (train_df['Filing date'] > valid_year) & (train_df['Filing date'] < test_year)
    test_mask = (train_df['Filing date'] > test_year)
    train_mask = (train_df['Filing date'] < valid_year)

    valid_df = pd.concat([valid_df, train_df[valid_mask]])
    test_df = pd.concat([test_df, train_df[test_mask]])
    train_df = train_df[train_mask]
    print(all_df.shape, test_df.shape[0] + valid_df.shape[0] + train_df.shape[0])

    # Normalize
    non_normalise_cols = ['Ticker', 'Filing date', 'Quarter start', 'Quarter end', 'Sector', 'Industry', 'Scale', 'Label']
    normalize_cols = [c for c in train_df.columns.values if c not in non_normalise_cols]
    mean = train_df[normalize_cols].mean()
    std = train_df[normalize_cols].std()

    train_df[normalize_cols] = (train_df[normalize_cols] - mean) / std
    valid_df[normalize_cols] = (valid_df[normalize_cols] - mean) / std
    test_df[normalize_cols] = (test_df[normalize_cols] - mean) / std
    print(train_df['Scale'].min())
    print(test_df['Scale'].min())
    print(valid_df['Scale'].min())

    print(train_df.index[np.isinf(train_df['Scale'])])

    train_df.to_csv(base_data_path + processed_data_path + 'train_dataset.csv')
    valid_df.to_csv(base_data_path + processed_data_path + 'valid_dataset.csv')
    test_df.to_csv(base_data_path + processed_data_path + 'test_dataset.csv')
    # Todo: why are there so many labels=8 ???
    train_df['Label'].hist()
    plt.show()
    valid_df['Label'].hist()
    plt.show()
    test_df['Label'].hist()
    plt.show()


def handle_nan(df):
    nan_columns = df.columns[df.isna().sum() > 0]
    nan_mask = df.loc[:, nan_columns].isna()
    # replace nan with 0, the mean of the normalised columns
    df = df.fillna(0)
    # add mask as feature
    nan_mask.columns = ['Mask {}'.format(c) for c in nan_mask.columns]
    df = pd.concat([df, nan_mask.astype(int)], axis=1)
    return df


if __name__ == '__main__':
    # make_labels()
    # tickers = pd.read_csv(base_data_path + 'interim/tickers.csv', index_col=0)
    # for ticker in tickers['Ticker'].values:
    #     # if ticker<'CELG': continue
    #     print(ticker)
    #     try:
    #         ticker_interim_data = pd.read_csv(base_data_path + interim_data_path + 'interim_data/' + ticker + '.csv', index_col=0)
    #         if ticker_interim_data.empty: continue
    #         df = pd.read_csv(base_data_path + processed_data_path + 'ticker_data/' + ticker + '.csv', index_col=0)
    #         # print(df)
    #         df = _scale_final_dataset(df)
    #         df.to_csv(base_data_path + processed_data_path + 'scaled_ticker_data/' + ticker + '.csv')
    #
    #     except (FileNotFoundError):
    #         print('=====>', ticker)
    # combine_interim_data()
    # final_dataset = pd.read_csv(base_data_path + processed_data_path + 'final_dataset.csv', index_col=0)
    #
    # split_normalize_final_dataset(final_dataset)
    for ds in ['train', 'valid', 'test']:
        print(ds)
        df = pd.read_csv(base_data_path + processed_data_path + '{}_dataset.csv'.format(ds), index_col=0)
        df=handle_nan(df)
        df.to_csv(base_data_path + processed_data_path + '{}_dataset.csv'.format(ds))
    pass
