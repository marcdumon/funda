# --------------------------------------------------------------------------------------------------------
# 2019/05/22
# fundamental_stock_analysis - make_interim_data.py
# md
# --------------------------------------------------------------------------------------------------------
# Todo: Move codeblocks to ../features

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
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 1500)

# data paths
data_path = '/mnt/Development/My_Projects/fundamental_stock_analysis/data/'
raw_data_path = data_path + 'raw/'
stockpup_path = raw_data_path + 'stockpup/'
yahoo_quotes_path = raw_data_path + 'yahoo_quotes/csv/'
economic_indicators_path = raw_data_path + 'economic_indicators/'

yahoo_info_path = raw_data_path + 'yahoo_info/'
edgar_data_path = raw_data_path + 'edgar/'
interim_data_path = data_path + 'interim/'

# Printing options
date_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())  # Session datetime
print_dt = '{}: '.format(date_time)
print_arrow = '{}: ====================>'.format(date_time)
print_ln = '\n' + '-' * 120 + '\n'
print_block = '\n' + '#' * 120 + '\n'


def make_tickers():
    tickers = pd.read_csv(raw_data_path + 'tickers.csv', index_col=0)
    indexes = pd.DataFrame(['^GSPC', '^DJI', '^IXIC', '^N225', '^VIX', '^TNX', '^TYX', '^FVX', '^IRX'], columns=['ticker'])
    # Remove blacklisted tickers
    tickers = tickers.loc[tickers['bl'] != True, 'ticker']
    tickers = tickers.reset_index()
    tickers=tickers.drop('index', axis=1)
    tickers.to_csv(interim_data_path + 'tickers.csv')
    indexes.to_csv(interim_data_path + 'indexes.csv')
make_tickers()

def make_inflation():
    inflation = pd.read_csv(economic_indicators_path + 'monthly_inflation_rates.csv')
    inflation['CPI_inverse'] = (1 + inflation['Value'].iloc[::-1].values / 100) ** (1 / 12)  # yearly to monthly
    inflation['CPI_multiplier'] = inflation['CPI_inverse'].cumprod().iloc[::-1].values
    inflation['20190404_multiplier'] = inflation['CPI_multiplier']  # .iloc[::-1].values
    inflation['date'] = pd.to_datetime(inflation['TIME'], format='%Y-%m')
    inflation = inflation[['TIME', 'Value', '20190404_multiplier']]
    df = pd.DataFrame()
    df[['date', 'value', '20190404_multiplier']] = inflation[['TIME', 'Value', '20190404_multiplier']]
    print(df)
    # Todo: Check if formula is correct, what happens with new data?
    df.to_csv(interim_data_path + 'economic_indicators/monthly_inflation_rates.csv')


def make_stockpup():
    """

    - Make dates
    - Add features
        - Month, Quarter
        - INC_earnings	- INC_earnings_available_for_common_stockholders


    :return:
    """
    stockpup = pd.read_csv(stockpup_path + 'stockpup.csv', index_col=0)
    print(stockpup.shape)
    # Make dates
    stockpup['quarter_end'] = pd.to_datetime(stockpup['quarter_end'], format='%Y-%m-%d')
    stockpup['quarter_start'] = stockpup['quarter_end'] - pd.dateOffset(months=3) + pd.offsets.MonthBegin(1)
    # Reorder columns
    new_order = [0, -1] + [i + 1 for i in range(len(stockpup.columns) - 2)]  # put quarter_end on 2 place
    stockpup = stockpup[stockpup.columns[new_order]]

    # Add/Remove features
    stockpup['m_start'] = stockpup['quarter_start'].dt.month
    stockpup['m_end'] = stockpup['quarter_end'].dt.month
    stockpup['q_start'] = stockpup['quarter_start'].dt.quarter  # # Don't use quarter_end. Ex: PAYX quarter_end 1996-02-29 is 1Q
    stockpup['INC_earnings_earnings_available'] = stockpup['INC_earnings'] - stockpup['INC_earnings_available_for_common_stockholders']

    stockpup = stockpup.drop(['INC_earnings_available_for_common_stockholders', 'RAT_cumulative_dividends_per_share'], axis=1)


    stockpup.to_csv(interim_data_path + 'stockpup.csv')


def make_yahoo_quotes():
    tickers = pd.read_csv(interim_data_path+'tickers.csv', index_col=0)
    indexes = pd.read_csv(interim_data_path + 'indexes.csv', index_col=0)
    tickers = np.concatenate([tickers['ticker'].values, indexes['ticker'].values])
    for ticker in tickers:
        # if ticker!='AMZN': continue
        print(ticker)
        ticker_quotes = pd.read_csv(yahoo_quotes_path + ticker + '.csv', index_col=0)
        # Remove nan rows
        ticker_quotes.dropna(axis=0, inplace=True)
        ticker_quotes['ticker'] = ticker
        ticker_quotes['date'] = pd.to_datetime(ticker_quotes['date'], format='%Y-%m-%d')
        # Add technical indicators
        # Todo: add money flow
        ticker_quotes['ma10'] = talib.MA(ticker_quotes['close'], timeperiod=10)
        ticker_quotes['ma22'] = talib.MA(ticker_quotes['close'], timeperiod=22)
        ticker_quotes['ma252'] = talib.MA(ticker_quotes['close'], timeperiod=252)
        ticker_quotes['midp10'] = talib.MIDPRICE(ticker_quotes['high'], ticker_quotes['low'], timeperiod=10)
        ticker_quotes['midp22'] = talib.MIDPRICE(ticker_quotes['high'], ticker_quotes['low'], timeperiod=22)
        ticker_quotes['midp252'] = talib.MIDPRICE(ticker_quotes['high'], ticker_quotes['low'], timeperiod=252)
        ticker_quotes['rsi10'] = talib.RSI(ticker_quotes['close'], timeperiod=10)
        ticker_quotes['rsi22'] = talib.RSI(ticker_quotes['close'], timeperiod=22)
        ticker_quotes['rsi252'] = talib.RSI(ticker_quotes['close'], timeperiod=252)
        ticker_quotes['obv'] = talib.OBV(ticker_quotes['close'], ticker_quotes['volume'])
        ticker_quotes['mfi10'] = talib.MFI(ticker_quotes['high'], ticker_quotes['low'], ticker_quotes['close'], ticker_quotes['volume'], timeperiod=10)
        ticker_quotes['mfi22'] = talib.MFI(ticker_quotes['high'], ticker_quotes['low'], ticker_quotes['close'], ticker_quotes['volume'], timeperiod=22)
        ticker_quotes['mfi252'] = talib.MFI(ticker_quotes['high'], ticker_quotes['low'], ticker_quotes['close'], ticker_quotes['volume'], timeperiod=252)
        ticker_quotes['vol10'] = ticker_quotes['close'].pct_change().rolling(10).std(ddof=0) * (252 ** 0.5)
        ticker_quotes['vol22'] = ticker_quotes['close'].pct_change().rolling(22).std(ddof=0) * (252 ** 0.5)
        ticker_quotes['vol252'] = ticker_quotes['close'].pct_change().rolling(252).std(ddof=0) * (252 ** 0.5)

        ticker_quotes = ticker_quotes[['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume',
                                       'ma10', 'ma22', 'ma252', 'midp10', 'midp22', 'midp252', 'mfi10', 'mfi22', 'mfi252',
                                       'rsi10', 'rsi22', 'rsi252', 'obv', 'vol10', 'vol22', 'vol252']]
        ticker_quotes.to_csv(interim_data_path + 'yahoo_quotes/' + ticker + '.csv')


def make_interim_yahoo_info():
    yahoo_info = pd.read_csv(base_data_path + raw_yahoo_info_path + 'yahoo_info_raw_data.csv', index_col=0)
    le = LabelEncoder()
    yahoo_info['Sector label'] = le.fit_transform(yahoo_info['Sector'])
    yahoo_info['Industry label'] = le.fit_transform(yahoo_info['Industry'])
    yahoo_info.to_csv(base_data_path + interim_data_path + 'yahoo_info_interim_data.csv')


def make_interim_edgar():
    tickers = pd.read_csv(base_data_path + 'interim/tickers.csv', index_col=0)
    for ticker in tickers['ticker'].values:
        print(ticker)
        try:
            ticker_edgar = pd.read_csv(base_data_path + raw_edgar_path + ticker + '.csv', index_col=0)
        except FileNotFoundError:
            print('File doesn\'t exists:', ticker)  # Todo: possible filing on a day when markets are closed
            continue
        ticker_edgar['Filing date'] = pd.to_datetime(ticker_edgar['Filing date'].astype(str), format='%Y%m%d')
        ticker_edgar['Quarter end'] = pd.to_datetime(ticker_edgar['Quarter end'].astype(str), format='%Y%m%d')
        ticker_edgar = ticker_edgar[['ticker', 'Quarter end', 'Filing date', 'Type']]
        ticker_edgar.to_csv(base_data_path + interim_data_path + 'edgar/' + ticker + '.csv')


def make_interim_data(start_from='A', end_till='ZZZZZ', write_log=False):
    if write_log: sys.stdout = open(base_data_path + interim_data_path + 'make_interin_data_log.csv', 'w')
    tickers = pd.read_csv(base_data_path + 'interim/tickers.csv', index_col=0)
    indexes = pd.read_csv(base_data_path + 'interim/indexes.csv', index_col=0)
    stockpup = pd.read_csv(base_data_path + interim_data_path + 'stockpup_interim_data.csv', index_col=0)

    yahoo_info = pd.read_csv(base_data_path + interim_data_path + 'yahoo_info_interim_data.csv', index_col=0)
    interim_data_row = {}

    for ticker in tickers['ticker'].values:
        if start_from > ticker: continue
        if ticker > end_till: continue
        blacklist = []
        ticker_interim_data = pd.DataFrame()
        print(ticker)
        try:
            ticker_edgar = pd.read_csv(base_data_path + interim_data_path + 'edgar/' + ticker + '.csv', index_col=0)
        except FileNotFoundError:
            print('File doesn\'t exists:', ',' + ticker)
            continue
        # Yayoo_quotes
        ticker_quotes = pd.read_csv(base_data_path + interim_data_path + 'yahoo_quotes/' + ticker + '.csv', index_col=0)
        ticker_quotes.dropna(axis=0, inplace=True)
        ticker_stockpup = stockpup[stockpup['ticker'] == ticker]
        ticker_stockpup = ticker_stockpup.sort_values(by='Quarter end')

        for _, row in ticker_stockpup.iterrows():

            # Look-up filing date for quarter
            mask = (ticker_edgar['Quarter end'] == row['Quarter end']) & ((ticker_edgar['Type'] == '10-K') | (ticker_edgar['Type'] == '10-Q'))
            try:
                filing_date = ticker_edgar.loc[mask, 'Filing date'].values[0]
                print(ticker, filing_date)
            except IndexError:
                print('Filing date does\'t exist:' + ',' + row['Quarter end'])
                continue
            # Get ticker_quotes for quarter start till filing date
            mask = (ticker_quotes['date'] >= row['Quarter start']) & (ticker_quotes['date'] <= filing_date)
            quarter_quotes = ticker_quotes[mask]
            if quarter_quotes.empty:
                print('No quotes for period: ', ',' + row['Quarter start'], filing_date)
                continue
            # Make interim data row
            interim_data_row['Filing date'] = filing_date
            interim_data_row['open'] = quarter_quotes.iloc[0]['open']
            interim_data_row['close'] = quarter_quotes.iloc[-1]['close']
            interim_data_row['O_ma252'] = quarter_quotes.iloc[0]['ma252']
            interim_data_row['C_ma252'] = quarter_quotes.iloc[-1]['ma252']
            interim_data_row['Cl_Min'] = quarter_quotes['low'].min()
            interim_data_row['Cl_Max'] = quarter_quotes['high'].max()
            interim_data_row['Cl_Mean'] = quarter_quotes['close'].mean()
            interim_data_row['Cl_Std'] = quarter_quotes['close'].std()

            # Add indexes
            for index in indexes['ticker'].values:
                index_quotes = pd.read_csv(base_data_path + interim_data_path + 'yahoo_quotes/' + index + '.csv', index_col=0)
                index_quotes.dropna(axis=0, inplace=True)
                # Get index_quotes for quarter start till filing date
                mask = (index_quotes['date'] >= row['Quarter start']) & (index_quotes['date'] <= filing_date)
                quarter_index_quotes = index_quotes[mask]
                if quarter_index_quotes.empty:
                    print('No index quotes for period: ', ',' + row['Quarter start'], filing_date)
                    continue
                # Make interim data row
                interim_data_row[index + ' ' + 'open'] = quarter_index_quotes.iloc[0]['open']
                interim_data_row[index + ' ' + 'close'] = quarter_index_quotes.iloc[-1]['close']
                interim_data_row[index + ' ' + 'O_ma252'] = quarter_index_quotes.iloc[0]['ma252']
                interim_data_row[index + ' ' + 'C_ma252'] = quarter_index_quotes.iloc[-1]['ma252']
                interim_data_row[index + ' ' + 'Cl_Min'] = quarter_index_quotes['low'].min()
                interim_data_row[index + ' ' + 'Cl_Max'] = quarter_index_quotes['high'].max()
                interim_data_row[index + ' ' + 'Cl_Mean'] = quarter_index_quotes['close'].mean()
                interim_data_row[index + ' ' + 'Cl_Std'] = quarter_index_quotes['close'].std()

            # Yahoo_info
            interim_data_row['Sector'] = yahoo_info.loc[yahoo_info['ticker'] == ticker, 'Sector label'].values[0]
            interim_data_row['Industry'] = yahoo_info.loc[yahoo_info['ticker'] == ticker, 'Industry label'].values[0]
            ticker_interim_data = ticker_interim_data.append({**row.to_dict(), **interim_data_row}, ignore_index=True)

        # CLEAN-UP
        # Remove and reorder columns
        col_info = ['ticker', 'Filing date', 'Quarter start', 'Quarter end', 'Sector', 'Industry', 'Shares', 'Shares split adjusted', 'Split factor']
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
        col_quotes = ['open', 'close', 'Cl_Max', 'Cl_Mean', 'Cl_Min', 'Cl_Std', 'O_ma252', 'C_ma252',
                      '^DJI C_ma252', '^DJI Cl_Max', '^DJI Cl_Mean', '^DJI Cl_Min', '^DJI Cl_Std', '^DJI close', '^DJI O_ma252', '^DJI open',
                      '^FVX C_ma252', '^FVX Cl_Max', '^FVX Cl_Mean', '^FVX Cl_Min', '^FVX Cl_Std', '^FVX close', '^FVX O_ma252', '^FVX open',
                      '^GSPC C_ma252', '^GSPC Cl_Max', '^GSPC Cl_Mean', '^GSPC Cl_Min', '^GSPC Cl_Std', '^GSPC close', '^GSPC O_ma252', '^GSPC open',
                      '^IRX C_ma252', '^IRX Cl_Max', '^IRX Cl_Mean', '^IRX Cl_Min', '^IRX Cl_Std', '^IRX close', '^IRX O_ma252', '^IRX open',
                      '^IXIC C_ma252', '^IXIC Cl_Max', '^IXIC Cl_Mean', '^IXIC Cl_Min', '^IXIC Cl_Std', '^IXIC close', '^IXIC O_ma252', '^IXIC open',
                      '^N225 C_ma252', '^N225 Cl_Max', '^N225 Cl_Mean', '^N225 Cl_Min', '^N225 Cl_Std', '^N225 close', '^N225 O_ma252', '^N225 open',
                      '^TNX C_ma252', '^TNX Cl_Max', '^TNX Cl_Mean', '^TNX Cl_Min', '^TNX Cl_Std', '^TNX close', '^TNX O_ma252', '^TNX open',
                      '^TYX C_ma252', '^TYX Cl_Max', '^TYX Cl_Mean', '^TYX Cl_Min', '^TYX Cl_Std', '^TYX close', '^TYX O_ma252', '^TYX open',
                      '^VIX C_ma252', '^VIX Cl_Max', '^VIX Cl_Mean', '^VIX Cl_Min', '^VIX Cl_Std', '^VIX close', '^VIX O_ma252', '^VIX open']
        # Removed columns
        # 'Cumulative dividends per share', #  Dividents from 1st reporting quarter until now. Irrelevant
        # 'Price', 'Price high', 'Price low', # Will get prices from yahoo_quotes
        try:
            # columns = col_info  # + col_balance+ col_income + col_cf + col_ratio + col_quotes
            # ticker_interim_data = ticker_interim_data[columns]
            # print(ticker_interim_data.columns.values)
            # columns = col_info + col_balance  # + col_income + col_cf + col_ratio + col_quotes
            # ticker_interim_data = ticker_interim_data[columns]
            # columns = col_info + col_balance + col_income  # + col_cf + col_ratio + col_quotes
            # ticker_interim_data = ticker_interim_data[columns]
            # columns = col_info + col_balance + col_income + col_cf  # + col_ratio + col_quotes
            # ticker_interim_data = ticker_interim_data[columns]
            # columns = col_info + col_balance + col_income + col_cf + col_ratio  # + col_quotes
            # ticker_interim_data = ticker_interim_data[columns]

            columns = col_info + col_balance + col_income + col_cf + col_ratio + col_quotes
            ticker_interim_data = ticker_interim_data[columns]
        except (KeyError, TypeError):
            print(ticker_interim_data.columns.values)
            print('=' * 120)
            print(columns)
            # raise

        ticker_interim_data.to_csv(base_data_path + interim_data_path + 'interim_data/' + ticker + '.csv')
        print(ticker_interim_data.head(15))
        print('-' * 200)


if __name__ == '__main__':
    # make_tickers()
    # make_inflation()
    # make_stockpup()
    make_yahoo_quotes()
    # make_interim_yahoo_info()
    # make_interim_edgar()
    # make_interim_data(start_from='A', end_till='E', write_log=False)
    # make_interim_data(start_from='E',end_till='I', write_log=False)
    # make_interim_data(start_from='I',end_till='M', write_log=False)
    # make_interim_data(start_from='M', end_till='Q', write_log=False)
    # make_interim_data(start_from='Q', end_till='U', write_log=False)
    # make_interim_data(start_from='U', end_till='Z', write_log=False)
    pass
