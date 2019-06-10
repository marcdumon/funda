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
from my_tools.my_toolbox import MyOsTools as mos
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
    tickers = tickers.drop('index', axis=1)
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
    tickers = pd.read_csv(interim_data_path + 'tickers.csv', index_col=0)
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


def make_yahoo_info():
    yahoo_info = pd.read_csv(yahoo_info_path + 'yahoo_info.csv', index_col=0)
    le = LabelEncoder()
    yahoo_info['sector'] = le.fit_transform(yahoo_info['sector'])
    yahoo_info['industry'] = le.fit_transform(yahoo_info['industry'])
    yahoo_info.to_csv(interim_data_path + 'yahoo_info.csv')


def make_edgar_filing_lists():
    """
    This function makes a list of all Edgar filings for each ticker
    :return:
    """
    tickers = pd.read_csv(interim_data_path + 'tickers.csv', index_col=0)
    all_types = pd.DataFrame()  # columns=['type', 'result'])
    types_in_ticker = pd.DataFrame()  # columns=['type', 'result'])
    type_list = []
    for ticker in tickers['ticker'].values:
        print(ticker)
        try:
            ticker_edgar = pd.read_csv(edgar_data_path + ticker + '.csv', index_col=0)
        except FileNotFoundError:
            print('File doesn\'t exists:', ticker)
            raise

        # Remove empty rows
        # Todo: there shouldn't be any empty rows. Check raw_factory.make_edgar_filing_lists
        ticker_edgar['type'].replace('', np.nan)
        ticker_edgar.dropna(how='all', inplace=True)

        # Rename column
        ticker_edgar.rename(columns={'period_of_report': 'quarter_end'}, inplace=True)

        # Statistics
        all_types = pd.concat([all_types, ticker_edgar['type']], ignore_index=True)
        types_in_ticker = pd.concat([types_in_ticker, pd.Series(ticker_edgar['type'].unique())], ignore_index=True)
        ticker_edgar = ticker_edgar[['ticker', 'quarter_end', 'filing_date', 'type', 'items']]
        ticker_edgar.to_csv(interim_data_path + 'edgar/' + ticker + '.csv')

    all_types['result'] = 1  # result can't be nan otherwise groupby.count doesn't work
    all_types = all_types.groupby(all_types.columns[0]).count().sort_values('result', ascending=False)
    types_in_ticker['result'] = 1
    types_in_ticker = types_in_ticker.groupby(types_in_ticker.columns[0]).count().sort_values('result', ascending=False)
    all_types.to_csv(interim_data_path + 'stat_edgar_count_types.csv')
    types_in_ticker.to_csv(interim_data_path + 'stat_edgar_count_types_in_ticker.csv')


def make_interim_data(start_from='A', end_till='ZZZZZ', redownload=False, write_log=False):
    # Todo: lots of filing_date does't exit errors (UNM 2010-12-31, huge amounts on 31/12 !!!!!)
    if write_log: sys.stdout = open(interim_data_path + 'make_interin_data_log.csv', 'w')
    tickers = pd.read_csv(interim_data_path + 'tickers.csv', index_col=0)
    indexes = pd.read_csv(interim_data_path + 'indexes.csv', index_col=0)
    stockpup = pd.read_csv(interim_data_path + 'stockpup.csv', index_col=0)
    yahoo_info = pd.read_csv(interim_data_path + 'yahoo_info.csv', index_col=0)
    interim_data_row = {}
    tickers_downloaded = mos.get_filenames(interim_data_path+'interim_data/', ext='csv')  # Todo: put this in ticker loop to avoid start end ticker ????
    tickers_downloaded = [t.split('.csv')[0] for t in tickers_downloaded]
    for ticker in tickers['ticker'].values:
        print(ticker)
        if (ticker < start_ticker) | (ticker > end_ticker): continue
        if (not redownload) & (ticker in tickers_downloaded): continue
        blacklist = []
        ticker_interim_data = pd.DataFrame()
        try:
            ticker_edgar = pd.read_csv(interim_data_path + 'edgar/' + ticker + '.csv', index_col=0)
        except FileNotFoundError:
            print('File doesn\'t exists:', ',' + ticker)
            continue
        # Yayoo_quotes
        ticker_quotes = pd.read_csv(interim_data_path + 'yahoo_quotes/' + ticker + '.csv', index_col=0)
        ticker_quotes.dropna(axis=0, inplace=True)
        ticker_stockpup = stockpup[stockpup['ticker'] == ticker]
        ticker_stockpup = ticker_stockpup.sort_values(by='quarter_end')

        for _, row in ticker_stockpup.iterrows():
            try:
                # Look-up filing_date for quarter
                mask = (ticker_edgar['quarter_end'] == row['quarter_end']) & ((ticker_edgar['type'] == '10-K') | (ticker_edgar['type'] == '10-Q'))
                filing_date = ticker_edgar.loc[mask, 'filing_date'].values[0]
                print(ticker, filing_date)
            except IndexError as e:
                print('Error: filing_date does\'t exist:' + ',' + row['quarter_end'])  # Todo: Ex: ADT, stockpup date before edgar
                continue
            # Get ticker_quotes for quarter start till filing_date
            mask = (ticker_quotes['date'] >= row['quarter_start']) & (ticker_quotes['date'] <= filing_date)
            quarter_quotes = ticker_quotes[mask]
            if quarter_quotes.empty:
                print('Error: No quotes for period: ', ',' + row['quarter_start'], filing_date)
                continue
            # Make interim data row
            interim_data_row['filing_date'] = filing_date
            interim_data_row['PR_open'] = quarter_quotes.iloc[0]['open']
            interim_data_row['PR_close'] = quarter_quotes.iloc[-1]['close']
            interim_data_row['PR_o_ma252'] = quarter_quotes.iloc[0]['ma252']
            interim_data_row['PR_c_ma252'] = quarter_quotes.iloc[-1]['ma252']
            interim_data_row['PR_min'] = quarter_quotes['low'].min()
            interim_data_row['PR_max'] = quarter_quotes['high'].max()
            interim_data_row['PR_mean'] = quarter_quotes['close'].mean()
            interim_data_row['PR_std'] = quarter_quotes['close'].std()

            # Add indexes
            for index in indexes['ticker'].values:
                index_quotes = pd.read_csv(interim_data_path + 'yahoo_quotes/' + index + '.csv', index_col=0)
                index_quotes.dropna(axis=0, inplace=True)
                # Get index_quotes for quarter start till filing_date
                mask = (index_quotes['date'] >= row['quarter_start']) & (index_quotes['date'] <= filing_date)
                quarter_index_quotes = index_quotes[mask]
                if quarter_index_quotes.empty:
                    print('Error: No index quotes for period: ', ',' + row['quarter_start'], filing_date)
                    continue
                # Make interim data row
                interim_data_row['PR_'+index + '_open'] = quarter_index_quotes.iloc[0]['open']
                interim_data_row['PR_'+index + '_close'] = quarter_index_quotes.iloc[-1]['close']
                interim_data_row['PR_'+index + '_o_ma252'] = quarter_index_quotes.iloc[0]['ma252']
                interim_data_row['PR_'+index + '_c_ma252'] = quarter_index_quotes.iloc[-1]['ma252']
                interim_data_row['PR_'+index + '_min'] = quarter_index_quotes['low'].min()
                interim_data_row['PR_'+index + '_max'] = quarter_index_quotes['high'].max()
                interim_data_row['PR_'+index + '_mean'] = quarter_index_quotes['close'].mean()
                interim_data_row['PR_'+index + '_std'] = quarter_index_quotes['close'].std()

            # Yahoo_info
            interim_data_row['sector'] = yahoo_info.loc[yahoo_info['ticker'] == ticker, 'sector'].values[0]
            interim_data_row['industry'] = yahoo_info.loc[yahoo_info['ticker'] == ticker, 'industry'].values[0]
            ticker_interim_data = ticker_interim_data.append({**row.to_dict(), **interim_data_row}, ignore_index=True)

        # CLEAN-UP
        start_cols = ['ticker', 'quarter_start', 'quarter_end', 'filing_date', 'm_start', 'm_end', 'q_start', 'sector', 'industry']
        columns = start_cols + [c for c in ticker_interim_data.columns if c not in start_cols]
        if ticker_interim_data.empty: continue  # Ex: ACC has problerms in yahoo_quote and ticker_interim_data is empty
        ticker_interim_data = ticker_interim_data[columns]

        ticker_interim_data.to_csv(interim_data_path + 'interim_data/' + ticker + '.csv')
        print(ticker_interim_data.head(15))
        print('-' * 200)


if __name__ == '__main__':
    start_ticker = input('Start ticker = ')
    end_ticker = input('End ticker = ')

    # make_tickers()
    # make_inflation()
    # make_stockpup()
    # make_yahoo_quotes()
    # make_yahoo_info()
    # make_edgar_filing_lists()
    make_interim_data(start_from=start_ticker, end_till=end_ticker, write_log=False)

    pass
