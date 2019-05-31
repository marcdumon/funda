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

base_data_path = '/mnt/Development/My_Projects/fundamental_stock_analysis/data/'
raw_stockpup_path = 'raw/stockpup/'
raw_yahoo_quotes_path = 'raw/yahoo_quotes/'
raw_yahoo_info_path = 'raw/yahoo_info/'
raw_edgar_path = 'raw/edgar/'
raw_economic_indicators = 'raw/economic_indicators/'

interim_data_path = 'interim/'


def make_tickers():
    tickers = pd.read_csv(base_data_path + 'raw/tickers.csv', index_col=0)
    indexes = pd.DataFrame(['^GSPC', '^DJI', '^IXIC', '^N225', '^VIX', '^TNX', '^TYX', '^FVX', '^IRX'], columns=['Ticker'])
    # Remove blacklisted tickers
    tickers = tickers[tickers['Blacklist YQ'] != True]
    tickers = tickers[tickers['Blacklist YI'] != True]
    tickers = tickers.drop(['Blacklist YQ', 'Blacklist YI'], axis=1)
    tickers.to_csv(base_data_path + interim_data_path + 'tickers.csv')
    indexes.to_csv(base_data_path + interim_data_path + 'indexes.csv')


def make_interim_inflation():
    inflation = pd.read_csv(base_data_path + raw_economic_indicators + 'monthly_inflation_rates.csv')
    inflation['CPI_inverse'] = (1 + inflation['Value'].iloc[::-1].values / 100) ** (1 / 12)
    inflation['CPI_multiplier'] = inflation['CPI_inverse'].cumprod().iloc[::-1].values
    inflation['20190404_multiplier'] = inflation['CPI_multiplier']  # .iloc[::-1].values
    inflation['Date'] = pd.to_datetime(inflation['TIME'], format='%Y-%m')
    inflation = inflation[['Date', 'Value', '20190404_multiplier']]
    inflation.to_csv(base_data_path + interim_data_path + 'economic_indicators/monthly_inflation_rates.csv')


def make_interim_stockpup():
    stockpup = pd.read_csv(base_data_path + raw_stockpup_path + 'stockpup_raw_data.csv', index_col=0)
    # Make dates
    stockpup['Quarter end'] = pd.to_datetime(stockpup['Quarter end'], format='%Y-%m-%d')
    stockpup['Quarter start'] = stockpup['Quarter end'] - pd.DateOffset(months=3) + pd.offsets.MonthBegin(1)

    # CLEAN-UP
    # Remove and reorder columns
    col_info = ['Ticker', 'Quarter start', 'Quarter end', 'Shares', 'Shares split adjusted', 'Split factor']
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
    # Removed columns
    # 'Cumulative dividends per share', #  Dividents from 1st reporting quarter until now. Irrelevant
    # 'Price', 'Price high', 'Price low', # Will get prices from yahoo_quotes

    cols = col_info + col_income + col_cf + col_ratio
    stockpup = stockpup[cols]
    print(stockpup.sample(10))
    stockpup.to_csv(base_data_path + interim_data_path + 'stockpup_interim_data.csv')


def make_interim_yahoo_quotes():
    tickers = pd.read_csv(base_data_path + 'interim/tickers.csv', index_col=0)
    indexes = pd.read_csv(base_data_path + 'interim/indexes.csv', index_col=0)
    tickers = np.concatenate([tickers['Ticker'].values, indexes['Ticker'].values])
    for ticker in tickers:
        # if ticker!='AMZN': continue
        print(ticker)
        ticker_quotes = pd.read_csv(base_data_path + raw_yahoo_quotes_path + '1d_' + ticker + '.csv', index_col=0)
        # Remove nan rows
        ticker_quotes.dropna(axis=0, inplace=True)
        ticker_quotes['Ticker'] = ticker
        ticker_quotes['Date'] = pd.to_datetime(ticker_quotes['Date'], format='%Y-%m-%d')
        # Add technical indicators
        # Todo: add money flow
        ticker_quotes['ma10'] = talib.MA(ticker_quotes['Close'], timeperiod=10)
        ticker_quotes['ma22'] = talib.MA(ticker_quotes['Close'], timeperiod=22)
        ticker_quotes['ma252'] = talib.MA(ticker_quotes['Close'], timeperiod=252)
        ticker_quotes['midp10'] = talib.MIDPRICE(ticker_quotes['High'], ticker_quotes['Low'], timeperiod=10)
        ticker_quotes['midp22'] = talib.MIDPRICE(ticker_quotes['High'], ticker_quotes['Low'], timeperiod=22)
        ticker_quotes['midp252'] = talib.MIDPRICE(ticker_quotes['High'], ticker_quotes['Low'], timeperiod=252)
        ticker_quotes['rsi10'] = talib.RSI(ticker_quotes['Close'], timeperiod=10)
        ticker_quotes['rsi22'] = talib.RSI(ticker_quotes['Close'], timeperiod=22)
        ticker_quotes['rsi252'] = talib.RSI(ticker_quotes['Close'], timeperiod=252)
        ticker_quotes['obv'] = talib.OBV(ticker_quotes['Close'], ticker_quotes['Volume'])
        ticker_quotes['mfi10'] = talib.MFI(ticker_quotes['High'], ticker_quotes['Low'], ticker_quotes['Close'], ticker_quotes['Volume'], timeperiod=10)
        ticker_quotes['mfi22'] = talib.MFI(ticker_quotes['High'], ticker_quotes['Low'], ticker_quotes['Close'], ticker_quotes['Volume'], timeperiod=22)
        ticker_quotes['mfi252'] = talib.MFI(ticker_quotes['High'], ticker_quotes['Low'], ticker_quotes['Close'], ticker_quotes['Volume'], timeperiod=252)
        ticker_quotes['vol10'] = ticker_quotes['Close'].pct_change().rolling(10).std(ddof=0) * (252 ** 0.5)
        ticker_quotes['vol22'] = ticker_quotes['Close'].pct_change().rolling(22).std(ddof=0) * (252 ** 0.5)
        ticker_quotes['vol252'] = ticker_quotes['Close'].pct_change().rolling(252).std(ddof=0) * (252 ** 0.5)

        # Scale ohlc otherwise we can't compare diffent stocks
        # price_range = np.concatenate([ticker_quotes[['High']].values, ticker_quotes[['Low']].values])
        # transformer = RobustScaler(quantile_range=(1., 99.)).fit(price_range)  # Robust scaler avoids problems with outliers
        # ticker_quotes['Open'] = transformer.transform(ticker_quotes[['Open']])
        # ticker_quotes['High'] = transformer.transform(ticker_quotes[['High']])
        # ticker_quotes['Low'] = transformer.transform(ticker_quotes[['Low']])
        # ticker_quotes['Close'] = transformer.transform(ticker_quotes[['Close']])
        # ticker_quotes['Adj Close'] = transformer.transform(ticker_quotes[['Adj Close']])

        # Order columns
        ticker_quotes = ticker_quotes[['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                                       'ma10', 'ma22', 'ma252', 'midp10', 'midp22', 'midp252', 'mfi10', 'mfi22', 'mfi252',
                                       'rsi10', 'rsi22', 'rsi252', 'obv', 'vol10', 'vol22', 'vol252']]
        ticker_quotes.to_csv(base_data_path + interim_data_path + 'yahoo_quotes/' + ticker + '.csv')


def make_interim_yahoo_info():
    yahoo_info = pd.read_csv(base_data_path + raw_yahoo_info_path + 'yahoo_info_raw_data.csv', index_col=0)
    le = LabelEncoder()
    yahoo_info['Sector label'] = le.fit_transform(yahoo_info['Sector'])
    yahoo_info['Industry label'] = le.fit_transform(yahoo_info['Industry'])
    yahoo_info.to_csv(base_data_path + interim_data_path + 'yahoo_info_interim_data.csv')


def make_interim_edgar():
    tickers = pd.read_csv(base_data_path + 'interim/tickers.csv', index_col=0)
    for ticker in tickers['Ticker'].values:
        print(ticker)
        try:
            ticker_edgar = pd.read_csv(base_data_path + raw_edgar_path + ticker + '.csv', index_col=0)
        except FileNotFoundError:
            print('File doesn\'t exists:', ticker)
            continue
        ticker_edgar['Filing date'] = pd.to_datetime(ticker_edgar['Filing date'].astype(str), format='%Y%m%d')
        ticker_edgar['Quarter end'] = pd.to_datetime(ticker_edgar['Quarter end'].astype(str), format='%Y%m%d')
        ticker_edgar = ticker_edgar[['Ticker', 'Quarter end', 'Filing date', 'Type']]
        ticker_edgar.to_csv(base_data_path + interim_data_path + 'edgar/' + ticker + '.csv')


def make_interim_data(start_from='A', end_till='ZZZZZ', write_log=False):
    if write_log: sys.stdout = open(base_data_path + interim_data_path + 'make_interin_data_log.csv', 'w')
    tickers = pd.read_csv(base_data_path + 'interim/tickers.csv', index_col=0)
    indexes = pd.read_csv(base_data_path + 'interim/indexes.csv', index_col=0)
    stockpup = pd.read_csv(base_data_path + interim_data_path + 'stockpup_interim_data.csv', index_col=0)
    yahoo_info = pd.read_csv(base_data_path + interim_data_path + 'yahoo_info_interim_data.csv', index_col=0)
    interim_data_row = {}

    for ticker in tickers['Ticker'].values:
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
        ticker_stockpup = stockpup[stockpup['Ticker'] == ticker]
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
            mask = (ticker_quotes['Date'] >= row['Quarter start']) & (ticker_quotes['Date'] <= filing_date)
            quarter_quotes = ticker_quotes[mask]
            if quarter_quotes.empty:
                print('No quotes for period: ', ',' + row['Quarter start'], filing_date)
                continue
            # Make interim data row
            interim_data_row['Filing date'] = filing_date
            interim_data_row['Open'] = quarter_quotes.iloc[0]['Open']
            interim_data_row['Close'] = quarter_quotes.iloc[-1]['Close']
            interim_data_row['O_ma252'] = quarter_quotes.iloc[0]['ma252']
            interim_data_row['C_ma252'] = quarter_quotes.iloc[-1]['ma252']
            interim_data_row['Cl_Min'] = quarter_quotes['Low'].min()
            interim_data_row['Cl_Max'] = quarter_quotes['High'].max()
            interim_data_row['Cl_Mean'] = quarter_quotes['Close'].mean()
            interim_data_row['Cl_Std'] = quarter_quotes['Close'].std()

            # Add indexes
            for index in indexes['Ticker'].values:
                index_quotes = pd.read_csv(base_data_path + interim_data_path + 'yahoo_quotes/' + index + '.csv', index_col=0)
                index_quotes.dropna(axis=0, inplace=True)
                # Get index_quotes for quarter start till filing date
                mask = (index_quotes['Date'] >= row['Quarter start']) & (index_quotes['Date'] <= filing_date)
                quarter_index_quotes = index_quotes[mask]
                if quarter_index_quotes.empty:
                    print('No index quotes for period: ', ',' + row['Quarter start'], filing_date)
                    continue
                # Make interim data row
                interim_data_row[index + ' ' + 'Open'] = quarter_index_quotes.iloc[0]['Open']
                interim_data_row[index + ' ' + 'Close'] = quarter_index_quotes.iloc[-1]['Close']
                interim_data_row[index + ' ' + 'O_ma252'] = quarter_index_quotes.iloc[0]['ma252']
                interim_data_row[index + ' ' + 'C_ma252'] = quarter_index_quotes.iloc[-1]['ma252']
                interim_data_row[index + ' ' + 'Cl_Min'] = quarter_index_quotes['Low'].min()
                interim_data_row[index + ' ' + 'Cl_Max'] = quarter_index_quotes['High'].max()
                interim_data_row[index + ' ' + 'Cl_Mean'] = quarter_index_quotes['Close'].mean()
                interim_data_row[index + ' ' + 'Cl_Std'] = quarter_index_quotes['Close'].std()

            # Yahoo_info
            interim_data_row['Sector'] = yahoo_info.loc[yahoo_info['Ticker'] == ticker, 'Sector label'].values[0]
            ticker_interim_data = ticker_interim_data.append({**row.to_dict(), **interim_data_row}, ignore_index=True)
        columns = ['Ticker', 'Filing date', 'Quarter start', 'Quarter end',
                   'Open', 'Close', 'Cl_Max', 'Cl_Mean', 'Cl_Min', 'Cl_Std', 'O_ma252', 'C_ma252',
                   'Asset turnover',
                   'Assets',
                   'Book value of equity per share',
                   'Capital expenditures',
                   'Cash at end of period',
                   'Cash change during period',
                   'Cash from financing activities',
                   'Cash from investing activities',
                   'Cash from operating activities',
                   'Current Assets',
                   'Current Liabilities',
                   'Current ratio',
                   'Dividend payout ratio',
                   'Dividend per share',
                   'EPS basic',
                   'EPS diluted',
                   'Earnings',
                   'Earnings available for common stockholders',
                   'Equity to assets ratio',
                   'Free cash flow per share',
                   'Goodwill & intangibles',
                   'Liabilities',
                   'Long-term debt',
                   'Long-term debt to equity ratio',
                   'Net margin',
                   'Non-controlling interest',
                   'P/B ratio',
                   'P/E ratio',
                   'Preferred equity',
                   'ROA',
                   'ROE',
                   'Revenue',
                   'Sector',
                   'Shareholders equity',
                   'Shares',
                   'Shares split adjusted',
                   'Split factor',
                   '^DJI C_ma252', '^DJI Cl_Max', '^DJI Cl_Mean', '^DJI Cl_Min', '^DJI Cl_Std', '^DJI Close', '^DJI O_ma252', '^DJI Open',
                   '^FVX C_ma252', '^FVX Cl_Max', '^FVX Cl_Mean', '^FVX Cl_Min', '^FVX Cl_Std', '^FVX Close', '^FVX O_ma252', '^FVX Open',
                   '^GSPC C_ma252', '^GSPC Cl_Max', '^GSPC Cl_Mean', '^GSPC Cl_Min', '^GSPC Cl_Std', '^GSPC Close', '^GSPC O_ma252', '^GSPC Open',
                   '^IRX C_ma252', '^IRX Cl_Max', '^IRX Cl_Mean', '^IRX Cl_Min', '^IRX Cl_Std', '^IRX Close', '^IRX O_ma252', '^IRX Open',
                   '^IXIC C_ma252', '^IXIC Cl_Max', '^IXIC Cl_Mean', '^IXIC Cl_Min', '^IXIC Cl_Std', '^IXIC Close', '^IXIC O_ma252', '^IXIC Open',
                   '^N225 C_ma252', '^N225 Cl_Max', '^N225 Cl_Mean', '^N225 Cl_Min', '^N225 Cl_Std', '^N225 Close', '^N225 O_ma252', '^N225 Open',
                   '^TNX C_ma252', '^TNX Cl_Max', '^TNX Cl_Mean', '^TNX Cl_Min', '^TNX Cl_Std', '^TNX Close', '^TNX O_ma252', '^TNX Open',
                   '^TYX C_ma252', '^TYX Cl_Max', '^TYX Cl_Mean', '^TYX Cl_Min', '^TYX Cl_Std', '^TYX Close', '^TYX O_ma252', '^TYX Open',
                   '^VIX C_ma252', '^VIX Cl_Max', '^VIX Cl_Mean', '^VIX Cl_Min', '^VIX Cl_Std', '^VIX Close', '^VIX O_ma252', '^VIX Open'
                   ]
        try:
            ticker_interim_data = ticker_interim_data[columns]
        except:
            blacklist.append(ticker)
        ticker_interim_data.to_csv(base_data_path + interim_data_path + 'interim_data/' + ticker + '.csv')
        print(ticker_interim_data.head(15))
        print('-' * 200)

    print('#' * 120)
    print(blacklist)


def adjust_inflation(df: DataFrame) -> DataFrame:
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


def make_labels(fwd_quarters=5):
    tickers = pd.read_csv(base_data_path + 'interim/tickers.csv', index_col=0)
    df = pd.DataFrame()
    for ticker in tickers['Ticker'].values:
        # if ticker>'B': continue
        print(ticker)
        # print(ticker_interim_data.loc[ticker_interim_data['Close'].pct_change(periods=-fwd_quarters)>0.1,'Close'])
        # print(ticker_interim_data.loc[ticker_interim_data['^DJI Close'].pct_change(periods=-fwd_quarters)>0.1,['Filing date', '^DJI Close']])
        try:
            ticker_interim_data = pd.read_csv(base_data_path + interim_data_path + 'interim_data/' + ticker + '.csv', index_col=0)
            x = ticker_interim_data['Close'].pct_change(periods=fwd_quarters).shift(-fwd_quarters)
            dji = ticker_interim_data['^DJI Close'].pct_change(periods=fwd_quarters).shift(-fwd_quarters)
            sp500 = ticker_interim_data['^GSPC Close'].pct_change(periods=fwd_quarters).shift(-fwd_quarters)
            nasdaq = ticker_interim_data['^IXIC Close'].pct_change(periods=fwd_quarters).shift(-fwd_quarters)
            ref = (dji + sp500 + nasdaq) / 3
            df = pd.concat([df, x-ref], axis=0)
            df.dropna(inplace=True)
        except (KeyError,FileNotFoundError):
            print('------------------->',ticker)
    df.to_csv('xxx.csv')
    df.plot.hist(bins=100)
    plt.show()


def combine_interim_data():
    pass


if __name__ == '__main__':
    # make_tickers()
    # make_interim_inflation()
    # make_interim_yahoo_quotes()
    # make_interim_yahoo_info()
    # make_interim_stockpup()
    # make_interim_edgar()
    # make_interim_data(start_from='A', end_till='E', write_log=False)
    # make_interim_data(start_from='E',end_till='I', write_log=False)
    # make_interim_data(start_from='I',end_till='M', write_log=False)
    # make_interim_data(start_from='M',end_till='Q', write_log=False) *
    # make_interim_data(start_from='Q',end_till='U', write_log=False)
    # make_interim_data(start_from='U',end_till='Z', write_log=False)
    make_labels()
    #
    # adjust_inflation()
