# --------------------------------------------------------------------------------------------------------
# 2019/05/15
# fundamental_stock_analysis - yahoo_quotes_make_dataset.py
# md
# --------------------------------------------------------------------------------------------------------
from datetime import datetime
from time import sleep, gmtime, strftime

import pandas as pd
import numpy as np
from yahoo_historical import Fetcher
from src.my_tools.my_toolbox import MyOsTools as mos

base_data_path = '/mnt/Development/My_Projects/fundamental_stock_analysis/data/'
raw_data_path = 'raw/yahoo_quotes/'


def download_yahoo_quotes(interval='1d', redownload=False):
    """
    :param interval: "1d", "1wk", "1mo"
    :param redownload: if True then redownlaod all ticker quotes, else download only the tickers that are not in directory
    """
    date_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())  # Session datetime
    tickers = pd.read_csv(base_data_path + 'raw/' + 'tickers.csv', index_col=0)
    # # Load existing yahoo_quotes to use when redownload=False
    yahoo_quotes_fnames = mos.get_filenames(base_data_path + raw_data_path, ext='csv')

    # for ticker in np.append(tickers['Ticker'].values, other_tickers):
    for ticker in tickers['Ticker'].values:
        if not redownload and interval + '_' + ticker + '.csv' in yahoo_quotes_fnames:
            print('===> Ticker already downloaded:', ticker)
            continue
        try:
            print('Download ticker:', ticker)
            quotes = pd.DataFrame(Fetcher(ticker, [1800, 1, 1], [2020, 1, 1], interval=interval).getHistorical())
            # Check if company is delisted or aquired. Then quotes can't be downloaded from Yahoo
            if not len(quotes):
                raise KeyError
            quotes.to_csv(base_data_path + raw_data_path + interval + '_' + ticker + '.csv')

        except KeyError:
            print('===> Blacklisted:', ticker)
            # Append if exist otherwise create new tickers_blacklist.csv
            mode = 'a'
            header = False
            if 'tickers_blacklist.csv' not in mos.get_filenames(path=base_data_path + 'raw/', ext='csv'):
                print('===> Create new tickers_blacklist.csv')
                mode = 'w'
                header = True
            tickers_blacklist = pd.DataFrame([[date_time, ticker, 'yahoo_quotes']], columns=['Date', 'Ticker', 'Source'])
            tickers_blacklist.to_csv(base_data_path + 'tickers_blacklist.csv', mode=mode, header=header, index=False)
            # Mark ticker as blacklisted in tickers.csv
            tickers.loc[tickers['Ticker'] == ticker, 'Blacklist YQ'] = True
        # Update tickers.csv
        tickers.to_csv(base_data_path + 'raw/' + 'tickers.csv', mode='w', index=True)


def update_yahooo_quotes():
    pass


if __name__ == '__main__':
    for interval in ['1d', '1wk', '1mo']:
        # for interval in ['1mo']:
        download_yahoo_quotes(interval=interval, redownload=False)
    # make_quarterly_quotes()
