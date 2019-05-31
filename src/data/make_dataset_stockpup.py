# -*- coding: utf-8 -*-

#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """

from time import strftime, gmtime

import numpy as np
import pandas as pd

from src.my_tools.my_toolbox import MyOsTools as mos

base_data_path = '/mnt/Development/My_Projects/fundamental_stock_analysis/data/'
raw_data_path = 'raw/stockpup/'
interim_data_path = 'interim/stockpup/'


def combine_stockpup_data():
    """
    Combines ticker csv files and save
    """
    date_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())  # Session datetime
    tickers = pd.read_csv(base_data_path + 'raw/' + 'tickers.csv', index_col=0)
    all_raw_data = pd.DataFrame()
    for ticker in tickers['Ticker'].values:
        try:
            print('Combine ticker:', ticker)
            raw_data = pd.read_csv(base_data_path + raw_data_path + 'csv/' + ticker + '_quarterly_financial_data.csv')
            # Add Ticker column
            raw_data['Ticker'] = ticker
            # Replace "None" with NaN
            raw_data.replace('None', np.nan, inplace=True)
            # read_csv makes dtype=object for columns with str "None". Convert them to float
            num_columns = raw_data.columns.drop(labels=['Quarter end', 'Ticker'])
            raw_data[num_columns] = raw_data[num_columns].astype(float)
            # Concatenate raw_data with all_raw_data
            all_raw_data = pd.concat([all_raw_data, raw_data])
        except FileNotFoundError:
            print('File doesn\'t exist:', ticker)
            continue
    print(all_raw_data)
    all_raw_data.to_csv(base_data_path + raw_data_path + 'stockpup_raw_data.csv')


def cleanup_stockpup_data():
    """
    Reads the all_raw_data.csv from raw/stockpup/,
    set the incorrect data to NaN and saves the manually_cleaned_raw_data.csv into interim/stockpup/
    """
    all_raw_data = pd.read_csv(base_data_path + raw_data_path + 'stockpup_raw_data.csv', na_filter=True, index_col=0)

    # NUE has extreme high Cash items
    wrong_data = [{'Ticker': 'NUE',
                   'Quarter end': ['1994-04-02',
                                   '1994-07-02',
                                   '1994-10-01',
                                   '1994-12-31',
                                   '1995-04-01',
                                   '1995-07-01',
                                   '1995-09-30',
                                   '1995-12-31',
                                   '1996-03-30',
                                   '1996-06-29',
                                   '1996-09-28',
                                   '1996-12-31',
                                   '1997-04-05',
                                   '1997-07-05',
                                   '1997-10-04',
                                   '1998-04-04',
                                   '1998-07-04',
                                   '1998-10-03',
                                   '1998-12-31',
                                   '1999-07-03',
                                   '1999-10-02',
                                   '2000-04-01',
                                   '2000-07-01'],
                   'Column': ['Cash from operating activities',
                              'Cash from investing activities',
                              'Cash from financing activities',
                              'Cash change during period',
                              'Capital expenditures']
                   }]
    for wd in wrong_data:
        t = wd['Ticker']
        for qe in wd['Quarter end']:
            for col in wd['Column']:
                all_raw_data.loc[(all_raw_data['Ticker'] == t) & (all_raw_data['Quarter end'] == qe), col] = None

    # Remove extreme high P/E ratio's
    all_raw_data.loc[all_raw_data['P/E ratio'] > 7000, 'P/E ratio'] = np.nan
    # Remove extreme high P/B ratio's
    all_raw_data.loc[all_raw_data['P/B ratio'] > 7000, 'P/B ratio'] = np.nan
    # Remove extreme high Current ratio's
    all_raw_data.loc[all_raw_data['Current ratio'] > 7000, 'Current ratio'] = np.nan
    # Save all_raw_data
    all_raw_data.to_csv(base_data_path + raw_data_path + 'stockpup_raw_data.csv', index=True)


if __name__ == '__main__':
    combine_stockpup_data()
    cleanup_stockpup_data()
    pass
