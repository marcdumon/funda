# --------------------------------------------------------------------------------------------------------
# 2019/06/04
# fundamental_stock_analysis - make_datasets.py
# md
# --------------------------------------------------------------------------------------------------------
import os
import re
from time import strftime, gmtime, sleep
from socket import gaierror
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from yahoo_historical import Fetcher

from my_tools.my_toolbox import MyOsTools as mos

# Set pandas display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1500)
# data paths
data_path = '/mnt/Development/My_Projects/fundamental_stock_analysis/data/'
raw_data_path = data_path + 'raw/'
stockpup_path = raw_data_path + 'stockpup/'
yahoo_quotes_path = raw_data_path + 'yahoo_quotes/'
yahoo_info_path = raw_data_path + 'yahoo_info/'
edgar_data_path = raw_data_path + 'edgar/'
# Printing options
date_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())  # Session datetime
print_dt = '{}: '.format(date_time)
print_arrow = '{}: ====================>'.format(date_time)
print_ln = '\n' + '-' * 120 + '\n'
print_block = '\n' + '#' * 120 + '\n'

""" 
Startup Instructions
- Use th Chrome Extension 'Batch Link Downloader' to manually download .csv files from stockpup.com/data
  to .data/raw/stockpup/csv
- The tickers list is generated by scanning tickernames from stockpup .csv files
"""


#######################################################################################################################
# STOCKPUP
#######################################################################################################################

def make_tickers() -> None:
    """
    Reads all the stockpup csv files and extract ticker names. Save the ticker names to tickers.csv
    """
    fnames = mos.get_filenames(stockpup_path + 'csv', ext='csv')
    tickers = ['{}'.format(t.split('_')[0]) for t in fnames]
    tickers.sort()
    tickers_df = pd.DataFrame()
    tickers_df['ticker'] = tickers
    tickers_df.to_csv(raw_data_path + 'tickers.csv')
    print(print_block, print_dt, 'Nr of tickers created: {}'.format(tickers_df.shape[0]), print_block)


def make_stockpup_data(start_ticker: str = 'A', end_ticker='ZZZZ') -> pd.DataFrame:
    # Todo: make function to download data from stockpup.com site.
    #  See: https://github.com/meretciel/backtesting_platform/blob/90160fd053996ac2f455892f7b4447eec4743d45/script/download_stock_fundamental_data.py
    """
    This function uses manualy downloaded stockpup files, combine then into one big csv file and rename columns.
        - Combines all the stockpup csv files into a big dataframe.
        - Replaces spaces from column names with '_' and make all small caps
        - Replaces 'None' values with NaN
        - Blacklist problem tickers
        - Save dataframe as stockpup.csv

        :param start_ticker: first ticker to start process
        :param end_ticker: last ticker to end process

        :return: None
        """
    stockpup_df = pd.DataFrame()
    tickers = pd.read_csv(raw_data_path + 'tickers.csv', index_col=0)

    # Combine
    for _, ticker in enumerate(tickers['ticker']):
        if (ticker < start_ticker) | (ticker > end_ticker): continue
        print(print_dt, 'Process Stockpup ticker: {}'.format(ticker))
        try:
            sp_ticker_data = pd.read_csv(stockpup_path + 'csv/' + ticker + '_quarterly_financial_data.csv')
            sp_ticker_data['ticker'] = ticker
            stockpup_df = pd.concat([stockpup_df, sp_ticker_data], axis=0)
        except FileNotFoundError:
            tickers = _blacklist_ticker(ticker, 'bl_stockpup')

    # Column names
    stockpup_df.columns = [c.lower().replace(' ', '_') for c in stockpup_df.columns]
    id_cols = ['ticker', 'quarter_end']
    info_cols = ['shares', 'shares_split_adjusted', 'split_factor']
    balance_cols = ['assets', 'current_assets', 'liabilities', 'current_liabilities', 'shareholders_equity',
                    'non-controlling_interest', 'preferred_equity', 'goodwill_&_intangibles', 'long-term_debt']
    income_cols = ['revenue', 'earnings', 'earnings_available_for_common_stockholders', ]
    cashflow_cols = ['cash_from_operating_activities', 'cash_from_investing_activities',
                     'cash_from_financing_activities', 'cash_change_during_period', 'cash_at_end_of_period', 'capital_expenditures']
    ratio_cols = ['roe', 'roa', 'book_value_of_equity_per_share', 'p/b_ratio', 'p/e_ratio',
                  'cumulative_dividends_per_share', 'dividend_payout_ratio', 'long-term_debt_to_equity_ratio',
                  'equity_to_assets_ratio', 'net_margin', 'asset_turnover', 'free_cash_flow_per_share', 'current_ratio', 'eps_basic', 'eps_diluted',
                   'dividend_per_share']

    columns = id_cols + info_cols + income_cols + balance_cols + cashflow_cols + ratio_cols  # Todo: better in interim_factory Removed  'price', 'price_high', 'price_low'
    stockpup_df = stockpup_df[columns]

    # Todo: Save space by using CamelCase column names iso using '_'
    info_cols = ['INF_{}'.format(c) for c in info_cols if c not in ['ticker', 'quarter_end']]
    balance_cols = ['BAL_{}'.format(c) for c in balance_cols]
    income_cols = ['INC_{}'.format(c) for c in income_cols]
    cashflow_cols = ['CF_{}'.format(c) for c in cashflow_cols]
    ratio_cols = ['RAT_{}'.format(c) for c in ratio_cols]
    columns = id_cols + info_cols + income_cols + balance_cols + cashflow_cols + ratio_cols
    stockpup_df.columns = columns

    # None values
    stockpup_df.replace('None', np.nan, inplace=True)
    num_cols = info_cols + income_cols + balance_cols + ratio_cols
    stockpup_df[num_cols] = stockpup_df[num_cols].astype(float)  # Convert object columns to float

    # Save
    stockpup_df.to_csv(stockpup_path + 'stockpup.csv')
    print(print_block, print_dt, 'Stockpup saved ... shape = {}'.format(stockpup_df.shape))
    print(print_dt, 'Nr of tickers blacklisted: {}'.format(tickers['bl'].sum()), print_block)
    return stockpup_df


def data_clean_stockpup(stockpup_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Manually clean incorrect data in stockpup.csv

    :param stockpup_df: The dataframe to clean
    :return: The cleaned dataframe
    """
    if stockpup_df is None:
        stockpup_df = pd.read_csv(stockpup_path + 'stockpup.csv', index_col=0)
    # NUE has extreme high Cash Flow items
    wrong_data = [{'ticker': 'NUE',
                   'quarter_end': ['1994-04-02',
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
                   'columns': ['CF_cash_from_operating_activities',
                               'CF_cash_from_investing_activities',
                               'CF_cash_from_financing_activities',
                               'CF_cash_change_during_period',
                               'CF_capital_expenditures']}]
    for wd in wrong_data:
        t = wd['ticker']
        for qe in wd['quarter_end']:
            for col in wd['columns']:
                mask = (stockpup_df['ticker'] == t) & (stockpup_df['quarter_end'] == qe)
                stockpup_df.loc[mask, col] = np.nan

    # Remove extreme high P/E ratio's
    stockpup_df.loc[stockpup_df['RAT_p/e_ratio'] > 100, 'RAT_p/e_ratio'] = 100
    # Remove extreme high P/B ratio's
    stockpup_df.loc[stockpup_df['RAT_p/b_ratio'] > 100, 'RAT_p/b_ratio'] = 100
    # Remove extreme high Current ratio's
    stockpup_df.loc[stockpup_df['RAT_current_ratio'] > 20, 'RAT_current_ratio'] = 20

    # Save
    stockpup_df.to_csv(stockpup_path + 'stockpup.csv')
    print(print_block, print_dt, 'Stockpup cleaned ... shape = {}'.format(stockpup_df.shape), print_block)

    return stockpup_df


#######################################################################################################################
# YAHOO QUOTES
#######################################################################################################################
def make_yahoo_quote_data(start_ticker: str = 'A', end_ticker: str = '^^^',
                          start_date: str = '1970-01-01', end_date: str = '2020-01-01', redownload: bool = True) -> None:
    """
    This function uses the 3th party library 'yahoo_historical' to download daily historical quotes for tickers.
    The quotes are saved as csv files

    :param start_date: First day to download quotes
    :param end_date: Last day to download quotes
    :param start_ticker: First ticker to download quotes from
    :param end_ticker: Last ticker to download historical quotes from
    :param redownload: If True, then download quotes even if they're already downloaded before.
    :return: None
    """
    start_date = [int(n) for n in start_date.split('-')]
    end_date = [int(n) for n in end_date.split('-')]
    tickers = pd.read_csv(raw_data_path + 'tickers.csv', index_col=0)
    indices = pd.DataFrame(['^GSPC', '^DJI', '^IXIC', '^N225', '^VIX', '^TNX', '^TYX', '^FVX', '^IRX'], columns=['ticker'])
    tickers = pd.concat([tickers, indices], sort=True)
    tickers_downloaded = mos.get_filenames(yahoo_quotes_path + 'csv/', ext='csv')
    tickers_downloaded = [t.split('.csv')[0] for t in tickers_downloaded]
    for _, ticker in enumerate(tickers['ticker']):
        print(ticker)
        if (ticker < start_ticker) | (ticker > end_ticker): continue
        if (not redownload) & (ticker in tickers_downloaded): continue
        print('{}: Process Yahoo Quotes ticker: {}'.format(date_time, ticker))
        try:
            # Download
            quotes = pd.DataFrame(Fetcher(ticker, start_date, end_date, interval='1d').getHistorical())
            if quotes.empty: raise KeyError  # Yahoo sometimes returns empty dataframe if company is delisted or aquired
            quotes.columns = [c.lower().replace(' ', '_') for c in quotes.columns]
            # Save
            quotes.to_csv(yahoo_quotes_path + 'csv/' + ticker + '.csv')
        except KeyError:
            tickers = _blacklist_ticker(ticker, 'bl_yhoo_quotes')
        except (ConnectionError, gaierror):  # Todo: gaierror doesn't seem to work + stall if connection lost
            print(print_block, '\n{}: Connection error'.format(date_time))
            print('{}: Last ticker: {}\n'.format(date_time, ticker), print_block)

    print(print_block, print_dt, 'Nr of tickers blacklisted: {}'.format(tickers['bl'].sum()), print_block)


#######################################################################################################################
# YAHOO INFO
#######################################################################################################################
def make_yahoo_info_data(start_ticker: str = 'A', end_ticker: str = 'ZZZZ', headless: bool = False):
    """
    Function to create sector, industry and description data for each ticker.
    It uses the 3th party library 'selenium' to download the info from yahoo and saves that info into a big datafile

    :param start_ticker: First ticker to download yahoo info
    :param end_ticker: Last ticker to download yahoo info
    :param headless: If True, then the selenium browser will run in the background.
    :return:
    """
    yahoo_info_df = pd.DataFrame(columns=['ticker', 'sector', 'industry', 'description'])
    driver = _get_selenium_driver(headless)

    tickers = pd.read_csv(raw_data_path + 'tickers.csv', index_col=0)
    for _, ticker in enumerate(tickers['ticker']):
        if (ticker < start_ticker) | (ticker > end_ticker): continue
        print('{}: Process Yahoo info ticker: {}'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), ticker))
        try:
            # Download
            url = 'https://finance.yahoo.com/quote/' + ticker + '/profile?p=A&guccounter=1'
            driver.get(url)
            # Check if on correct webpage
            # If on https://guce.oath.com/collectConsent?sessionId=3_cc-session_9c0c4cff-7523-4078-ad58-2e6be1b2606a&lang=&inline=false
            # Click to redirect to yahoo finance page
            if 'collectConsent' in driver.current_url:
                # Click away redirection
                driver.find_element_by_xpath('/html/body/div/div/div/form/div/button[2]').click()
                driver.find_element_by_xpath('//*[@id="loaderContainer"]/p/a').click()
            assert 'https://finance.yahoo.com/quote/' + ticker + '/profile' in driver.current_url

            # Get info
            sector_industry_employees = driver.find_element_by_xpath('//*[@id="Col1-0-Profile-Proxy"]/section/div[1]/div/div/p[2]')
            sector_industry_employees = sector_industry_employees.text

            sector = re.search('Sector: (.*)', sector_industry_employees).group(1)
            industry = re.search('Industry: (.*)', sector_industry_employees).group(1)
            employees = re.search('Employees: (.*)', sector_industry_employees).group(1)
            description = driver.find_element_by_xpath('//*[@id="Col1-0-Profile-Proxy"]/section/section[2]/p').text

            # Add info
            row = {'ticker': ticker, 'sector': sector, 'industry': industry, 'description': description}
            yahoo_info_df = yahoo_info_df.append(row, ignore_index=True)

        except (AssertionError, NoSuchElementException):
            tickers = _blacklist_ticker(ticker, 'bl_yahoo_info')
    # Save
    yahoo_info_df.to_csv(yahoo_info_path + 'yahoo_info.csv')
    print(print_block, print_dt, 'Yahoo info saved ... shape = {}'.format(yahoo_info_df.shape))
    print(print_dt, 'Nr of tickers blacklisted: {}'.format(tickers['bl'].sum()), print_block)


#######################################################################################################################
# EDGAR
#######################################################################################################################

# Directory structure See: https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm

def make_edgar_filing_list(start_ticker: str = 'A', end_ticker: str = 'ZZZZ', headless: bool = True, redownload: bool = False):
    # Todo: Ex IBM: 424B5 filings return blank line, Ex EQ: SC 13G/A filinges return blank line
    # Todo: Ex BAC: Has huge amounts filings and breaks with >2000 filings => only &dateb= iso
    tickers_downloaded = mos.get_filenames(edgar_data_path, ext='csv') # Todo: put this in ticker loop to avoid start end ticker ????
    tickers_downloaded = [t.split('.csv')[0] for t in tickers_downloaded]
    driver = _get_selenium_driver(headless=headless)
    tickers = pd.read_csv(raw_data_path + 'tickers.csv', index_col=0)
    for _, ticker in enumerate(tickers['ticker']):
        if (ticker < start_ticker) | (ticker > end_ticker): continue
        if (not redownload) & (ticker in tickers_downloaded): continue

        print('{}: Process Edgar filing list: {}'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), ticker))

        ticker_filing_df = pd.DataFrame(columns=['ticker', 'type', 'filing_date', 'period_of_report', 'items', 'accession_no'])
        try:
            params = {'CIK': ticker,  # Edgar allows ticker as CIK
                      'type': '',  # all or 10-K, 10-Q, 8-K, 11-K, ... Full list at https://www.sec.gov/forms
                      'owner': 'exclude',  # exclude or include.
                      'output': 'xml',  # xml, html
                      'dateb': '',  # Prior to yyyymmdd
                      'count': 100,  # Max results per page. 100 quarters is +/- 33 years if no amends (3x10Q + 1x10k per year)
                      }
            base_url = 'https://www.sec.gov/cgi-bin/browse-edgar' \
                       '?action=getcompany&CIK={CIK}&type={type}&dateb={dateb}&owner={owner}&count={count}&output={output}'.format(**params)

            # collect all links
            url_list = []
            stop = False
            while not stop:
                print('{}: Collect links: {}'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), len(url_list)))
                url = base_url + '&start={}'.format(len(url_list))
                driver.get(url)
                filing_tabel = driver.find_element_by_xpath('//*[@id="results"]/table')
                stop = True  # Assume last page with no elements in the tabel
                for element in filing_tabel.find_elements_by_tag_name('a'):  # Todo: are tags easier than xpath's here?
                    link = element.get_attribute('href')
                    url_list.append(link)
                    stop = False  # This was not the last page
            url_list = set(url_list)  # remove duplcate links if they exist

            # Goto each link and scrape info
            for url in url_list:
                driver.get(url)
                # Xpaths to data on Filing Detail page
                var_xpaths = [['type', '//*[@id="formName"]/strong'],
                              ['period_of_report', '//*[@id="formDiv"]/div[2]/div[2]/div[2]'],
                              ['filing_date', '//*[@id="formDiv"]/div[2]/div[1]/div[2]'],  # Todo: period of report not always correct: example Ebay
                              ['items', '//*[@id="formDiv"]/div[2]/div[3]/div[2]'],
                              ['accession_no', '//*[@id="secNum"]']]
                row = {}
                for var, xpath in var_xpaths:
                    try:
                        row[var] = driver.find_element_by_xpath(xpath).text
                    except NoSuchElementException as e:
                        pass # Todo: does this creates empty rows ????
                ticker_filing_df = ticker_filing_df.append(row, ignore_index=True)
                ticker_filing_df['ticker'] = ticker

            if ticker_filing_df.empty: raise NoSuchElementException  # Ex: LIZ

            ticker_filing_df['type'] = ticker_filing_df['type'].str.split('Form ', expand=True)[1]
            ticker_filing_df['accession_no'] = ticker_filing_df['accession_no'].str.split('SEC Accession No. ', expand=True)[1]
            # Todo: period of report not always correct: example Ebay
            ticker_filing_df[ticker_filing_df['period_of_report'].isna()] = 'XXX'  # With nan's in Series we can't do ~df
            ticker_filing_df[~ticker_filing_df['period_of_report'].str.contains(r'^[12]\d{3}-')] = np.nan

            ticker_filing_df = ticker_filing_df.sort_values(by='filing_date')
            ticker_filing_df.to_csv(edgar_data_path + ticker + '.csv')
        except NoSuchElementException:
            tickers = _blacklist_ticker(ticker, 'bl_edgar')


# def make_edgar_insider_transaction_list(start_ticker: str = 'A', end_ticker: str = 'ZZZZ', headless: bool = True):
#     driver = _get_selenium_driver(headless=headless)
#     tickers = pd.read_csv(raw_data_path + 'tickers.csv', index_col=0)
#     for _, ticker in enumerate(tickers['ticker']):
#         if (ticker < start_ticker) | (ticker > end_ticker): continue
#
#
def download_edgar_filings(start_ticker: str = 'A', end_ticker: str = 'ZZZZ', filing_types=None,
                           redownload: bool = False, headless: bool = True):
    # Todo: use this routine for downloading any filing.

    # Todo: implement download missing filings
    # Todo: implement reconstructing edgar raw files
    # Todo: it's more efficient to scrape filing dates from http://rankandfiled.com/
    """

    :param start_ticker:
    :param end_ticker:
    :param filing_types:
    :param redownload:
    :param headless:
    :return:
    """
    if filing_types is None: filing_types = ['10-K', '10-Q']
    driver = _get_selenium_driver(headless=headless)
    tickers = pd.read_csv(raw_data_path + 'tickers.csv', index_col=0)
    tickers_downloaded = mos.get_filenames(edgar_data_path, ext='csv')
    tickers_downloaded = [t.split('.csv')[0] for t in tickers_downloaded]
    for _, ticker in enumerate(tickers['ticker']):
        # Manualy block tickers
        if ticker == 'AEP': continue  # alway's timeout after 3th link
        if ticker == 'ETR': continue  # alway's timeout after 1st link
        if ticker == 'O': continue  # alway's timeout after 7st link

        if (ticker < start_ticker) | (ticker > end_ticker): continue
        if (not redownload) & (ticker in tickers_downloaded): continue

        print('{}: Process Edgar ticker: {}'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), ticker))
        try:
            for type in filing_types:
                params = {'CIK': ticker,  # Edgar allows ticker as CIK
                          'type': type,  # all or 10-K, 10-Q, 8-K, 11-K, ... Full list at https://www.sec.gov/forms
                          'owner': 'exclude',  # exclude or include.
                          'output': 'xml',
                          'dateb': '',  # Prior to yyyymmdd
                          'count': 100,  # Max results per page. 100 quarters is +/- 33 years if no amends (3x10Q + 1x10k per year)
                          }
                url = 'https://www.sec.gov/cgi-bin/browse-edgar' \
                      '?action=getcompany&CIK={CIK}&type={type}&dateb={dateb}&owner={owner}&count={count}&output={output}'.format(**params)
                driver.get(url)
                # Download
                filing_tabel = driver.find_element_by_xpath('//*[@id="results"]/table')
                # collect all links to filing
                url_list = []
                for element in filing_tabel.find_elements_by_tag_name('a'):
                    link = element.get_attribute('href')
                    # To get the complete submission text file remove '-index.htm' and add '.txt'
                    link = link[:link.rfind("-")] + '.txt'
                    url_list.append(link)

                # Goto each link and save filing
                for link in url_list:
                    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()), link)
                    driver.get(link)
                    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()), 'Downloaded')

                    statement = driver.page_source
                    filing_type = re.search('(?<=CONFORMED SUBMISSION TYPE:\t)(.*)', statement).group(1)
                    quarter_end = re.search('(?<=CONFORMED PERIOD OF REPORT:\t)(.*)', statement).group(1)
                    filing_type = filing_type.replace('/', '-')  # for 10-K/A etc

                    # Save the filing
                    fname = '{}_{}_{}.txt'.format(ticker, quarter_end, filing_type)
                    directory = edgar_data_path + 'filings/' + ticker + '/'
                    if not mos.check_dir_exists(directory)['success']: mos.create_directory(directory)
                    with open(directory + fname, 'w') as f:
                        f.write(statement)

        except AttributeError:
            # Ex: https://www.sec.gov/Archives/edgar/data/815094/0000950109-95-000983.txt
            # Has no standard filing date.
            print(print_dt, 'Filing for ticker: {} missing info: {}'.format(ticker, link))
            pass
        except NoSuchElementException:
            tickers = _blacklist_ticker(ticker, 'bl_edgar')

    print(print_dt, 'Nr of tickers blacklisted: {}'.format(tickers['bl'].sum()), print_block)


#######################################################################################################################
# TOOLS
#######################################################################################################################

def _blacklist_ticker(ticker: str, bl_column: str) -> pd.DataFrame:
    """
    Helper function to blacklist ticker by putting the session date_time in the dataset's bl-column
    and set the general bl-columnt to True

    :param ticker: ticker to blacklist
    :param bl_column: blacklist column name
    :return: tickers
    """
    tickers = pd.read_csv(raw_data_path + 'tickers.csv', index_col=0)
    tickers.loc[tickers['ticker'] == ticker, 'bl'] = True
    tickers.loc[tickers['ticker'] == ticker, bl_column] = date_time
    tickers.to_csv(raw_data_path + 'tickers.csv')
    print(print_arrow, 'Blacklisted {}'.format(ticker))
    return tickers


def _get_selenium_driver(headless: bool = True, log: bool = False):
    """
    Helper function to set the options for the selenium driver.
    It returns a selenium web driver.

    """

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option("detach", True)  # Keep browser open
    chrome_options.headless = headless
    chrome_options.add_argument('blink-settings=imagesEnabled=false')  # Don't download images
    chrome_options.add_argument('--no-sandbox')  # selenium.common.exceptions.WebDriverException: Message: unknown error: session deleted because of page crash
    chrome_options.add_argument('--start-maximized')
    chrome_options.add_argument('--dissable-gpu')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-infobars')
    chrome_options.add_argument('--disable-dev-shm-usage')  # overcome limited resource problems
    chrome_options.add_argument('--proxy-server="direct://"')  # should speedup headless but doesn't work See: https://github.com/Codeception/CodeceptJS/issues/561
    chrome_options.add_argument('--proxy-bypass-list=*')  # should speedup headless but doesn't work See: https://github.com/Codeception/CodeceptJS/issues/561
    driver = webdriver.Chrome(chrome_options=chrome_options,
                              service_args=[
                                  # '--verbose',
                                  '--log-path=//home/md/Temp/Selenium_log'])
    # Set Chrome bandwidth
    driver.set_network_conditions(offline=False,
                                  latency=0,  # additional latency (ms)
                                  download_throughput=500000 * 1024,  # maximal throughput
                                  upload_throughput=500000 * 1024)  # maximal throughput
    return driver


def _compress_edgar_filings(delete_txt: bool = True, tickers: list = None):
    """
    Compresses txt filings
    :param delete_txt: If True, delete txt file
    :param tickers: List of tickers. If None, comresses all ticker filings
    :return: None
    """
    if not tickers:
        tickers = mos.get_directories(edgar_data_path + 'filings/')
    tickers.sort()
    print(print_block, print_dt, 'Tickers to compress: {}'.format(tickers), print_block)
    for ticker in tickers:
        print(print_dt, 'Compressing:', ticker)
        os.chdir(edgar_data_path + 'filings/' + ticker + '/')
        files = mos.get_filenames('.', ext='txt')
        files.sort()
        for file in files:
            zip_file = file.split('.')[0] + '.zip'
            ZipFile(zip_file, 'w', ZIP_DEFLATED).write(file, file)
            if delete_txt: os.remove(file)


def _uncompress_edgar_filings(tickers: list, delete_zip: bool = True, ):
    """
    Unompresses zip filings
    :param delete_zip: If True, delete zip file
    :param tickers: List of tickers.
    :return: None
    """
    for ticker in tickers:
        print(ticker)
        os.chdir(edgar_data_path + 'filings/' + ticker + '/')
        files = mos.get_filenames('.', ext='zip')
        files.sort()
        for file in files:
            print(file)
            ZipFile(file, 'r').extractall('.')
            if delete_zip: os.remove(file)


def x():
    file = pd.DataFrame(columns=['ticker', 'quarter_end', 'filing_date', 'link', 'fname'])
    tickers = pd.read_csv(raw_data_path + 'tickers.csv', index_col=0)
    for _, ticker in enumerate(tickers['ticker']):
        try:
            print(ticker)
            file = pd.read_csv(edgar_data_path + ticker + '.csv', index_col=0)
            file.columns = [c.lower() for c in file.columns]
            file.to_csv(edgar_data_path + ticker + '.csv')
        except FileNotFoundError:
            _blacklist_ticker(ticker, 'bl_edgar')


if __name__ == '__main__':
    start_ticker = input('Start ticker = ')
    end_ticker = input('End ticker = ')
    # make_tickers()
    make_stockpup_data()
    data_clean_stockpup()
    # make_yahoo_quote_data(start_ticker,end_ticker,redownload=True)
    # make_yahoo_info_data(headless=True)
    # make_edgar_filing_lists(start_ticker, end_ticker, redownload=False, headless=True)
    # x()

    pass
