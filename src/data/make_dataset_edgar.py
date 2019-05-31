# --------------------------------------------------------------------------------------------------------
# 2019/05/24
# fundamental_stock_analysis - make_dataset_edgar.py
# md
# --------------------------------------------------------------------------------------------------------

from pathlib import Path
from time import strftime, gmtime
import re
from time import strftime, gmtime

import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from my_tools.my_toolbox import MyOsTools as mos
import os
from zipfile import ZipFile, ZIP_DEFLATED

base_data_path = '/mnt/Development/My_Projects/fundamental_stock_analysis/data/'
raw_data_path = 'raw/edgar/'

date_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())  # Session datetime


def download_edgar_data(headless=True, save_filing=True, start_from: str = 'A', end_till: str = 'ZZZZZ', redownload=False):
    # We choose Selenium iso Requests because it allows to fill in forms, it's more 'human like' and less likely to be flagged,
    # however, Requests is much faster

    chrome_options = webdriver.ChromeOptions()
    chrome_options.headless = headless  # Todo: Headless seems much slower. Max 600KB/sec iso 3MB/sec !!!!
    chrome_options.add_argument('blink-settings=imagesEnabled=false')  # Don't download images
    chrome_options.add_argument('--no-sandbox')  # selenium.common.exceptions.WebDriverException: Message: unknown error: session deleted because of page crash
    chrome_options.add_argument('--start-maximized')
    chrome_options.add_argument('--dissable-gpu')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('disable-infobars')
    chrome_options.add_argument('--disable-dev-shm-usage')  # overcome limited resource problems
    chrome_options.add_argument('--proxy-server="direct://"')  # should speedup headless but doesn't work See: https://github.com/Codeception/CodeceptJS/issues/561
    chrome_options.add_argument('--proxy-bypass-list=*')  # should speedup headless but doesn't work See: https://github.com/Codeception/CodeceptJS/issues/561

    tickers = pd.read_csv(base_data_path + 'raw/' + 'tickers.csv', index_col=0)
    edgar_fnames = mos.get_filenames(base_data_path + raw_data_path, ext='csv')

    driver = webdriver.Chrome(chrome_options=chrome_options,
                              service_args=[
                                  # '--verbose',
                                  '--log-path=//home/md/Temp/Selenium_logs']
                              )
    # Set Chrome babdwidth
    driver.set_network_conditions(offline=False,
                                  latency=0,  # additional latency (ms)
                                  download_throughput=50000 * 1024,  # maximal throughput
                                  upload_throughput=50000 * 1024)  # maximal throughput

    for ticker in tickers['Ticker'].values:
        # if ticker < start_from: continue
        if start_from > ticker: continue
        if ticker > end_till: continue
        if ticker == 'AEP': continue  # Todo: alway's timeout after 3th link
        if ticker == 'ETR': continue  # Todo: alway's timeout after 1st link
        if ticker == 'O': continue  # Todo: alway's timeout after 7st link

        if not redownload and ticker + '.csv' in edgar_fnames:
            print('===> Ticker already downloaded:', ticker)
            continue
        print(ticker)
        params = {'CIK': ticker,  # Edgar allows ticker as CIK
                  'type': '',  # all or 10-K, 10-Q, 8-K, 11-K, ... Full list at https://www.sec.gov/forms
                  'owner': 'exclude',  # exclude or include.
                  'output': 'xml',
                  'dateb': '',  # Prior to yyyymmdd
                  'count': 100,  # Max results per page. 100 quarters is +/- 33 years if no amends (3x10Q + 1x10k per year)
                  }
        filings = pd.DataFrame()
        for type in ['10-K', '10-Q']:
            params['type'] = type
            url = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany' \
                  '&CIK={CIK}&type={type}&dateb={dateb}&owner={owner}&count={count}&output={output}'.format(**params)

            try:
                driver.get(url)
                # Get the table
                filing_results = driver.find_element_by_xpath('//*[@id="results"]/table')
                # collect all links to filing
                url_list = []
                for element in filing_results.find_elements_by_tag_name('a'):
                    link = element.get_attribute('href')
                    # To get the complete submission text file remove '-index.htm' and add '.txt'
                    link = link[:link.rfind("-")] + '.txt'
                    url_list.append(link)

                # Goto each link, extract type, period, filing date and save filing
                for link in url_list:
                    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()), link)
                    driver.get(link)
                    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()), 'Downloaded')
                    # element_present = EC.presence_of_element_located((By.ID, 'element_id'))
                    # WebDriverWait(driver, timeout=5000).until(element_present)

                    report = driver.page_source
                    filing_type = re.search('(?<=CONFORMED SUBMISSION TYPE:\t)(.*)', report).group(1)
                    quarter_end = re.search('(?<=CONFORMED PERIOD OF REPORT:\t)(.*)', report).group(1)
                    filing_date = re.search('(?<=FILED AS OF DATE:\t\t)(.*)', report).group(1)  # Todo: make more generic without \t\t
                    filing_type = filing_type.replace('/', '-')  # for 10-K/A etc
                    fname = '{}_{}_{}.txt'.format(ticker, quarter_end, filing_type)
                    filings = filings.append({'Ticker': ticker, 'Type': filing_type,
                                              'Quarter end': quarter_end, 'Filing date': filing_date, 'Link': link,
                                              'Fname': fname}, ignore_index=True)
                    if save_filing:
                        directory = base_data_path + raw_data_path + 'filings/' + ticker + '/'
                        res = mos.check_dir_exists(directory)
                        if not res['success']:
                            mos.create_directory(directory)
                        with open(directory + fname, 'w') as f:
                            f.write(report)

            except (NoSuchElementException, AttributeError):
                # Todo: blacklist not always correct. Fi: AVP, https://www.sec.gov/Archives/edgar/data/8868/0000008868-95-000002.txt get blacklist, but only that doc doesn't contain "filing date"
                print('===> Blacklisted:', ticker)
                # Append if exist otherwise create new tickers_blacklist.csv
                mode = 'a'
                header = False
                if 'tickers_blacklist.csv' not in mos.get_filenames(path=base_data_path + 'raw/', ext='csv'):
                    print('===> Create new tickers_blacklist.csv')
                    mode = 'w'
                    header = True
                tickers_blacklist = pd.DataFrame([[date_time, ticker, 'edgar']], columns=['Date', 'Ticker', 'Source'])
                tickers_blacklist.to_csv(base_data_path + 'raw/' + 'tickers_blacklist.csv', mode=mode, header=header, index=False)
                # Mark ticker as blacklisted in tickers.csv
                tickers.loc[tickers['Ticker'] == ticker, 'Blacklist Edgar'] = True
            except TimeoutException:
                print('===> TIMEOUT!')
                print('===> Kill driver')
                driver.quit()
                raise

        if not filings.empty:
            filings.to_csv(base_data_path + raw_data_path + ticker + '.csv')
            tickers.to_csv(base_data_path + 'raw/' + 'tickers.csv', mode='w', index=True)

    driver.quit()


def compress_edgar_filings(delete_txt: bool = True, tickers: list = None):
    """
    Compresses txt filings
    :param delete_txt: If True, delete txt file
    :param tickers: List of tickers. If None, comresses all ticker filings
    :return: None
    """
    if not tickers:
        tickers = mos.get_directories(base_data_path + raw_data_path + 'filings/')
    tickers.sort()
    print('Tickers to compress:', tickers)
    for ticker in tickers:
        print('Compressing:', ticker)
        os.chdir(base_data_path + raw_data_path + 'filings/' + ticker + '/')
        files = mos.get_filenames('.', ext='txt')
        files.sort()
        for file in files:
            zip_file = file.split('.')[0] + '.zip'
            ZipFile(zip_file, 'w', ZIP_DEFLATED).write(file, file)
            if delete_txt: os.remove(file)


def uncompress_edgar_filings(tickers: list, delete_zip: bool = True, ):
    """
    Unompresses zip filings
    :param delete_zip: If True, delete zip file
    :param tickers: List of tickers.
    :return: None
    """
    for ticker in tickers:
        print(ticker)
        os.chdir(base_data_path + raw_data_path + 'filings/' + ticker + '/')
        files = mos.get_filenames('.', ext='zip')
        files.sort()
        for file in files:
            print(file)
            ZipFile(file, 'r').extractall('.')
            if delete_zip: os.remove(file)


if __name__ == '__main__':
    # download_edgar_data(headless=False, start_from='A', end_till='ZZZZZ', redownload=False)
    compress_edgar_filings()
