# --------------------------------------------------------------------------------------------------------
# 2019/05/20
# fundamental_stock_analysis - yahoo_add_ticker_info.py
# md
# --------------------------------------------------------------------------------------------------------
import re
from time import strftime, gmtime

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from src.my_tools.my_toolbox import MyOsTools as mos

base_data_path = '/mnt/Development/My_Projects/fundamental_stock_analysis/data/'
raw_data_path = 'raw/yahoo_info/'

date_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())  # Session datetime


def download_yahoo_info(headless: bool = True):
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)
    tickers_info = pd.DataFrame(columns=['Ticker', 'Sector', 'Industry', 'Employees', 'Description'])
    tickers = pd.read_csv(base_data_path + 'raw/' + 'tickers.csv', index_col=0)
    for ticker in tickers['Ticker'].values:
        print(ticker)
        url = 'https://finance.yahoo.com/quote/' + ticker + '/profile?p=A&guccounter=1'
        try:
            driver.get(url)
            # Check if on correct webpage
            # If on https://guce.oath.com/collectConsent?sessionId=3_cc-session_9c0c4cff-7523-4078-ad58-2e6be1b2606a&lang=&inline=false
            # Click to redirect to yahoo finance page
            if 'https://guce.oath.com' in driver.current_url:
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

            # Todo: implement corporate governance info
            # corp_governance = driver.find_element_by_xpath('// *[ @ id = "Col1-0-Profile-Proxy"] / section / section[3] / div / p')
            # print(corp_governance.text)

            tickers_info = tickers_info.append({'Ticker': ticker, 'Sector': sector, 'Industry': industry, 'Employees': employees, 'Description': description}, ignore_index=True)

        except (AssertionError, NoSuchElementException):  # Correct webpage doesn't exist, redirection failed or ellement doesn't exists
            print('===> Blacklisted:', ticker)
            # Append if exist otherwise create new tickers_blacklist.csv
            mode = 'a'
            header = False
            if 'tickers_blacklist.csv' not in mos.get_filenames(path=base_data_path + 'raw/', ext='csv'):
                print('===> Create new tickers_blacklist.csv')
                mode = 'w'
                header = True
            tickers_blacklist = pd.DataFrame([[date_time, ticker, 'yahoo_info']], columns=['Date', 'Ticker', 'Source'])
            tickers_blacklist.to_csv(base_data_path + 'raw/' + 'tickers_blacklist.csv', mode=mode, header=header, index=False)
            # Mark ticker as blacklisted in tickers.csv
            tickers.loc[tickers['Ticker'] == ticker, 'Blacklist YI'] = True
    # Update tickers.csv
    tickers.to_csv(base_data_path + 'raw/' + 'tickers.csv', mode='w', index=True)
    tickers_info.to_csv(base_data_path + raw_data_path + 'yahoo_info_raw_data.csv', mode='w', index=True)


if __name__ == '__main__':
    download_yahoo_info()
