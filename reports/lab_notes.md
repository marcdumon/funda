# Funda Lab Notes


## Data
### Data collection
The following data were collected:

- **Fundamental data**:  

  The data was collected using the 'Batch Link Downloader' Chrome extention. For each ticker a csv-file was saved under the name 'ticker.csv'. The column names were changed in following way:
  All upper cases were replaced by lower cases, spaces were replaced by underscores and a a upper case prefix indicating the data type was added. In the end all data was combined into one big csv-file named 'stockpup.csv'.  
  Mistakes in the raw data were found. Ticker NUE had extremely high cash flow numbers for the periods from 199-04-07 till 2000-07-01. This data has been deleted.  
  Many companies had extremely high P/E, P/B and current ratio's caused by very low (<< 1) denominator. For instance an earning of less than 0.001 $/share gives a P/E high P/E. This high ratio's have been clipped to 100 for P/E and P/B and to 20 for the current ratio's.

  - Source: [stockpup](http://www.stockpup.com/data)
  - Features:
      - ticker, quarter_end
      - INF_shares, INF_shares_split_adjusted, INF_split_factor, INC_revenue, INC_earnings, INC_earnings_available_for_common_stockholders
      - BAL_assets, BAL_current_assets, BAL_liabilities, BAL_current_liabilities, BAL_shareholders_equity, BAL_non-controlling_interest, BAL_preferred_equity, BAL_goodwill_&\_intangibles, BAL_long-term_debt
      - CF_cash_from_operating_activities, CF_cash_from_investing_activities, CF_cash_from_financing_activities, CF_cash_change_during_period,	CF_cash_at_end_of_period, CF_capital_expenditures
      - RAT_roe, RAT_roa, RAT_book_value_of_equity_per_share, RAT_p/b_ratio, RAT_p/e_ratio, RAT_cumulative_dividends_per_share,	RAT_dividend_payout_ratio, RAT_long-term_debt_to_equity_ratio, RAT_equity_to_assets_ratio, RAT_net_margin,	RAT_asset_turnover,	RAT_free_cash_flow_per_share, RAT_current_ratio, RAT_eps_basic, RAT_eps_diluted, RAT_dividend_per_share  


- **Price data**:

    The data was downloaded using the ['yahoo_historical'](https://github.com/AndrewRPorter/yahoo-historical) Python package. For each ticker a csv-file was saved under the name 'ticker.csv'.  
    Column names were changed to lower case.  
    Data was not available for delisted or bankrupt companies or for companies that changed their ticker.

    - Source: [Yahoo finance](http://finance.yahoo.com)
    - Features:
      - date
      - open,	high,	low,	close,	adj_close,	volume


- **Company information data**:

  The data was collected by scraping the Yahoo finance profile page with Selenium for each ticker. The data for each ticker was combined into a csv-file with the name 'yahoo_info.csv'.
  Data was not available for delisted or bankrupt companies or for companies that changed their ticker.

  - Source: [Yahoo finance profile](http://finance.yahoo.com)
  - Features:
    - ticker
    - sector,	industry,	description


- **Company filings data**:

  The data was collected by scraping the Edgar page with Selenium for each company. The data for each ticker was saved under the name 'ticker.csv'.   
  Some filing types like 424B5 or SC 13G/A could not be collected.

  - Source: [Edgar](https://www.sec.gov/edgar/searchedgar/companysearch.html)
  - Features:
    - ticker
    -	type,	filing_date,	period_of_report,	items

#### Results:
Stockpup provides data for **760 companies** starting **from 1993-06-30 till 2018-11-30**. For 207 companies it was impossible to download price, company information or filings data. One reason is that Yahoo doesn't provides data for delisted companies (bankruptcy, take-over, ...) or some companies changed their ticker and Stockpup, Yahoo and Edgar use different ticker for the same company.  The final dataset contains data for **516 companies**.




### Feature selection and creation



### Label creation
The procedure used to create the labels was:
  - calculate the percentage change for the closing price from the stock for up to 4 quarters.
  - calculate the percentage change for the closing price from indices ^DJI, ^GSPC and ^IXIC for up to 4 quarters, and take the average for the 3 indices.
  - calculate the excess gain and loss for each quarter. The formula is:
    $$
    \begin{cases}
    \ \ \ 1 & \text{if $return_{stock} - \frac{\sum_{i=1}^3 return_{index_i}}{3}>(1 + expected\ return_{stock})^{(r/4)}-1$}\\
    -1 & \text{if $return_{stock} - \frac{\sum_{i=1}^3 return_{index_i}}{3}<-(1 + expected\ return_{stock})^{(r/4)}-1$}\\
    \ \ \  0 & \text{otherwise}
    \end{cases}
    $$
  - the final label is 1 if two consecutive quarters with excess gain, -1 if two consecutive quarters with excess loss and 0 otherwise.  
The reason for choosing the excess returns iso the return of the stock was to select the best stocks, and the assumption is that this is  a stock that exceed the average return of indices when the market goes up or down. A good stock can also loose when the whole market is down, but less than average.

### Limitations

- Stockpup manually compiles its data and will likely contain more mistakes than the ones that were detected and corrected/removed.
- The features provided by Stockpup are limited. Many interesting features like R&D, COGS, etc are not available.
- The final dataset contains only 516 companies, which is less than 10% of the amount of companies on US stock market.
- The final dataset contains only companies that were operational at the time of the data collection and no delisted or bankrupt companies. The reason for this is that price and company information for delisted and bankrupt companies are not available on Yahoo.
- The final dataset is quite imbalanced. Roughly 60% has label = 0, 20% has label=1 and 20% has label=-1


## Baseline model

The baseline model takes as input 142 continuous and 5 categorical (dates + sector + industry). The categorical input is
