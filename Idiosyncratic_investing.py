#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 09:58:03 2020

@author: owen
"""

from bs4 import BeautifulSoup as soup
import requests
import datetime as dt
import numpy as np
import pandas as pd
import pandas_datareader as web
from arch.univariate import LS, GARCH

# scrape LQ45 companies
page = requests.get('https://en.wikipedia.org/wiki/LQ45')
soup = soup(page.text, 'lxml')

y = []
tickers = []
total = pd.DataFrame()
for row in soup.find_all('td'):
    y.append(row.get_text().strip())

for i in range(0, len(y), 2):
    tickers.append(y[i])

tickers = [ticker + '.JK' for ticker in tickers]

# period
start = dt.datetime(2015,1,1)
end = dt.datetime(2020,1,1)

# garch model
def idiosyncratic_forecast(x,y,p,o,q):
    ls = LS(y,x)
    ls.volatility = GARCH(p=p,o=o,q=q)
    res = ls.fit()
    forecast = res.forecast(horizon=2)
    return forecast.residual_variance[-1:]['h.2'].to_list()[0]

# get market return
market_price = web.get_data_yahoo('^JKLQ45', start,end)
market_ret = (np.log(market_price['Adj Close']) - np.log(market_price['Adj Close']).shift(1)).dropna()
week_market_ret = market_ret.resample('W').sum()
week_market_ret = week_market_ret*100

# get factor
df = pd.DataFrame()
date = []
period = []
company = []
forecast_eivol = []
ret = []
for ticker in tickers:
    
    # get stock return
    stock_price = web.get_data_yahoo(ticker, start, end)
    stock_ret = (np.log(stock_price['Adj Close']) - np.log(stock_price['Adj Close']).shift(1)).dropna()
    
    # weekly resample and scaling
    week_ret = stock_ret.resample('W').sum()
    week_ret = week_ret*100
    
    # get forecast
    forecast = []
    test_size = 30
    for i in range(0,len(week_ret)-(test_size)):
        ivol = idiosyncratic_forecast(week_market_ret[:(test_size+i)], week_ret[:(test_size+i)], 3, 1, 1)
        forecast.append(ivol)
       
    date += week_ret[test_size:].index
    period += list(range(1,len(forecast)+1))
    company += [ticker]*len(forecast)
    forecast_eivol += forecast
    ret += week_ret[test_size:].to_list()

df = pd.DataFrame({'DATE':date, 'PERIOD':period, 'COMP':company, 'EIVOL':forecast_eivol, 'RET':ret})
#df.to_excel('LQ45_CAPM_1.xlsx')

# back-testing strategy
# long top 10% idiosyncratic volatility
df = df.set_index('PERIOD')
print(df)
ret = []
for i in range(1,df.index[-1]+1):
    p = df.loc[i]
    s = p.sort_values(by='EIVOL', ascending=False)
    ret.append(s.iloc[:5]['RET'].sum())
    

print('Average weekly return from 2015 to 2020: {}%'.format(np.mean(ret)))
print('Cumulative return from 2015 to 2020: {}%'.format(np.sum(ret))) 






