import os
import pandas as pd
from nsepy import get_history
from constants import *
from datetime import date, datetime
import re

regexp = re.compile('&')


def get_nifty50_livedata_nse():
    nifty50_url = NIFTY50_URL
    df_n50 = pd.read_csv(nifty50_url)
    df_n50.rename(columns={"Company Name": "CompanyName"}, inplace=True)
    current_price = []
    for index, row in df_n50.iterrows():
        stock_code = row['Symbol']
        if (regexp.search(stock_code) != None):
            stock_code = stock_code.replace('&', '%26')
        current_date = datetime.now()
        data = get_history(symbol=stock_code, start=date(current_date.year, current_date.month, current_date.day),
                           end=date(current_date.year, current_date.month, current_date.day))
        current_price.append(data["Last"].values if data["Last"].values else 0.0)
    df_n50["current_price"] = pd.DataFrame(current_price)
    return df_n50.to_dict('records')


def get_historical_data_nifty50(startdate: date, enddate: date):
    """Model training purpose"""
    nifty50_hist_data = pd.DataFrame()
    nifty50_url = NIFTY50_URL
    df_n50 = pd.read_csv(nifty50_url)
    df_n50.rename(columns={"Company Name": "CompanyName"}, inplace=True)
    for index, row in df_n50.head(1).iterrows():
        stock_code = row['Symbol']
        if (regexp.search(stock_code) != None):
            stock_code = stock_code.replace('&', '%26')
        data = get_history(symbol=stock_code, start=date(startdate.year, startdate.month, startdate.day),
                           end=date(enddate.year, enddate.month, enddate.day))
        nifty50_hist_data = nifty50_hist_data.append(data)
    nifty50_hist_data.to_csv(os.getcwd() + '\\NSE\\nifty50_1yr_dataset\\' + f"{stock_code}_hist_data.csv")
    return True
