import pandas as pd
from flask import Flask, render_template, request,jsonify
from NSE.get_nifty50_livedata_nse import get_nifty50_livedata_nse
from ML_Algorithms.ml_model import exponential_smooth,get_indicator_data,produce_prediction,cross_Validation
import os
from datetime import date

TEMPLATE_DIR = os.path.abspath('../templates')
STATIC_DIR = os.path.abspath('../styles')
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)


@app.route("/")
def future_price_predict():
    nifty50_list = get_nifty50_livedata_nse()
    data = pd.read_csv(r"/NSE/nifty50_1yr_dataset/ADANIPORTS_hist_data.csv")
    data = exponential_smooth(data,0.65)
    data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'},inplace=True)
    data = get_indicator_data(data)
    data = produce_prediction(data, window=5)
    nifty50_list = pd.DataFrame(nifty50_list)
    for col in ['ema50', 'ema21', 'ema15', 'ema5', 'normVol', 'pred']:
        nifty50_list[col] = data[col].copy()
    # print(data.columns.tolist())
    # # print(data[data["pred"]==0.0])
    data = data.dropna()
    dat = cross_Validation(data)
    nifty50_list = nifty50_list.to_dict('records')
    print(nifty50_list)
    return render_template("index.html", context={"data": nifty50_list})





if __name__ == "__main__":
    app.run(debug = True)
