import pandas as pd
from flask import Flask, render_template, request, jsonify
from NSE.get_nifty50_livedata_nse import get_nifty50_livedata_nse
from ML_Algorithms.ml_model import exponential_smooth, get_indicator_data, produce_prediction, cross_Validation
import os

TEMPLATE_DIR = os.path.abspath('templates')
STATIC_DIR = os.path.abspath('styles')
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)


@app.route("/")
def future_price_predict():
    nifty50_list = get_nifty50_livedata_nse()
    # Todo: only `ADANIPORTS` stock  is model train with as time was limited.
    # for exatracting historical data for other symbols you use `get_historical_data_nifty50` function from `NSE` folder
    data = pd.read_csv(os.getcwd() + r"\NSE\nifty50_1yr_dataset\ADANIPORTS_hist_data.csv")
    data = exponential_smooth(data, 0.65)
    data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'},
                inplace=True)
    data = data.dropna()
    data = get_indicator_data(data)
    data = produce_prediction(data, window=5)  # here the price prediction is done
    nifty50_list = pd.DataFrame(nifty50_list)
    for col in ['ema50', 'ema21', 'ema15', 'ema5', 'normVol', 'pred']:
        nifty50_list[col] = data[col].copy()
    data = data.dropna()
    dat = cross_Validation(data)
    nifty50_list["RF_Accuracy"] = float(dat["RF_Accuracy"]) * 100
    nifty50_list["KNN_Accuracy"] = float(dat["KNN_Accuracy"]) * 100
    nifty50_list["ENSEMBLE_Accuracy"] = float(dat["ENSEMBLE_Accuracy"]) * 100
    nifty50_list = nifty50_list.to_dict('records')
    return render_template("index.html", context={"data": nifty50_list})


if __name__ == "__main__":
    # Calls the run method, runs the app on port 5000
    app.run(host='0.0.0.0', port='5000')
