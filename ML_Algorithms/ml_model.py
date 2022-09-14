from Nifty50_future_price_prediction.constants import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
Next we clean our data and perform feature engineering to create new technical indicator features that our
model can learn from
"""
def exponential_smooth(data, alpha):
    """
    Function that exponentially smooths dataset so values are less 'rigid'
    :param alpha: weight factor to weight recent values more
    """

    return data.ewm(alpha=alpha).mean()


def get_indicator_data(data):
    """
    Function that uses the finta API to calculate technical indicators used as the features
    :return:
    """
    for indicator in INDICATORS:
        try:
            ind_data = eval('TA.' + indicator + '(data)')
        except:
            continue
        if not isinstance(ind_data, pd.DataFrame):
            ind_data = ind_data.to_frame()
        data = data.merge(ind_data, left_index=True, right_index=True)
    data.rename(columns={"14 period EMV.": '14 period EMV'}, inplace=True)

    # Also calculate moving averages for features
    data['ema50'] = data['close'] / data['close'].ewm(50).mean()
    data['ema21'] = data['close'] / data['close'].ewm(21).mean()
    data['ema15'] = data['close'] / data['close'].ewm(14).mean()
    data['ema5'] = data['close'] / data['close'].ewm(5).mean()

    # Instead of using the actual volume value (which changes over time), we normalize it with a moving volume average
    data['normVol'] = data['volume'] / data['volume'].ewm(5).mean()

    # Remove columns that won't be used as features
    del (data['open'])
    del (data['high'])
    del (data['low'])
    del (data['volume'])

    return data


def produce_prediction(data, window):
    """
    Function that produces the 'truth' values
    At a given row, it looks 'window' rows ahead to see if the price increased (1) or decreased (0)
    :param window: number of days, or rows to look ahead to see what the price did
    """

    prediction = (data.shift(-window)['close'] >= data['close'])
    prediction = prediction.iloc[:-window]
    data['pred'] = prediction.astype(int)

    return data


def cross_Validation(data):
    # Split data into equal partitions of size len_train

    num_train = 10  # Increment of how many starting points (len(data) / num_train  =  number of train-test sets)
    len_train = 40  # Length of each train-test set

    # Lists to store the results from each model
    rf_RESULTS = []
    knn_RESULTS = []
    gbt_RESULTS = []
    ensemble_RESULTS = []

    i = 0

    # Models which will be used
    rf = RandomForestClassifier()
    knn = KNeighborsClassifier()

    # Create a tuple list of our models
    estimators = [('knn', knn), ('rf', rf)]
    ensemble = VotingClassifier(estimators, voting='soft')

    while True:

        # Partition the data into chunks of size len_train every num_train days
        df = data.iloc[i * num_train: (i * num_train) + len_train]
        i += 1
        # print(i * num_train, (i * num_train) + len_train)

        if len(df) < 40:
            break

        y = df['pred']
        features = [x for x in df.columns if x not in ['pred']]
        X = df[features]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=7 * len(X) // 10, shuffle=False)

        # fit models
        rf.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        ensemble.fit(X_train, y_train)

        # get predictions
        rf_prediction = rf.predict(X_test)
        knn_prediction = knn.predict(X_test)
        ensemble_prediction = ensemble.predict(X_test)

        #         print('rf prediction is ', rf_prediction)
        #         print('knn prediction is ', knn_prediction)
        #         print('ensemble prediction is ', ensemble_prediction)
        #         print('truth values are ', y_test.values)

        # determine accuracy and append to results
        rf_accuracy = accuracy_score(y_test.values, rf_prediction)
        knn_accuracy = accuracy_score(y_test.values, knn_prediction)
        ensemble_accuracy = accuracy_score(y_test.values, ensemble_prediction)

        #         print(rf_accuracy)
        #         print(knn_accuracy)
        #         print(ensemble_accuracy)
        rf_RESULTS.append(rf_accuracy)
        knn_RESULTS.append(knn_accuracy)
        ensemble_RESULTS.append(ensemble_accuracy)

    print('RF Accuracy = ' + str(sum(rf_RESULTS) / len(rf_RESULTS)))
    print('KNN Accuracy = ' + str(sum(knn_RESULTS) / len(knn_RESULTS)))
    print('ENSEMBLE Accuracy = ' + str(sum(ensemble_RESULTS) / len(ensemble_RESULTS)))

    return {"RF_Accuracy" : str(sum(rf_RESULTS) / len(rf_RESULTS)) , "KNN_Accuracy": str(sum(knn_RESULTS) / len(knn_RESULTS)) , "ENSEMBLE Accuracy":str(sum(ensemble_RESULTS) / len(ensemble_RESULTS))}








