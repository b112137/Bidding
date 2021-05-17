import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import keras

#You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()


def output(path, data):
    
    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return


if __name__ == "__main__":
    args = config()
    
    consumption_input = pd.read_csv(args.consumption)
    scaler = joblib.load("scaler_consumption.save")
    model = keras.models.load_model('model_consumption.h5')
    data_input = consumption_input['consumption']
    test_X = data_input.values
    
    temp = scaler.transform([np.concatenate( ( test_X, list(range(0,24)) ) )])
    test_X = np.array([[  temp[0][:-24] ]])

    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    inv_yhat = np.concatenate((test_X[:, 0:], yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,-24:]
    consumption_prediction = inv_yhat    
    
    
    
    generation_input = pd.read_csv(args.generation)
    scaler = joblib.load("scaler_generation.save")
    model = keras.models.load_model('model_generation.h5')
    data_input = generation_input['generation']
    test_X = data_input.values
    
    temp = scaler.transform([np.concatenate( ( test_X, list(range(0,24)) ) )])
    test_X = np.array([[  temp[0][:-24] ]])

    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    inv_yhat = np.concatenate((test_X[:, 0:], yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,-24:]
    generation_prediction = inv_yhat    
    
    

    dateString = consumption_input.iloc[-1]['time']
    dateFormatter = "%Y-%m-%d %H:%M:%S"
    DT = datetime.strptime(dateString, dateFormatter)
    prediction_date = DT+timedelta(days=1)
    prediction_data_temp = prediction_date.replace(hour = 0)
    
    data = []
    
    for i in range(24):
        overflow = generation_prediction[0][i] - consumption_prediction[0][i]
        if overflow > 0:
            data.append([str(prediction_data_temp), "sell", 2.49, str(overflow)])
        elif overflow < 0:
            data.append([str(prediction_data_temp), "buy", 2.47, str(-overflow)])
        prediction_data_temp = prediction_data_temp + timedelta(hours=1)
        
    output(args.output, data)
