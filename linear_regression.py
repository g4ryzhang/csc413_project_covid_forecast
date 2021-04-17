import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_data(lag_vars, lag_time):
    # Load data from the source csv file and transform into lag observations
    # of the daily confirmed cases (single variable). 
    # Return (X_train, y_train, X_val, y_val) 

    df = pd.read_csv('covid19_sg_clean.csv', sep=',', header=None)
    data = df.values[1:, 1]
    data = data[:int(np.round(data.shape[0]*0.8))]       # only use 80% data for training and validation
    data = data.astype(np.int32)

    num_examples = len(data) - lag_vars - lag_time + 1
    X = np.zeros((num_examples, 1))

    # Append examples
    for i in range(lag_vars):
        X = np.hstack((X, np.reshape(data[i:i+num_examples], (num_examples, 1))))

    # Targets
    # y = np.reshape(data[lag_vars+lag_time-1:], (num_examples, 1))
    y = data[lag_vars+lag_time-1:]

    X = np.delete(X, 0, 1)        # remove placeholder zeros
    split = int(np.round(num_examples*0.7))     # split the data 7:3 for training and test without shuffling

    return X[:split], y[:split], X[split:], y[split:] 


if __name__ == '__main__':
    
    experiments = []
    for i in range(1, 21):
        for j in range(1, 11):

            # hyperparameters
            lag_vars = i
            lag_time = j

            X_train, y_train, X_val, y_val = load_data(lag_vars, lag_time)
            reg = LinearRegression().fit(X_train, y_train)
            train_score = reg.score(X_train, y_train)
            val_score = reg.score(X_val, y_val)
            RMSE = mean_squared_error(reg.predict(X_val), y_val, squared=False)

            print(f'lag_vars = {lag_vars} \nlag_time = {lag_time}')
            print(f'Training set score: {train_score}')
            print(f'Test set score: {val_score}')
            print(f'RMSE: {RMSE}\n')

            experiments.append((lag_vars, lag_time, train_score, val_score, RMSE))

    s = sorted(experiments, key=lambda t: t[3])
    r = sorted(experiments, key=lambda t: t[4])
    print(f'The best model based on val_score: {s[-1]}')
    print(f'The best model based on RMSE: {r[0]}')
    

