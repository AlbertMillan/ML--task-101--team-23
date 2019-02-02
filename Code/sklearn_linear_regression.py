import numpy as np
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from math import sqrt

import pandas as pd

# Load Module - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# Uses Ordinary Least Squares instead of Gradient Descent
# https://stackoverflow.com/questions/34469237/linear-regression-and-gradient-descent-in-scikit-learn-pandas
if __name__ == '__main__':
    # load training data
    # data = np.loadtxt('../Data/FlightClassificationClNoHeaders.csv', delimiter=',')
    # data = np.loadtxt('DatasetManipulation/FCNoOutliersNoHeaders.csv', delimiter=',')
    data = np.loadtxt('DatasetManipulation/FDnoHeaders.csv', delimiter=',')
    # data = np.loadtxt('DatasetManipulation/FDnoHeaders.csv', delimiter=',', usecols=(5))
    print(data)
    

    # Only take relevant features
    X = data[:,[1,5,6,7,8]]        # Load 5 first columns from the data
    y = data[:,3]

    kf = KFold(n_splits=10, shuffle=True)
    reg = LinearRegression()

    ACC = 0
    MAE = 0
    MSE = 0
    RMSE = 0

    # 10-folds cross-validation
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index, '\n')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        reg.fit(X_train, y_train)
        predictions = reg.predict(X_test)

        ACC += reg.score(X_test, y_test)
        MAE += np.mean(abs(predictions-y_test))
        MSE += mean_squared_error(y_test, predictions)
        RMSE += sqrt(mean_squared_error(y_test, predictions))


    ACC = ACC / kf.get_n_splits(X)
    MAE = MAE / kf.get_n_splits(X)
    MSE = MSE / kf.get_n_splits(X)
    RMSE = RMSE / kf.get_n_splits(X)

    print('Accuracy: {:.3f}' .format(ACC) )
    print('MAE:', MAE)
    print('MSE:', MSE)
    print('RMSE: ', RMSE)

    # MSE
    # print('Mean Square Error: ', np.mean((predictions-y_test)**2))
    #print('Mean Square Erros: ', mean_squared_error(y_test, predictions))
    
    # MAE
    # print('Mean Absolute Error: ', np.mean(abs(predictions-y_test)))

    # RMSE
    # print('Root Mean Squared Error: ', sqrt(mean_squared_error(y_test, predictions)))



