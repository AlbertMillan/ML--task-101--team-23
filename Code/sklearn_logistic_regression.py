 import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics

import matplotlib.pyplot as plt

def plotData( X, y):
    # plots the data points with o for the positive examples and x for 
    # the negative examples. Output is saved to file graph.png

    fig, ax = plt.subplots(figsize=(12, 8))     # figsize - size of the image

    onTime = y<=0
    delayed = y>0

    ax.scatter( X[positive, 0], X[positive, 1], c='b', marker='o', label='On-time')
    ax.scatter( X[negative, 0], X[negative, 1], c='r', marker='x', label='Delayed')

    ax.set_xlabel('Departure Delay Time (Minutes)')
    ax.set_ylabel('Departure Time')
    fig.savefig('graph.png')


 # Load Module - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
if __name__ == '__main__':
    # load training data
    # data = np.loadtxt('../Data/FlightClassificationClNoHeaders.csv', delimiter=',')
    # data = np.loadtxt('DatasetManipulation/FCNoOutliersNoHeaders.csv', delimiter=',')
    data = np.loadtxt('DatasetManipulation/FDnoHeaders.csv', delimiter=',')

    # X = data[:,[0,1,2,3,4]]        # Load 5 first columns from the data
    X = data[:,[1,5,6,7,8]]
    y = data[:,9]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


    kf = KFold(n_splits=10, shuffle=True)
    log_reg = LogisticRegression(C=0.001, tol=1e-07, solver='liblinear')

    ACC = 0
    PRE = 0
    TPR = 0
    
    # 10-folds cross-validation
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index, '\n')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        log_reg.fit(X_train, y_train)
        prediction = log_reg.predict(X_test)

        CM = metrics.confusion_matrix(y_test, prediction)
        # Guide
        # TN = CM[0][0]
        # FN = CM[1][0]
        # TP = CM[1][1]
        # FP = CM[0][1]

        # TPR = TP / (TP+FN)
        TPR += CM[1][1] / (CM[1][1]+CM[1][0])    # Ability of the classifier not to label as positive a sample that is negative
        PRE += CM[1][1] / (CM[1][1]+CM[0][1])
        ACC += log_reg.score(X_test, y_test)  

    ACC = ACC / kf.get_n_splits(X)
    PRE = PRE / kf.get_n_splits(X)
    TPR = TPR / kf.get_n_splits(X)

    # print('Accuracy on the training subset: {:.3f}' .format(log_reg.score(X, y)))
    print('ACC: {:.3f}' .format(ACC))
    print('PRE: ', PRE)
    print('TPR: ', TPR)
