import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics


if __name__ == '__main__':
    # load training data
    # data = np.loadtxt('../Data/FlightClassificationClNoHeaders.csv', delimiter=',')
    # data = np.loadtxt('DatasetManipulation/FCNoOutliersNoHeaders.csv', delimiter=',')
    data = np.loadtxt('DatasetManipulation/FDnoHeaders.csv', delimiter=',')

    # X = data[:,[0,1,2,3,4]]        # Load 5 first columns from the data
    X = data[:,[1,5,6,7,8]]
    y = data[:,9]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # Load Module - http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    NN = MLPClassifier(hidden_layer_sizes=(4,4), activation='tanh', max_iter=100)
    kf = KFold(n_splits=10, shuffle=True)

    ACC = 0
    PRE = 0
    TPR = 0

    # 10-folds cross-validation
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index, '\n')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        NN.fit(X_train, y_train)
        prediction = NN.predict(X_test)

        CM = metrics.confusion_matrix(y_test, prediction)
        # Guide
        # TN = CM[0][0]
        # FN = CM[1][0]
        # TP = CM[1][1]
        # FP = CM[0][1]

        # TPR = TP / (TP+FN)
        TPR += CM[1][1] / (CM[1][1]+CM[1][0])    # Ability of the classifier not to label as positive a sample that is negative
        PRE += CM[1][1] / (CM[1][1]+CM[0][1])
        ACC += NN.score(X_test, y_test)  

    # NN.fit(X_train, y_train)

    # prediction = NN.predict(X_test)

    # True Positive Rate
    # CM = metrics.confusion_matrix(y_test, prediction)

    # TPR = TP / (TP+FP)
    # TPR = CM[1][1] / (CM[1][1]+CM[1][0])    # Ability of the classifier not to label as positive a sample that is negative

    ACC = ACC / kf.get_n_splits(X)
    PRE = PRE / kf.get_n_splits(X)
    TPR = TPR / kf.get_n_splits(X)

    print('Accuracy on the training subset: {:.3f}' .format(ACC))
    # Accuracy on training data: 0.843

    print('Precision: ', PRE)

    print('True Positive Rate: ', TPR)
    # TPR: 0.713
