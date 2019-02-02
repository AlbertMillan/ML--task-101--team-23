import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import median

columnID = 3

def getQuartiles(data):
    mid = int(len(data) / 2)
    # print(mid)
    if(len(data) % 2 == 0):
        # even
        lowerQ = median(data[:mid, columnID])
        upperQ = median(data[mid:, columnID])
    else:
        # odd
        lowerQ = median(data[:mid, columnID])
        upperQ = median(data[mid+1:, columnID])

    return lowerQ, upperQ


if __name__ == '__main__':
    # load training data
    data = np.loadtxt('FDnoHeaders.csv', delimiter=',')
    # data = pd.read_csv('../Data/FlightClassificationCleaned.csv')
    # res = data['DEP_DELAY'].hist(bins=200, range=[-48, 2500])

    # data.view('i8,i8,i8,i8,i8,i8,i8').sort(axis=0, order=['f1'])
    data = data[data[:,columnID].argsort()]

    myMedian = median(data[:,columnID])

    q1, q3 = getQuartiles(data)

    IQR = q3 - q1

    lowerBoundary = q1 - 1.5 * IQR
    upperBoundary = q3 + 1.5 * IQR

    data = data[np.logical_and(data[:,columnID] >= lowerBoundary, data[:,columnID] <= upperBoundary)]
    print(len(data[:,0]))

    df = pd.DataFrame(data)
    df.to_csv("test.csv")

    # res = data['DEP_DELAY'].hist(bins=200, range=[-48, 2500])
    
    # print(data[0, -1])
    print(myMedian)
    print(q1)
    print(q3)
    print(lowerBoundary)
    print(upperBoundary)