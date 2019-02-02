import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches   # To create your own legend.

if __name__ == '__main__':
    # load training data
    data = np.loadtxt('../DatasetManipulation/FCNoOutliersNoHeaders.csv', delimiter=',')
    res = data[:,-1]
    # data = pd.read_csv('../DatasetManipulation/FCNoOutliers.csv')
    # res = data['ARR_DELAY'].hist(bins=10, range=[-50, 50])

    n_bins = 97

    fig, ax = plt.subplots(figsize=(12, 8))
    N, bins, patches = ax.hist(res, bins=n_bins)

    # Set the range and spacing for the labels in the x- and y-axes.
    ax.set_xticks(np.arange(-50, 51, 10))
    ax.set_yticks(np.arange(0, 30000+1, 3000))

    # Set the background dashed lines.
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')

    # Set labels.
    ax.set_xlabel('Arrival Delay Time (minutes)', fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)

    # Fully customize the variables in the legend.
    blue_patch = mpatches.Patch(color='b', label='On-time')
    red_patch = mpatches.Patch(color='red', label='Delayed')
    ax.legend(handles=[blue_patch, red_patch])

    # Set red color for those delayed flights
    for i in range(50,n_bins):
        patches[i].set_facecolor('r')

    fig.savefig('histo.png')

    