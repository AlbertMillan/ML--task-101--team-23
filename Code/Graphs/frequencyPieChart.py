import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # load training data
    data = np.loadtxt('../Data/FlightClassificationClNoHeaders.csv', delimiter=',')
    # data = pd.read_csv('../Data/FlightClassificationCleaned.csv')
    # res = data['DEP_DELAY'].hist(bins=200, range=[-48, 2500])

    X = data[:,1]
    onTime1  = np.where( X<=-10 )
    onTime2  = np.where( (X>-10) & (X<=-5))
    onTime3  = np.where( (X>-5) & (X<=0))
    delayed1 = np.where( (X>0) & (X<=5))
    delayed2 = np.where( (X>0) & (X<=10))
    delayed3 = np.where( X>10 )


    onTime1  = ( data[ onTime1, 0].size /  X.size ) * 100
    onTime2  = ( data[ onTime2, 0].size /  X.size ) * 100
    onTime3  = ( data[ onTime3, 0].size /  X.size ) * 100
    delayed1 = ( data[ delayed1, 0].size /  X.size ) * 100
    delayed2 = ( data[ delayed2, 0].size /  X.size ) * 100
    delayed3 = ( data[ delayed3, 0].size /  X.size ) * 100
    
    print(onTime1)

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))

    size=0.3

    vals = np.array([[onTime1, onTime2, onTime3], [delayed1,delayed2,delayed3]])
    # , ['0-5', '5-10', '>10']

    valsnorm = vals/np.sum(vals)*2*np.pi
    #obtain the ordinates of the bar edges
    valsleft = np.cumsum(np.append(0, valsnorm.flatten()[:-1])).reshape(vals.shape)

    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap(np.arange(3)*4)
    inner_colors = cmap(np.array([1, 2, 3, 5, 6, 7]))

    ax.bar(x=valsleft[:, 0],
       width=valsnorm.sum(axis=1), bottom=1-size, height=size,
       color=outer_colors, edgecolor='w', linewidth=1, align="edge")

    ax.bar(x=valsleft.flatten(), 
       width=valsnorm.flatten(), bottom=1-2*size, height=size,
       color=inner_colors, edgecolor='w', linewidth=1, align="edge")

    ax.set(title="Frequency of Delays in Minutes")
    ax.set_axis_off()
    fig.savefig("PieChart3")
    

    # print(early)
    # print(punctual)
    # print(delayed)

    # fig = res.get_figure()
    # fig.savefig('graph.png')

    print("Hello world!")



    # plotData(X)
