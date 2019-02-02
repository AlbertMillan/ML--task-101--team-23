# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to 
import pandas as pd
import seaborn as sns

# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
def azureml_main(dataframe1 = None, dataframe2 = None):

    sns.set(style="whitegrid")
    X = ['TensorFlow', 'Scikit-Learn', 'Azure ML']
    
    axc = get_cod(X)
    axc.figure.savefig("cod.png")
    
    axm = get_mae(X)
    axm.figure.savefig("mae.png")
    
    axr = get_rmse(X)
    axr.figure.savefig("rmse.png")
    
    
    return dataframe1,

def get_mae(X):
    Y = [7.9, 7.58, 7.54]
    d = {'Platform': X, 'Mean Absolute Error': Y}
    df = pd.DataFrame(d)
    
    ax = sns.barplot(x="Platform", y="Mean Absolute Error", data = df)
    
    return ax
    
def get_rmse(X):
    Y = [10.28, 9.89, 9.86]
    d = {'Platform': X, 'Root Mean Square Error': Y}
    df = pd.DataFrame(d)
    
    ax = sns.barplot(x="Platform", y="Root Mean Square Error", data = df)
    
    return ax
    
def get_cod(X):
    Y = [0.595, 0.963, 0.961]
    d = {'Platform': X, 'Coefficient of Determination': Y}
    df = pd.DataFrame(d)
    
    ax = sns.barplot(x="Platform", y="Coefficient of Determination", data = df)
    
    return ax