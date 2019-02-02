# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to 
import pandas as pd
import seaborn as sns

# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame

# Data Format:
# Algorithm			        Platform		  Metric		  Value
# Logistic Regression	  Tensorflow		Accuracy	
# Logistic Regression	  Scikit-Learn	Accuracy
# Logistic Regression	  Azure ML		  Accuracy
# Logistic Regression	  Tensorflow		Precision
# Logistic Regression	  Scikit-Learn	Precision
# Logistic Regression	  Azure ML		  Precision
# Logistic Regression	  Tensorflow		Recall
# Logistic Regression	  Scikit-Learn	Recall
# Logistic Regression	  Azure ML		  Recall
def azureml_main(dataframe1 = None, dataframe2 = None):

    sns.set(style="whitegrid")
    Platforms = ['TensorFlow', 'Scikit-Learn', 'Azure ML','TensorFlow', 'Scikit-Learn', 'Azure ML','TensorFlow', 'Scikit-Learn', 'Azure ML']
    Metrics = ['Accuracy', 'Accuracy', 'Accuracy', 'Precision', 'Precision', 'Precision', 'Recall', 'Recall', 'Recall']
    # Values = [0.91,0.93,0.92,0.95,0.93,0.92,0.96,0.95,0.92]
    # d = {'Platform': Platforms, 'Metric': Metrics, 'Percentage (%)': Values}    
    # df = pd.DataFrame(d)
    
    # ax = sns.barplot(x="Metric", y="Percentage (%)", hue="Platform", data = df)
   
    
    # ax.figure.savefig("graph.png")

    # get_logistic_regression(Platforms, Metrics)
    get_dnn(Platforms, Metrics)
    
    return dataframe1,

def get_logistic_regression(Platforms, Metrics):
    Values = [0.812,0.86,0.848,0.695,0.93,0.983,0.729,0.701,0.691]
    dl = {'Platform': Platforms, 'Metric': Metrics, 'Value': Values}    
    dfl = pd.DataFrame(dl)
    
    axl = sns.barplot(x="Metric", y="Value", hue="Platform", data = dfl)
    axl = fix_legend(axl)
    axl.figure.savefig("LogisticRegression-Metrics.png")


def get_dnn(Platforms, Metrics):
  Values = [0.792,0.86,0.864,0.764,0.92,0.939,0.78,0.713,0.697]
  dd = {'Platform': Platforms, 'Metric': Metrics, 'Value': Values}    
  dfd = pd.DataFrame(dd)
  
  axd = sns.barplot(x="Metric", y="Value", hue="Platform", data = dfd)
  axd = fix_legend(axd)
  axd.figure.savefig("DNN-Metrics.png")


def fix_legend(ax):
  # Put the legend out of the figure
  ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  ax.figure.tight_layout(pad=12)
  
  return ax