from feature_selector import FeatureSelector
import matplotlib.pyplot as plt
import pandas as pd 

if __name__ == '__main__':
    model = pd.read_csv('../Data/FlightClassificationCleaned.csv')
    target = model['ARR_DELAY']
    model.head()
    model = model.drop(columns=['ARR_DELAY', 'ARR_DELAY_BIN'])


    fs = FeatureSelector(data=model, labels=target)

    fs.identify_collinear(correlation_threshold=0.9)

    correlated_features = fs.ops['collinear']
    print(correlated_features[:5])
    fs.record_collinear.head()
    print(fs.plot_collinear())
    # fs.record_collinear.head()
    # graph.savefig('VarCorrelation.png')