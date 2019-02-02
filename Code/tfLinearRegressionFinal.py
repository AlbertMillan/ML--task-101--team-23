import tensorflow as tf
import pandas as pd
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
import numpy as np
import math
from sklearn.model_selection import KFold

#def easy_input_function(df, label, num_epochs, shuffle, batch_size):
#    ds = tf.data.Dataset.from_tensor_slices((dict(df),label))
#
#    if shuffle:
#        ds = ds.shuffle(10000)
#
#    ds = ds.batch(batch_size).repeat(num_epochs)
#
#    return ds


flights = pd.read_csv("/Users/jamessherlock/Documents/MachineLearning/GroupAssignment/ML1819--task-101--team-23/Code/DatasetManipulation/FCNoOutliers.csv")
#record_defaults = [tf.float32] * 5
#X_dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[1,2,3,4,5])
#Y_dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[6])

#print(X_dataset.output_shapes)
#print(Y_dataset.output_shapes)
print('flights: ', flights.columns)
x_data = flights.drop(['ARR_DELAY', 'DEP_TIME', 'ARR_TIME', 'DISTANCE', 'ARR_DELAY_BIN'], axis=1)
print('x data updated: ', x_data)

#x_data = x_data.values
#pre_x_data.reset_index(drop=True, inplace=True)
y_labels = flights['ARR_DELAY']
#y_labels = y_labels.values
#Hot encoding the Carrier data
#x_data = pd.concat([pre_x_data, finalEncoding], axis=1)
print('x data: ', x_data)
#X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.3, random_state=0)

#TRAIN_DATASET_SIZE = 4211244
#TEST_DATASET_SIZE = 701874

#X_TRAIN_DATASET_SIZE = 2947872
#X_TEST_DATASET_SIZE = 1263372
#
#Y_TRAIN_DATASET_SIZE = 491312
#Y_TEST_DATASET_SIZE = 210562

#train_size = int(0.7 * TRAIN_DATASET_SIZE)
#test_size = int(0.3 * DATASET_SIZE)

#x_train_size = int(X_TRAIN_DATASET_SIZE)
#x_test_size = int(X_TEST_DATASET_SIZE)
#y_train_size = int(Y_TRAIN_DATASET_SIZE)
#y_test_size = int(Y_TEST_DATASET_SIZE)
#
#X_train = X_dataset.take(x_train_size)
#X_test = X_dataset.skip(x_train_size)
#
#Y_train = Y_dataset.take(y_train_size)
#X_test = Y_dataset.skip(y_train_size)

#print(train_dataset.output_shapes)
#print(test_dataset.output_shapes)

#depTime = tf.feature_column.numeric_column("DEP_TIME")
depDelay = tf.feature_column.numeric_column("DEP_DELAY")
carrierDelay = tf.feature_column.numeric_column("CARRIER_DELAY")
weatherDelay = tf.feature_column.numeric_column("WEATHER_DELAY")
nasDelay = tf.feature_column.numeric_column("NAS_DELAY")
lateAircraftDelay = tf.feature_column.numeric_column("LATE_AIRCRAFT_DELAY")

mae = 0
rmse = 0
cod = 0

print('test: ', x_data.iloc[[0]])

#arrTime = tf.feature_column.numeric_column("ARR_TIME")
#arrDelay = tf.feature_column.categorical_column_with_vocabulary_list("ARR_DELAY_BIN", [0,1])
#airTime = tf.feature_column.numeric_column("AIR_TIME")
#distance = tf.feature_column.numeric_column("DISTANCE")
#
feat_cols = [depDelay, carrierDelay, weatherDelay, nasDelay, lateAircraftDelay]
regressor = tf.estimator.LinearRegressor(feature_columns=feat_cols)
#feature_column = [tf.feature_column.numeric_column(key=’features’,shape=(784,))]
#
k_fold = KFold(n_splits=10, shuffle=True)
for train_index, test_index in k_fold.split(x_data):
    # print("TRAIN:", train_index, "TEST:", test_index, '\n')
    print('train_index: ', train_index)
    print('test_index: ', train_index)
    X_train, X_test = x_data.iloc[train_index], x_data.iloc[test_index]
    y_train, y_test = y_labels.iloc[train_index], y_labels.iloc[test_index]

    train_func = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size=100, num_epochs=1, shuffle=False)
#train_func = easy_input_function(X_train, Y_train, num_epochs=10, shuffle=False, batch_size=100)
#
    regressor.train(input_fn=train_func)
#
#test_func = easy_input_function(X_test, X_test, num_epochs=1, shuffle=False, batch_size=100)
    test_func = tf.estimator.inputs.pandas_input_fn(X_test, y_test, batch_size=100, num_epochs=1, shuffle=False)
    result = regressor.evaluate(input_fn=test_func)
    preds = regressor.predict(input_fn=test_func)
    predictions = np.array([item['predictions'][0] for item in preds])

#    for key,value in sorted(result.items()):
#        print('%s: %s' % (key, value))
    mae += skm.mean_absolute_error(y_test.values,predictions)
    rmse += math.sqrt(skm.mean_squared_error(y_test.values,predictions))
    cod += skm.r2_score(y_test.values,predictions)

mae = mae/k_fold.get_n_splits(x_data)
rmse = rmse/k_fold.get_n_splits(x_data)
cod = cod/k_fold.get_n_splits(x_data)
#print('y_labels: ', type(y_test))
#print('preds: ', type(list(preds)))

print('mae: ', mae)
print('rmse: ', rmse)
print('coefficient of determination: ', cod)
#rmse = tf.metrics.root_mean_squared_error(labels=y_test, predictions=preds)
#
print(result)








#Working as is
#import tensorflow as tf
#import pandas as pd
#import sklearn.metrics as skm
#from sklearn.model_selection import train_test_split
#
##def easy_input_function(df, label, num_epochs, shuffle, batch_size):
##    ds = tf.data.Dataset.from_tensor_slices((dict(df),label))
##
##    if shuffle:
##        ds = ds.shuffle(10000)
##
##    ds = ds.batch(batch_size).repeat(num_epochs)
##
##    return ds
#
#
#flights = pd.read_csv("/Users/jamessherlock/Documents/MachineLearning/GroupAssignment/ML1819--task-101--team-23/Data/FlightClassificationCleaned.csv")
#print(type(flights))
##record_defaults = [tf.float32] * 5
##X_dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[1,2,3,4,5])
##Y_dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[6])
#
##print(X_dataset.output_shapes)
##print(Y_dataset.output_shapes)
#x_data = flights.drop(['ARR_DELAY_BIN', 'ARR_DELAY'], axis=1)
#y_labels = flights['ARR_DELAY']
#print(x_data)
#print(y_labels)
#X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.3, random_state=0)
#print('X_train type: ', type(X_train))
#
##TRAIN_DATASET_SIZE = 4211244
##TEST_DATASET_SIZE = 701874
#
##X_TRAIN_DATASET_SIZE = 2947872
##X_TEST_DATASET_SIZE = 1263372
##
##Y_TRAIN_DATASET_SIZE = 491312
##Y_TEST_DATASET_SIZE = 210562
#
##train_size = int(0.7 * TRAIN_DATASET_SIZE)
##test_size = int(0.3 * DATASET_SIZE)
#
##x_train_size = int(X_TRAIN_DATASET_SIZE)
##x_test_size = int(X_TEST_DATASET_SIZE)
##y_train_size = int(Y_TRAIN_DATASET_SIZE)
##y_test_size = int(Y_TEST_DATASET_SIZE)
##
##X_train = X_dataset.take(x_train_size)
##X_test = X_dataset.skip(x_train_size)
##
##Y_train = Y_dataset.take(y_train_size)
##X_test = Y_dataset.skip(y_train_size)
#
##print(train_dataset.output_shapes)
##print(test_dataset.output_shapes)
#
#depTime = tf.feature_column.numeric_column("DEP_TIME")
#depDelay = tf.feature_column.numeric_column("DEP_DELAY")
#arrTime = tf.feature_column.numeric_column("ARR_TIME")
#arrDelay = tf.feature_column.categorical_column_with_vocabulary_list("ARR_DELAY_BIN", [0,1])
#airTime = tf.feature_column.numeric_column("AIR_TIME")
#distance = tf.feature_column.numeric_column("DISTANCE")
##
#feat_cols = [depTime, depDelay, arrTime, airTime, distance]
##
#train_func = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size=100, num_epochs=10, shuffle=False)
##train_func = easy_input_function(X_train, Y_train, num_epochs=10, shuffle=False, batch_size=100)
##
#regressor = tf.estimator.LinearRegressor(feature_columns=feat_cols)
#regressor.train(input_fn=train_func)
##
##test_func = easy_input_function(X_test, X_test, num_epochs=1, shuffle=False, batch_size=100)
#test_func = tf.estimator.inputs.pandas_input_fn(X_test, y_test, batch_size=100, num_epochs=1, shuffle=False)
#result = regressor.evaluate(input_fn=test_func)
#preds = regressor.predict(input_fn=test_func)
#
#print('y_labels: ', type(y_test))
#print('preds: ', type(list(preds)))
#
##rmse = tf.metrics.root_mean_squared_error(labels=y_test, predictions=preds)
##
#print(result)


#Works with one hot
#import tensorflow as tf
#import pandas as pd
#import sklearn.metrics as skm
#from sklearn.model_selection import train_test_split
#
##def easy_input_function(df, label, num_epochs, shuffle, batch_size):
##    ds = tf.data.Dataset.from_tensor_slices((dict(df),label))
##
##    if shuffle:
##        ds = ds.shuffle(10000)
##
##    ds = ds.batch(batch_size).repeat(num_epochs)
##
##    return ds
#
#
#flights = pd.read_csv("/Users/jamessherlock/Documents/MachineLearning/GroupAssignment/ML1819--task-101--team-23/Data/Flight-Delays-July-2018-Cleaned.csv")
##record_defaults = [tf.float32] * 5
##X_dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[1,2,3,4,5])
##Y_dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[6])
#
##print(X_dataset.output_shapes)
##print(Y_dataset.output_shapes)
#pre_x_data = flights.drop(['ARR_DELAY', 'ORIGIN', 'DEST'], axis=1)
##pre_x_data.reset_index(drop=True, inplace=True)
#y_labels = flights['ARR_DELAY']
##Hot encoding the Carrier data
#df_oh = pd.get_dummies(pre_x_data)
#finalEncoding = df_oh.drop('MKT_CARRIER_WN', axis=1)
#x_data = finalEncoding
##x_data = pd.concat([pre_x_data, finalEncoding], axis=1)
#print('x data: ', x_data)
#X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.3, random_state=0)
#
##TRAIN_DATASET_SIZE = 4211244
##TEST_DATASET_SIZE = 701874
#
##X_TRAIN_DATASET_SIZE = 2947872
##X_TEST_DATASET_SIZE = 1263372
##
##Y_TRAIN_DATASET_SIZE = 491312
##Y_TEST_DATASET_SIZE = 210562
#
##train_size = int(0.7 * TRAIN_DATASET_SIZE)
##test_size = int(0.3 * DATASET_SIZE)
#
##x_train_size = int(X_TRAIN_DATASET_SIZE)
##x_test_size = int(X_TEST_DATASET_SIZE)
##y_train_size = int(Y_TRAIN_DATASET_SIZE)
##y_test_size = int(Y_TEST_DATASET_SIZE)
##
##X_train = X_dataset.take(x_train_size)
##X_test = X_dataset.skip(x_train_size)
##
##Y_train = Y_dataset.take(y_train_size)
##X_test = Y_dataset.skip(y_train_size)
#
##print(train_dataset.output_shapes)
##print(test_dataset.output_shapes)
#
#depTime = tf.feature_column.numeric_column("DEP_TIME")
#depDelay = tf.feature_column.numeric_column("DEP_DELAY")
#arrTime = tf.feature_column.numeric_column("ARR_TIME")
#arrDelay = tf.feature_column.categorical_column_with_vocabulary_list("ARR_DELAY_BIN", [0,1])
#airTime = tf.feature_column.numeric_column("AIR_TIME")
#distance = tf.feature_column.numeric_column("DISTANCE")
##
#feat_cols = [depTime, depDelay, arrTime, airTime, distance]
##
#train_func = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size=100, num_epochs=10, shuffle=False)
##train_func = easy_input_function(X_train, Y_train, num_epochs=10, shuffle=False, batch_size=100)
##
#regressor = tf.estimator.LinearRegressor(feature_columns=feat_cols)
#regressor.train(input_fn=train_func)
##
##test_func = easy_input_function(X_test, X_test, num_epochs=1, shuffle=False, batch_size=100)
#test_func = tf.estimator.inputs.pandas_input_fn(X_test, y_test, batch_size=100, num_epochs=1, shuffle=False)
#result = regressor.evaluate(input_fn=test_func)
#preds = regressor.predict(input_fn=test_func)
#
#print('y_labels: ', type(y_test))
#print('preds: ', type(list(preds)))
#
##rmse = tf.metrics.root_mean_squared_error(labels=y_test, predictions=preds)
##
#print(result)
