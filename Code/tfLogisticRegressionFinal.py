import tensorflow as tf
import pandas as pd
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np

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
x_data = flights.drop(['ARR_DELAY', 'DEP_TIME', 'ARR_TIME', 'DISTANCE', 'ARR_DELAY_BIN'], axis=1)
y_labels = flights['ARR_DELAY_BIN']
print(x_data)
print(y_labels)
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
#depDelay = tf.feature_column.numeric_column("DEP_DELAY")
#arrTime = tf.feature_column.numeric_column("ARR_TIME")
#arrDelay = tf.feature_column.categorical_column_with_vocabulary_list("ARR_DELAY_BIN", [0,1])
#airTime = tf.feature_column.numeric_column("AIR_TIME")
#distance = tf.feature_column.numeric_column("DISTANCE")
depDelay = tf.feature_column.numeric_column("DEP_DELAY")
carrierDelay = tf.feature_column.numeric_column("CARRIER_DELAY")
weatherDelay = tf.feature_column.numeric_column("WEATHER_DELAY")
nasDelay = tf.feature_column.numeric_column("NAS_DELAY")
lateAircraftDelay = tf.feature_column.numeric_column("LATE_AIRCRAFT_DELAY")
#
feat_cols = [depDelay, carrierDelay, weatherDelay, nasDelay, lateAircraftDelay]
classifier = tf.estimator.LinearClassifier(feature_columns=feat_cols)
accuracy = 0
recall = 0
precision = 0
acc = 0
#

k_fold = KFold(n_splits=10, shuffle=True)
for train_index, test_index in k_fold.split(x_data, y_labels):
    print('train_index: ', train_index)
    print('test_index: ', train_index)
    X_train, X_test = x_data.iloc[train_index], x_data.iloc[test_index]
    y_train, y_test = y_labels.iloc[train_index], y_labels.iloc[test_index]

    train_func = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size=100, num_epochs=1, shuffle=False)
    #train_func = easy_input_function(X_train, Y_train, num_epochs=10, shuffle=False, batch_size=100)
#
    classifier.train(input_fn=train_func)
#
#test_func = easy_input_function(X_test, X_test, num_epochs=1, shuffle=False, batch_size=100)
    test_func = tf.estimator.inputs.pandas_input_fn(X_test, y_test, batch_size=100, num_epochs=1, shuffle=False)
    result = classifier.evaluate(input_fn=test_func)
    preds = classifier.predict(input_fn=test_func)
    predictions = []
    for pred in preds:
        predictions.append(pred['class_ids'][0])

    #print('len(predictions): ', len(predictions))
    #print('len(y_test): ', len(y_test))
    correctPreds = 0
    incorrectPreds = 0
    length = len(predictions)
    #print('y_test: ', y_test.values)
    #print('preds: ', predictions)
    for j in range(length):
        if(y_test.values[j] == predictions[j]):
            correctPreds += 1
        #print('indices: ', j)
        else:
            incorrectPreds += 1

    print('correctPreds: ', correctPreds)
    print('incorrectPreds: ', incorrectPreds)
    accuracy += result['accuracy']
    print('Accuracy: ', accuracy)
    recall += result['recall']
    precision += result['precision']

accuracy = accuracy/k_fold.get_n_splits(x_data)
recall = recall/k_fold.get_n_splits(x_data)
precision = precision/k_fold.get_n_splits(x_data)

print('Accuracy: ', accuracy)
print('Recall: ', recall)
print('Precision: ', precision)
#
print(result)







#def input_fn(data_file, num_epochs, shuffle, batch_size):
#    #Generate an input function for the Estimator
#    assert tf.gfile.Exists(data_file), ('%s not found. Please make sure you have run census_dataset.py and set the --data_dir argument to the correct path.' % data_file)
#
#    def parse_csv(value):
#        tf.logging.info('Parsing {}'.format(data_file))
#        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
#        features = dict(zip(_CSV_COLUMNS, columns))
#        labels = features.pop('income_bracket')
#        classes = tf.equal(labels, '>50K')  # binary classification
#        return features, classes
#
#    # Extract lines from input files using the Dataset API.
#    dataset = tf.data.TextLineDataset(data_file)
#
#    if shuffle:
#        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])
#
#    dataset = dataset.map(parse_csv, num_parallel_calls=5)
#
#    # We call repeat after shuffling, rather than before, to prevent separate
#    # epochs from blending together.
#    dataset = dataset.repeat(num_epochs)
#    dataset = dataset.batch(batch_size)
#    return dataset






























## Creates a dataset that reads all of the records from two CSV files, each with eight float columns
#filenames = ["/ML1819--task-101--team-23/Data/FlightClassificationCleaned.csv"]
#record_defaults = [tf.float32] * 5   # Eight required float columns
#dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[1,2,3,4,5])
#
#print(dataset.output_types)
#print(dataset.output_shapes)
#
#iterator = dataset.make_one_shot_iterator()
#next_element = iterator.get_next()
#
#
#sess = tf.Session()
#
#sess.run(iterator.initializer)
#while True:
#    try:
#        print(sess.run(result))
#    except tf.errors.OutOfRangeError:
#        break







#filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
#dataset = tf.data.TFRecordDataset(filenames)
#dataset = dataset.map(...)
#dataset = dataset.batch(32)
#iterator = dataset.make_initializable_iterator()
#next_element = iterator.get_next()
#
## Compute for 100 epochs.
#for _ in range(100):
#    sess.run(iterator.initializer)
#    while True:
#    try:
#        sess.run(next_element)
#    except tf.errors.OutOfRangeError:
#        break
#
## [Perform end-of-epoch calculations here.]


## TensorFlow and tf.keras
#import tensorflow as tf
#from tensorflow import keras
#import sklearn.metrics as skm
#from sklearn.model_selection import train_test_split
#
## Helper libraries
#import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
#from matplotlib import pyplot as plt
#import pandas as pd
#
#flights = pd.read_csv("/Users/jamessherlock/Documents/MachineLearning/GroupAssignment/ML1819--task-101--team-23/Data/FlightClassificationCleaned.csv")
#x_data = flights.drop(['ARR_DELAY_BIN', 'ARR_DELAY'], axis=1)
#y_labels = flights['ARR_DELAY_BIN']
#X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.3, random_state=0)
#class_names = ['Punctual', 'Late']
#
#model = keras.Sequential()
#model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(128, activation=tf.nn.relu))
#model.add(keras.layers.Dense(len(class_names), activation=tf.nn.softmax))
#model.add(keras.layers.Flatten())
#model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),loss='mse',metrics=['accuracy'])
#model.fit(X_train.values, y_train.values, epochs=10)
#test_loss, test_acc = model.evaluate(X_test.values, y_test.values)
#print('Test accuracy:', test_acc)
#print('Test loss:', test_loss)
#predictions = model.predict(X_test)
##other, true = tf.metrics.recall(predictions, y_test)
#print('pred length', len(predictions))
#print('pred[0]: ', predictions[0])
##print('trues: ', true)





#OLD NNC
# TensorFlow and tf.keras
#import tensorflow as tf
#from tensorflow import keras
#import sklearn.metrics as skm
#from sklearn.model_selection import train_test_split
#
## Helper libraries
#import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
#from matplotlib import pyplot as plt
#import pandas as pd
#
#def label_fix(label):
#    #delayed = 1, not delayed = 0
#    if label<=0:
#        return 0
#    else:
#        return 1
#
#if __name__ =='__main__':
#    flights = pd.read_csv("/Users/jamessherlock/Documents/MachineLearning/GroupAssignment/ML1819--task-101--team-23/Data/Flight-Delays-July-2018-Cleaned.csv")
#    #fashion_mnist = keras.datasets.fashion_mnist
#    #print('f', fashion_mnist)
#    #print('p', flights)
#    #(X_train, y_train), (X_test, y_test) = flights.load_data()
#    flights['ARR_DELAY'] = flights['ARR_DELAY'].apply(label_fix)
#    x_data = flights.drop(['MKT_CARRIER', 'ORIGIN', 'DEST', 'ARR_DELAY'], axis=1)
#    y_labels = flights['ARR_DELAY']
#    X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.25, random_state=0)
#    class_names = ['early', 'delayed']
#    print('Cols: ', x_data.columns)
#    print('Y_train: ', y_train)
#    print(len(X_train))
#    print('X_train: ', X_train)
#    model = keras.Sequential()
#    model.add(keras.layers.Flatten())
#    model.add(keras.layers.Dense(128, activation=tf.nn.relu))
#    model.add(keras.layers.Dense(len(class_names), activation=tf.nn.softmax))
#    model.add(keras.layers.Flatten())
#    model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),loss='mse',metrics=['accuracy'])
#    model.fit(X_train.values, y_train.values, epochs=10)
#    test_loss, test_acc = model.evaluate(X_test.values, y_test.values)
#    print('Test accuracy:', test_acc)
#    print('Test loss:', test_loss)
#    predictions = model.predict(X_test)
#    #other, true = tf.metrics.recall(predictions, y_test)
#    print('pred length', len(predictions))
#    print('pred[0]: ', predictions[0])
##print('trues: ', true)


#straight up works
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
##record_defaults = [tf.float32] * 5
##X_dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[1,2,3,4,5])
##Y_dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[6])
#
##print(X_dataset.output_shapes)
##print(Y_dataset.output_shapes)
#x_data = flights.drop(['ARR_DELAY_BIN', 'ARR_DELAY'], axis=1)
#y_labels = flights['ARR_DELAY_BIN']
#print(x_data)
#print(y_labels)
#X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.3, random_state=0)
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
#classifier = tf.estimator.LinearClassifier(feature_columns=feat_cols)
#classifier.train(input_fn=train_func)
##
##test_func = easy_input_function(X_test, X_test, num_epochs=1, shuffle=False, batch_size=100)
#test_func = tf.estimator.inputs.pandas_input_fn(X_test, y_test, batch_size=100, num_epochs=1, shuffle=False)
#result = classifier.evaluate(input_fn=test_func)
##
#print(result)
