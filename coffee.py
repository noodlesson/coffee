from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
#from sklearn.metrics import auc_score
import pandas as pd

#load in train data file
 #result, list of blocks of time series
df = pd.read_csv('Coffee_TRAIN.txt', header=None)

#process the list above, splitting to list of number values, using double spaces
l = []
for i in df.values:
    l = l + [i[0].strip().split('  ')]

 #convert string numbers to float numbers
ts = []
for i in l:
    ts = ts + [[float(j) for j in i]]
    
#get train targets to a list of 1s and 0s
target_train= [i[0] for i in ts]

#convert X data to 2D dataframe for a binary model, naming columns as time indices
X_train = pd.DataFrame(data=ts, columns = ['target'] + ['time_index_' + str(i) for i in range(286)])
X_train = X_train[['time_index_' + str(i) for i in range(286)]]

#create a pipe, having just 2 components
column_names = ['time_index_' + str(i) for i in range(286)]

 #component 1 of the pipte
col_selector = ColumnTransformer(transformers = [('selector','passthrough', column_names)], remainder = 'drop')

 #component 2 of the pipe
model = RandomForestClassifier(random_state=1, n_estimators = 130)

 #pipe build
pipe = Pipeline([('select columns', col_selector), ('model', model)])


#load in test data
df_test = pd.read_csv('Coffee_TEST.txt', header=None)
l_test = []
for i in df_test.values:
    l_test = l_test + [i[0].strip().split('  ')]

#convert string numbers to float numbers
ts_test = []
for i_test in l_test:
    ts_test = ts_test + [[float(j) for j in i_test]]

#get test target data
target_test = [i[0] for i in ts_test]

#convert X data to 2D dataframe for binary model, naming columns as time indices
X_test = pd.DataFrame(data=ts_test, columns = ['target'] + ['time_index_' + str(i) for i in range(286)])
X_test = X_test[['time_index_' + str(i) for i in range(286)]]

##Pipe working##
#pipe train
pipe.fit(X_train, target_train)

#pipe score
print('acc {:.2%}'.format(pipe.score(X_test, target_train)))

#model evaluation for monitoring
#todo
