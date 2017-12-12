#declaring imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt


TrainSet=np.loadtxt('/home/tharindra/PycharmProjects/WorkBench/FinalizedFinalProject/MusicClassification/DatasetAfterClustering.csv',delimiter=',',skiprows=1)
print(TrainSet.shape)

features = TrainSet[:,0:3]
labels = TrainSet[:,3]

tr_features,ts_features,tr_labels,ts_labels=train_test_split(features,labels,test_size=0.5,random_state=42)

eval_set=[(tr_features,tr_labels),(ts_features,ts_labels)]
#creating xgb classifier
model=xgb.XGBClassifier(max_depth=3,learning_rate=0.01,n_estimators=100,objective='multi:softmax')
model.fit(tr_features,tr_labels,eval_metric='merror',eval_set=eval_set,early_stopping_rounds=10)

y=model.predict(ts_features)

xgb.plot_importance(model)
plt.show()

print(pd.Series(ts_labels).value_counts())
print (pd.Series(y).value_counts())

print (accuracy_score(ts_labels,y))
