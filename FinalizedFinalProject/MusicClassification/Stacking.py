import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

dataset=np.loadtxt('/home/tharindra/PycharmProjects/WorkBench/FinalizedFinalProject/MusicClassification/DatasetAfterClustering.csv',delimiter=',',skiprows=1)

features=dataset[:,0:3]
labels=dataset[:,3]

tr_features,ts_features,tr_labels,ts_labels=train_test_split(features,labels,test_size=0.5,random_state=42)

#--------------RandomForest model------------------
model=RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=4, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", oob_score=True, n_jobs=-1)

model.fit(tr_features,tr_labels)
y=model.predict(ts_features)
pred=pd.DataFrame(y)
act=pd.DataFrame(ts_labels)

final=pd.DataFrame(pd.concat([pred,act],axis=1))

#--------------XGBoost model------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.preprocessing import LabelBinarizer

Train=np.loadtxt('/home/tharindra/PycharmProjects/WorkBench/FinalizedFinalProject/MusicClassification/DatasetAfterClustering.csv',delimiter=',',skiprows=1)
print(Train.shape)

Features = Train[:,0:3]
Labels = Train[:,3]

Tr_features,Ts_features,Tr_labels,Ts_labels=train_test_split(Features,Labels,test_size=0.5,random_state=42)

eval_set=[(Tr_features,Tr_labels),(Ts_features,Ts_labels)]

model=xgb.XGBClassifier(max_depth=3,learning_rate=0.01,n_estimators=100,objective='multi:softmax')
model.fit(Tr_features,Tr_labels,eval_metric='merror',eval_set=eval_set,early_stopping_rounds=10)

preds=model.predict(Ts_features)
preds=pd.DataFrame(preds)
print(preds)

data=pd.concat([preds,final],axis=1)

data.to_csv('NewDataset.csv',index=False)

#-------------------------DecisionTree model-------------------------------------
from sklearn.tree import DecisionTreeClassifier

data=np.loadtxt('NewDataset.csv',delimiter=',',)

x=data[:,0:2]
y=data[:,2]

train_features,test_features,train_labels,test_labels=train_test_split(x,y,test_size=0.5,random_state=42)
stacker=DecisionTreeClassifier(max_depth=3)
stacker.fit(train_features,train_labels)
Sp=stacker.predict(test_features)

print (stacker.score(test_features,test_labels))

print (pd.Series(test_labels).value_counts())
print (pd.Series(Sp).value_counts())
