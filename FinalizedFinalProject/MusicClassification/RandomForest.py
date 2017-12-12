#declaring imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

dataset=np.loadtxt('/home/tharindra/PycharmProjects/WorkBench/FinalizedFinalProject/MusicClassification/DatasetAfterClustering.csv',delimiter=',',skiprows=1)

features=dataset[:,0:3]
labels=dataset[:,3]

tr_features,ts_features,tr_labels,ts_labels=train_test_split(features,labels,test_size=0.5,random_state=42)

#creating an instance of the random forest classifier
model=RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=4, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", oob_score=True, n_jobs=-1)

#training model
model.fit(tr_features,tr_labels)
#predicting based on trained model 
y=model.predict(ts_features)
pred=pd.DataFrame(y)
act=pd.DataFrame(ts_labels)

#concatenating preictions against actual
final=pd.DataFrame(pd.concat([pred,act],axis=1))

count=0.0

for i in range(len(y)):
    if(y[i]==ts_labels[i]):
        count=count+1
    else:
        count=count+0

#accuracy calculation
Accuracy=model.score(ts_features,ts_labels)

print (count/(ts_labels.shape[0]))*100 #99.8317712088
print pd.Series(ts_labels).value_counts()
print pd.Series(y).value_counts()
print model.feature_importances_ #[ 0.92408555  0.02874635  0.0471681 ]
