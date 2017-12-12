#declaring imports
from sklearn.cluster import MiniBatchKMeans
import pandas as pd

#reading csv file to dataframe object
df=pd.read_csv('/home/tharindra/PycharmProjects/WorkBench/FinalizedFinalProject/MusicClassification/DuplicatesRemoved.csv',header=None)

#extracting features based on index based location
features=df.iloc[:,0:3]

#fitting the model instance to datapoints
kmeans = MiniBatchKMeans(n_clusters=3, random_state=42,batch_size=6).fit(features)

#retreiving labels
trainCluster=kmeans.labels_

#converting the array of labels to a pandas dataframe for concatenation
df1=pd.DataFrame(trainCluster)

#concatenating the two dataframes on acolumn wise manner
dff=pd.concat([features,df1],axis=1)

#return all records in the dataframe in random
dff1=dff.sample(frac=1)

#save dataframe values to csv
dff1.to_csv('MiniBatchKMeans.csv',index=False,header=False)



