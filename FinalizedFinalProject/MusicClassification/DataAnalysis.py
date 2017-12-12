#importing data manipulation library
import pandas as pd

#reading csv file
df=pd.read_csv('/home/tharindra/PycharmProjects/WorkBench/FinalizedFinalProject/MusicClassification/DatasetAfterClustering.csv')

#grouping minimums and maximums of each feature based on user ratings
Max=df.groupby('rating').max()
Min=df.groupby('rating').min()

#printing results
print Min
print Max
