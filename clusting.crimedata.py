# -*- coding: utf-8 -*-
"""
Created on Mon May 28 23:00:50 2022

@author: lalith kumar
"""
# Performing Clustering(Hierarchical, Kmeans & DBSCAN) for the crime data. 

#importing dataset.

import pandas as pd  

df = pd.read_csv('E:\data science\ASSIGNMENTS\ASSIGNMENTS\clustring\crime_data.csv') 
df.shape
df.head()
list(df)
df.describe()
df.info()
data = df.iloc[:,1:]

# standardization
from sklearn.preprocessing import StandardScaler
stscaler = StandardScaler()
X = stscaler.fit_transform(data)
X

# DBSCAN.

from sklearn.cluster import DBSCAN
DBSCAN()
dbscan = DBSCAN(eps=1, min_samples=3)
dbscan = dbscan.fit(X)

#Noisy samples are given the label -1.
dbscan.labels_

data['clusters']=dbscan.labels_
data

noisedata = data[data['clusters']==-1]

finaldata = data[data['clusters']==0]

data.groupby('clusters').agg(['mean']).reset_index()

# Ploting Clusters.

import matplotlib.pyplot as plt
import seaborn as sns 

sns.pairplot(data)
plt.figure(figsize=(12, 8))
plt.scatter(data['clusters'],data['Murder'], c=dbscan.labels_)
plt.scatter(data['clusters'],data['Assault'], c=dbscan.labels_)
plt.scatter(data['clusters'],data['UrbanPop'], c=dbscan.labels_)
plt.scatter(data['clusters'],data['Rape'], c=dbscan.labels_)

#------------------------------------------------------------------
# heirarchial Clustering

import scipy.cluster.hierarchy as shc

# construction of Dendogram
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 7))  
plt.title("crime data")  
plt.xlabel('Index')
plt.ylabel('Distance')
dend = shc.dendrogram(shc.linkage(data, method='complete')) 

## Forming a group using clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
Y = cluster.fit_predict(data)

Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()


Y_clust = pd.DataFrame(Y)
Y_clust[0].value_counts()

from sklearn.cluster import KMeans

# Visualising the  clusters.
plt.scatter(X[Y==0, 0], X[Y==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[Y==1, 0], X[Y==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[Y==2, 0], X[Y==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[Y==3, 0], X[Y==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(X[Y==4, 0], X[Y==4, 1], s=100, c='magenta', label ='Cluster 5')


#--------------------------------------------------------------------------

# import KMeans
#using the elbow method to find the optimal no. of clusters

KM = []
for i in range(1,15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state =30)
    kmeans.fit(X)
    KM.append(kmeans.inertia_)
plt.plot(range(1, 15), KM)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('KM')
plt.show()

# for centroid initialization:
#K-Means++ is a smart centroid initialization technique and the rest of the algorithm is the same as that of K-Means.

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state =30)
y_means=kmeans.fit(X)
labels=y_means.labels_

y_cluster_km=pd.Series(labels)
data["cluster_kmean"]=y_cluster_km
data
data.groupby(data.cluster_kmean).mean()

y_kmeans1=labels+1 #just for visualization
cluster = list(y_kmeans1)

plt.figure(figsize=(12,6))
sns.scatterplot(x=data['Murder'],y=data['Assault'],hue=y_kmeans1)
sns.scatterplot(x=data['Murder'],y=data['Rape'],hue=y_kmeans1)
sns.scatterplot(x=data['Rape'],y=data['Assault'],hue=y_kmeans1)

# Visualising the clusters
plt.scatter(X[y_kmeans1==0, 0], X[y_kmeans1==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_kmeans1==1, 0], X[y_kmeans1==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_kmeans1==2, 0], X[y_kmeans1==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_kmeans1==3, 0], X[y_kmeans1==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(X[y_kmeans1==4, 0], X[y_kmeans1==4, 1], s=100, c='magenta', label ='Cluster 5')
#Plot the centroid. This time we're going to use the cluster centres  #attribute that returns here the coordinates of the centroid.
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label = 'Centroids')
plt.show()

#------------------------#--------------------#--------------------------#-----------------#
 

