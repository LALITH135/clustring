# -*- coding: utf-8 -*-
"""
Created on Mon May 30 23:07:03 2022

@author: lalith kumar
"""

# Performing clustering (hierarchical,K means clustering and DBSCAN) for the airlines dataset.  

# importing datasets.
import pandas as pd 
df=pd.read_excel('E:\\data science\\ASSIGNMENTS\\ASSIGNMENTS\\clustring\\EastWestAirlines.xlsx',sheet_name=1)
df
df.shape
df.describe()
df.info()
df.head()
df.columns

# fixing the data, by removing the unwanted column.
ewa=df.iloc[:,1:]
ewa.info()
ewa.describe()

# standardization. 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(ewa)
X

# DBSCAN
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=7,min_samples=5)
dbscan=dbscan.fit(X)

#Noisy samples are given the label -1.
dbscan.labels_

# Adding clusters to dataset
ewa['clusters']=dbscan.labels_
ewa.columns

noisedata = ewa[ewa['clusters']==-1]
noisedata.sum()
finaldata = ewa[ewa['clusters'] ==0]
finaldata

ewa.groupby('clusters').agg(['mean']).reset_index()

# Ploting Clusters.

import seaborn as sns 

sns.pairplot(ewa.iloc[:,:3])
sns.pairplot(ewa.iloc[:,3:6])
sns.pairplot(ewa.iloc[:,6:9])
sns.pairplot(ewa.iloc[:,9:12])

#-------------------------------------------------------

# kmeans
#using the elbow method to find the optimal no. of clusters

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
KM = []
for i in range(1,15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state =42)
    kmeans.fit(X)
    KM.append(kmeans.inertia_)
plt.plot(range(1, 15), KM)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('KM')
plt.show()

#The scree plot levels of at k=3 and let's use it to determine the clusters

# for centroid initialization:
#K-Means++ is a smart centroid initialization technique and the rest of the algorithm is the same as that of K-Means. 
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state =30)
y_means=kmeans.fit(X)
labels=y_means.labels_


y_cluster_km=pd.Series(labels)
ewa["cluster_kmean"]=y_cluster_km
ewa
ewa.groupby(ewa.cluster_kmean).mean()

plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label = 'Centroids')
plt.show()

#-----------------------------------------------------------------------

# heirarchial Clustering.

import scipy.cluster.hierarchy as shc

# construction of Dendogram
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))  
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
dend = shc.dendrogram(shc.linkage(X, method='ward')) 

## Forming a group using clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
Y = cluster.fit_predict(X)

#Ward is similar to k-means.
# It may be a good choice if you want to talk about centroids and data partitioned completely into disjoint subsets.

Y_clust_H = pd.DataFrame(Y)
Y_clust_H.value_counts()

ewa['clust_H']=Y_clust_H

ewa.groupby(ewa.clust_H).mean()

#visualization
# Visualising the clusters
plt.scatter(X[Y==0, 0], X[Y==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[Y==1, 0], X[Y==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[Y==2, 0], X[Y==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[Y==3, 0], X[Y==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(X[Y==4, 0], X[Y==4, 1], s=100, c='magenta', label ='Cluster 5')
#Plot the centroid. This time we're going to use the cluster centres 
#attribute that returns here the coordinates of the centroid.
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label = 'Centroids')
plt.show()

#-----------------------------------------------------------------------









