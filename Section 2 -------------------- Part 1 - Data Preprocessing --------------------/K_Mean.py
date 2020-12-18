import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet=pd.read_csv("Mall_Customers.csv")
x=dataSet.iloc[:,[3,4]].values
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)

kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
yMeans=kmeans.fit_predict(x)

plt.scatter(x[yMeans==0,0],x[yMeans==0,1],s=100,c='red')
plt.scatter(x[yMeans==1,0],x[yMeans==1,1],s=100,c='blue')
plt.scatter(x[yMeans==2,0],x[yMeans==2,1],s=100,c='green')
plt.scatter(x[yMeans==3,0],x[yMeans==3,1],s=100,c='cyan')
plt.scatter(x[yMeans==4,0],x[yMeans==4,1],s=100,c='magenta')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow')