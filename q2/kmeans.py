import numpy as np
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture 
import pandas as pd


X = pd.read_csv("kmeansdata.csv")
x1 = X['Distance_Feature'].values
x2 = X['Speeding_Feature'].values

X = np.array(list(zip(x1,x2))).reshape(len(x1),2)
print(X)

plt.plot()
plt.xlim([0,100])
plt.ylim([0,50])
plt.title('Dataset')
plt.scatter(x1,x2)
plt.show()

#code for EM
gmm = GaussianMixture(n_components = 3)
gmm.fit(X)
em_predictins = gmm.predict(X)
em_predictins

gmm.means_
gmm.covariances_
#plots
plt.title('Exception Maximum')
plt.scatter(X[:,0],X[:,1],c=em_predictins,s=50)
plt.show()


#KMeans
kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)

kmeans.cluster_centers_
kmeans.labels_
#plot
plt.title('kmeans')
plt.scatter(X[:,0],X[:,1],c=kmeans.labels_,cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='black')