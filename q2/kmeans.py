import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


X = pd.read_csv("kmeansdata.csv")
x1 = X['Distance_Feature'].values
x2 = X['Speeding_Feature'].values
X = np.array(list(zip(x1,x2))).reshape(len(x1),2)


plt.scatter(x1,x2)

#code for EM
gmm = GaussianMixture(n_components = 3)
gmm.fit(X)
em_predictins = gmm.predict(X)
em_predictins

gmm.means_
gmm.covariances_
#plots
plt.scatter(X[:,0],X[:,1],c=em_predictins,s=50)

#KMeans
kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)
kmeans.cluster_centers_
kmeans.labels_
#plot
plt.scatter(X[:,0],X[:,1],c=kmeans.labels_,cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='black')