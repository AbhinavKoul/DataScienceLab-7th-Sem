import pandas as pd
from sklearn.cluster import KMeans 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix

#read data and transform class data
data = pd.read_csv("iris.csv")
data['Class'] = data['Class'].map(
    {
        'Iris-setosa' : 0,
        'Iris-versicolor':1,
        'Iris-virginica':2
    }
)

#select X(inp) and Y(out)
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3)

kmeans = KMeans(n_clusters = 3)
model = kmeans.fit(x_train,y_train)
y_pred = model.predict(x_test)

accuracy_score(y_test, y_pred)
recall_score(y_test, y_pred, average='weighted')
precision_score(y_test, y_pred, average='weighted')
f1_score(y_test, y_pred, average='weighted')