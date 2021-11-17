from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split 
import pandas as pd

dataset = pd.read_csv("iris.csv") 

X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

classifier = KNeighborsClassifier(n_neighbors=8,p=3,metric='euclidean')
classifier.fit(X_train,y_train)

#predict the test resuts 
y_pred=classifier.predict(X_test)

cm=confusion_matrix(y_test,y_pred) 

print('Accuracy Metrics') 
print(classification_report(y_test,y_pred))
print(" correct predicition",accuracy_score(y_test,y_pred)) 
print(" wrong predicition",(1-accuracy_score(y_test,y_pred)))