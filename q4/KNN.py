import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("iris.csv") 

X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 4].values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

model = KNeighborsClassifier(n_neighbors=7,p=3,metric="euclidean")
model.fit(x_train,y_train)

#predict the test resuts 
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test,y_pred)

print('Accuracy Metrics') 
print(classification_report(y_test,y_pred))
accuracy_score(y_test,y_pred)
1-accuracy_score(y_test,y_pred)