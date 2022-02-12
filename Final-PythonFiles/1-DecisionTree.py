import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,f1_score,precision_score

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

id3 = DecisionTreeClassifier(criterion="entropy")
cart = DecisionTreeClassifier(criterion="gini")
id3.fit(x_train,y_train)
cart.fit(x_train,y_train)
y_pred_id3 = id3.predict(x_test)
y_pred_cart = cart.predict(x_test)

print("ID3 METRICS")
accuracy_score(y_test,y_pred_id3)
recall_score(y_test,y_pred_id3,average='micro')
f1_score(y_test,y_pred_id3,average='micro')
precision_score(y_test,y_pred_id3,average='micro')
confusion_matrix(y_test,y_pred_id3)

print("CART_METRICS")
accuracy_score(y_test,y_pred_cart)
recall_score(y_test,y_pred_cart,average='micro')
f1_score(y_test,y_pred_cart,average='micro')
precision_score(y_test,y_pred_cart,average='micro')
confusion_matrix(y_test,y_pred_cart)