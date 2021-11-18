import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,f1_score,precision_score

data = pd.read_csv('breast-cancer.data') 

id3 = DecisionTreeClassifier(criterion="entropy")
cart = DecisionTreeClassifier(criterion="gini")
encoder = OneHotEncoder(drop="first")

encoder.fit(data)
tf_df = pd.DataFrame(encoder.transform(data).toarrayray())

X = tf_df.iloc[:,:-1]
y = tf_df.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

id3.fit(x_train,y_train)
cart.fit(x_train,y_train)
y_pred_id3 = id3.predict(x_test)
y_pred_cart = cart.predict(x_test)

print("ID3 Tree Metrics")
accuracy_score(y_test,y_pred_id3)
recall_score(y_test,y_pred_id3)
precision_score(y_test,y_pred_id3)
f1_score(y_test,y_pred_id3)
confusion_matrix(y_test,y_pred_id3)

print("CART Tree Metrics")
accuracy_score(y_test,y_pred_cart)
recall_score(y_test,y_pred_cart)
precision_score(y_test,y_pred_cart)
f1_score(y_test,y_pred_cart)
confusion_matrix(y_test,y_pred_cart)