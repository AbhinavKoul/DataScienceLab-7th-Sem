import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('Advertising.csv')
data.drop(['Unnamed: 0'], axis=1)

#plot
plt.figure(figsize=(16,8))
plt.scatter(data['TV'],data['sales'])

#linear regression
X = data['TV'].values.reshape(-1,1)
y = data['sales'].values.reshape(-1,1)
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

reg = LinearRegression()
reg.fit(x_train,y_train)
reg.coef_
reg.intercept_
predictions = reg.predict(x_test)

#plot
plt.figure(figsize = (16,8))
plt.scatter(x_test,y_test,c='black')
plt.plot(x_test,predictions,linewidth=2)

rmse = np.sqrt(mean_squared_error(y_test,predictions))
r2 = r2_score(y_test,predictions)