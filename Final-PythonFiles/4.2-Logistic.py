import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

x = np.arange(10).reshape(-1,1)
y = np.array([0,0,0,0,1,1,1,1,1,1])

model = LogisticRegression()
model.fit(x,y)
model.classes_
model.intercept_
model.coef_
model.predict_proba(x)
model.predict(x)
model.score(x,y)


confusion_matrix(y,model.predict(x))
classification_report(y,model.predict(x))