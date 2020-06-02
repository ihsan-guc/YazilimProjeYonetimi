import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv("AracSecme.csv", encoding = 'iso-8859-9')
X = dataset.iloc[:, 2:5]

X = X.values
y = dataset.iloc[:,1]
y = dataset.iloc[:,1].values.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder 
onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y).toarray() 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

pickle.dump(regressor, open('AracSecmeModel.pkl', 'wb')) 
model = pickle.load(open('AracSecmeModel.pkl', 'rb'))
y_pred = regressor.predict(X_test)
from sklearn.metrics import roc_auc_score
predictions = model.predict(X_test)
print('ROC_AUC_SCORE: ', roc_auc_score(y_test, predictions))
