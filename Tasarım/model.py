#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


dataset = pd.read_csv("AracSecme.csv", encoding = 'iso-8859-9')
X = dataset.iloc[:, 2:5]

def convert_to_int(neighborhood):
    neighborhood_val = { 'car': 1, 'walk': 2, 'cycle': 3}
    return neighborhood_val[neighborhood]
X = X.values
y = dataset.iloc[:,1]
y = dataset.iloc[:,1].values.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder 
onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y).toarray() 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)
y_train_scale = scaler.fit_transform(y_train) 
y_test_scale = scaler.fit_transform(y_test) 


# veriler ile decision tree algoritması eğitilir.
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

pickle.dump(regressor, open('model.pkl', 'wb')) 
model = pickle.load(open('model.pkl', 'rb'))
y_pred = regressor.predict(X_test)
from sklearn.metrics import roc_auc_score
predictions = model.predict(X_test)
print('"0,3962" AUC: ', roc_auc_score(y_test, predictions))
