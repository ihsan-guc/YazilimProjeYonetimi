#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv("TitanikVeri.csv", encoding = 'iso-8859-9')
X = dataset.iloc[:, 2:9]
X = X.values
y = dataset.iloc[:,1]

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1]) 
X[:, 6] = labelencoder.fit_transform(X[:, 4])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)
pickle.dump(regressor, open('TitanikModel.pkl', 'wb')) 
model = pickle.load(open('TitanikModel.pkl', 'rb'))
y_pred = regressor.predict(X_test)

from sklearn.metrics import roc_auc_score
predictions = model.predict(X_test)

print('ROC_AUC_SCORE :', roc_auc_score(y_test, predictions))
