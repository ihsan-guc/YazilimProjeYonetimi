import pandas as pd
import numpy as np 
import pickle
import joblib

# Verinin Okunması #s
column = ['MansetBaslık']
df = pd.read_csv('HeadLines.csv',encoding = 'utf-8',sep = '"')
df.columns = column
df.info()
        
# 1 Ekonomi 2 Sağlık
df['Manset'] = 1
df.Manset.iloc[10:21] = 2

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(df['MansetBaslık'],df['Manset'],test_size = 0.6,random_state = 0)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(encoding ='iso-8859-9').fit(X_train)
feature_list = vect.get_feature_names()
joblib.dump(feature_list, 'NewspaperShelvesvocabulary.pkl')
X_train_vectorizer = vect.transform(X_train)

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train_vectorizer , Y_train)

pickle.dump(regressor, open('NewspaperShelvesModel.pkl', 'wb'))  
model = pickle.load(open('NewspaperShelvesModel.pkl', 'rb'))

from sklearn.metrics import roc_auc_score
predictions = model.predict(vect.transform(X_test))

from sklearn.metrics import roc_auc_score
roc_auc_score(Y_test, predictions)
print(roc_auc_score(Y_test, predictions))
#print('"0,8333" AUC: ', roc_auc_score(Y_test, predictions))





