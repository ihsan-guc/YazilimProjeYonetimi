import pandas as pd
import numpy as np 
import pickle
import joblib

# Verinin Okunması #
column = ['Cümle']
df = pd.read_csv('SortingHat.csv',encoding = 'utf-8',sep = '"')
df.columns = column
df.info()
        
# 0 gryffindor 1 slytherin 2 hufflepuff 3 ravenclaw 
df['Tur'] = 0
df.Tur.iloc[5:10] = 1
df.Tur.iloc[10:15] = 2
df.Tur.iloc[15:20] = 3

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(df['Cümle'],df['Tur'],test_size = 0.20 ,random_state = 0)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(encoding ='iso-8859-9').fit(X_train)
feature_list = vect.get_feature_names()
joblib.dump(feature_list, 'SortingHatVocabulary.pkl')
#X_Train'deki belgeleri bir belge terim matrisine dönüştürüyoruz
X_train_vectorizer = vect.transform(X_train)

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train_vectorizer , Y_train)

pickle.dump(regressor, open('SortingHatModel.pkl', 'wb'))  
model = pickle.load(open('SortingHatModel.pkl', 'rb'))

from sklearn.metrics import roc_auc_score
predictions = model.predict(vect.transform(X_test))

from sklearn.metrics import precision_score
precision_score(Y_test, predictions, average='macro')
print(precision_score(Y_test, predictions, average='macro'))
#print('"0,745" AUC: ', roc_auc_score(Y_test, predictions))


















