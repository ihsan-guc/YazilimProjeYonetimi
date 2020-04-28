import pandas as pd
import numpy as np 
import pickle
import joblib

# Verinin Okunması #
column = ['Cümle']
df = pd.read_csv('TouristinInfo.csv',encoding = 'utf-8',sep = '"')
df.columns = column
df.info()
#Veri setindeki Türkçe Dolgu kelimlerinin kaldırılması 
def remove_stopwords(df_fon):
    stopwords = open('turkce-stop-words','r').read().split()
    df_fon['stopwords_removed'] = list(map(lambda doc:
        [word for word in doc if word not in stopwords],df_fon['yorum']))
        
# 1 Müze 2 Balıklar 3 Araba
df['Info'] = 1
df.Info.iloc[5:10] = 2
df.Info.iloc[10:16] = 3

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(df['Cümle'],df['Info'],test_size = 0.4,random_state = 0)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(encoding ='iso-8859-9').fit(X_train)
feature_list = vect.get_feature_names()
joblib.dump(feature_list, 'TouristinInfoVocabulary.pkl')
#X_Train'deki belgeleri bir belge terim matrisine dönüştürüyoruz
X_train_vectorizer = vect.transform(X_train)

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train_vectorizer , Y_train)

pickle.dump(regressor, open('TouristinInfoModel.pkl', 'wb'))  
model = pickle.load(open('TouristinInfoModel.pkl', 'rb'))

from sklearn.metrics import roc_auc_score
predictions = model.predict(vect.transform(X_test))

from sklearn.metrics import recall_score
recall_score(Y_test, predictions, average='macro')
print(recall_score(Y_test, predictions, average='macro'))
#print('"0,745" AUC: ', roc_auc_score(Y_test, predictions))

