import pandas as pd
import numpy as np 
import pickle
import joblib

# Verinin Okunması #s
column = ['Cümle']
df = pd.read_csv('SmartClassRoom.csv',encoding = 'utf-8',sep = '"')
df.columns = column
df.info()
#Veri setindeki Türkçe Dolgu kelimlerinin kaldırılması 
def remove_stopwords(df_fon):
    stopwords = open('turkece-stop-words.txt','r').read().split()
    df_fon['stopwords_removed'] = list(map(lambda doc:
        [word for word in doc if word not in stopwords],df_fon['Cümle']))
        
# 1 Fan Açık 2 Fan Kapalı 3 Işık Açık 4 Işık Kapalı
daf = remove_stopwords(df)
df['Sinif'] = 1
df.Sinif.iloc[5:10] = 2
df.Sinif.iloc[10:15] = 3
df.Sinif.iloc[15:21] = 4

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(df['Cümle'],df['Sinif'],test_size = 0.5,random_state = 0)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(encoding ='iso-8859-9').fit(X_train)
feature_list = vect.get_feature_names()
joblib.dump(feature_list, 'SmartClassroomvocabulary.pkl')
#X_Train'deki belgeleri bir belge terim matrisine dönüştürüyoruz
X_train_vectorizer = vect.transform(X_train)

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train_vectorizer , Y_train)

pickle.dump(regressor, open('SmartClassroomModel.pkl', 'wb'))  
model = pickle.load(open('SmartClassroomModel.pkl', 'rb'))

from sklearn.metrics import roc_auc_score
predictions = model.predict(vect.transform(X_test))

from sklearn.metrics import zero_one_loss
zero_one_loss(Y_test, predictions)
print("Zero_One_Loss : " , zero_one_loss(Y_test, predictions))