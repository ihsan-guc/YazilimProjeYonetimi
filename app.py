import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from flask_cors import CORS

app = Flask(__name__)
AracSecmeModel = pickle.load(open('AracSecmeModel.pkl', 'rb'))

OkulKutuphanesiModel = pickle.load(open('OkulKutuphanesiModel.pkl', 'rb'))

TitanikModel = pickle.load(open('TitanikModel.pkl', 'rb'))

SmartClassroomvocabulary = joblib.load('SmartClassroomvocabulary.pkl') 
vectSmartClassroom = CountVectorizer(ngram_range=(1,2), vocabulary = SmartClassroomvocabulary)
CORS(app, resources={r"/*": {"origins": "*"}})
SmartClassroomModel = pickle.load(open('SmartClassroomModel.pkl', 'rb'))

NewspaperShelvesVocabulary = joblib.load('NewspaperShelvesvocabulary.pkl') 
vectNewspaperShelves = CountVectorizer(ngram_range=(1,2), vocabulary = NewspaperShelvesVocabulary)
CORS(app, resources={r"/*": {"origins": "*"}})
NewspaperShelvesModel = pickle.load(open('NewspaperShelvesModel.pkl', 'rb'))

SortingHatModelVocabulary = joblib.load('SortingHatVocabulary.pkl') 
vectSortingHat = CountVectorizer(ngram_range=(1,2), vocabulary = SortingHatModelVocabulary)
CORS(app, resources={r"/*": {"origins": "*"}})
SortingHatModel = pickle.load(open('SortingHatModel.pkl', 'rb'))

TouristinInfoVocabulary = joblib.load('TouristinInfoVocabulary.pkl') 
vectTouristinInfo = CountVectorizer(ngram_range=(1,2), vocabulary = TouristinInfoVocabulary)
CORS(app, resources={r"/*": {"origins": "*"}})
TouristinInfoModel = pickle.load(open('TouristinInfoModel.pkl', 'rb'))

MakeMeHappyVocabulary = joblib.load('MakeMeHappyVocabulary.pkl') 
vectMakeMeHappy = CountVectorizer(ngram_range=(1,2), vocabulary = MakeMeHappyVocabulary)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
MakeMeHappyModel = pickle.load(open('MakeMeHappyModel.pkl', 'rb'))

@app.route("/",methods=['GET', 'POST'])
def Home():
    return render_template("index.htm")

@app.route("/OkulAracSecme", methods = ['POST'])
def OkulAracSecme():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = AracSecmeModel.predict(final_features)
    predictionValue = str(prediction)
    print(str(prediction))
    if str(prediction) == "[[0. 0. 1.]]":
        output = "Yürüyerek ";
    if str(prediction) == "[[0. 1. 0.]]":
        output = "Bisiklet ile";    
    if str(prediction) == "[[1. 0. 0.]]":
        output = "Araba ile ";
    return render_template("index.htm", prediction_text = "Girdiğiniz verilere göre  {} Gidersiniz".format(output));

@app.route("/OkulKutuphanesiPredict", methods = ['POST'])
def OkulKutuphanesiPredict():
    int_features = [int(x) for x in request.form.values()]
    print(str(int_features))
    final_features = [np.array(int_features)]
    prediction = OkulKutuphanesiModel.predict(final_features)
    predictionValue = str(prediction)
    print(str(prediction))
    if str(prediction) == "[[0. 0. 1.]]":
        output = "Year R";
    if str(prediction) == "[[1. 0. 0.]]":
        output = "Key Stage 1";
    if str(prediction) == "[[0. 1. 0.]]":
        output = "Key Stage 2";
    return render_template("index.htm", prediction_text = "Size {} ".format(output) + "Öneriyoruz");

@app.route("/predictTitanik", methods = ['POST'])
def predictTitanik():
    int_features = [int(x) for x in request.form.values()]
    print(str(int_features))
    final_features = [np.array(int_features)]
    prediction = TitanikModel.predict(final_features)
    predictionValue = str(prediction)
    print(str(prediction))
    if str(prediction) == "[1.]":
        output = "Hayatda Kalacaksınız";
    if str(prediction) == "[0.]":
        output = "Öleceksiniz";
    return render_template("index.htm", prediction_text = "{}".format(output) + "");

@app.route("/SmartClassroom", methods = ['POST'])
def SmartClassroom():
    int_features = [request.json['predict']]
    print(int_features)
    prediction = SmartClassroomModel.predict(vectSmartClassroom.transform(int_features))
    predictionValue = str(prediction)
    print(predictionValue)
    output = predictionValue
    return  output

@app.route("/NewspaperShelvesPredict", methods = ['POST'])
def NewspaperShelvesPredict():
    int_features = [request.json['predict']]
    print(int_features)
    prediction = NewspaperShelvesModel.predict(vectNewspaperShelves.transform(int_features))
    predictionValue = str(prediction)
    print(predictionValue)
    output = predictionValue
    return  output

@app.route("/SortingHatPredict", methods = ['POST'])
def SortingHatPredict():
    int_features = [request.json['predict']]
    prediction = SortingHatModel.predict(vectSortingHat.transform(int_features))
    predictionValue = str(prediction)
    print(predictionValue)
    output = predictionValue
    return  output

@app.route("/TouristinInfoPredict", methods = ['POST'])
def TouristinInfoPredict():
    int_features = [request.json['predict']]
    print(int_features)
    prediction = TouristinInfoModel.predict(vectTouristinInfo.transform(int_features))
    predictionValue = str(prediction)
    print(predictionValue)
    output = predictionValue
    return  output

@app.route("/MakeMeHappyPredict", methods = ['POST'])
def MakeMeHappyPredict():
    int_features = [request.json['predict']]
    prediction = MakeMeHappyModel.predict(vectMakeMeHappy.transform(int_features))
    predictionValue = str(prediction)
    print(predictionValue)
    output = predictionValue
    return  output


if __name__ == "__main__":
    app.run(debug = True)