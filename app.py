from flask import Flask, jsonify, request
from flask_cors import CORS
from database import get_connection
import uuid
import joblib
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import string
import unicodedata
import emoji
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB



app = Flask(__name__)
CORS(app)

model = joblib.load('model_multinomial_nb.pkl')


def prep(tweet):
    tweet = re.sub("@[A-Za-z0-9_]+", "", tweet)
    tweet = re.sub(r'([a-zA-Z])([A-Z])', r'\1 \2', tweet)

    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet)
    tweet = re.sub(r'(.)\1+', r'\1', tweet)

    tweet = re.sub(r"\d+", "", tweet)


    tweet = ''.join(c for c in tweet if not emoji.is_emoji(c))

    tweet = re.sub(r'([a-zA-Z])(\.|,|\?|!|;|:)([a-zA-Z])', r'\1 \3', tweet)
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))

    tweet = " ".join(tweet.split())


    tweet = unicodedata.normalize('NFKD', tweet).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    tweet = tweet.replace("#", "").replace("_", " ")
    return tweet
  
def predict_sentiment(text_input):
    # Load model dan vectorizer
    model = joblib.load('model_multinomial_nb.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    # Transformasi teks input menggunakan vectorizer
    text_tfidf = vectorizer.transform([text_input]).toarray()

    # Prediksi menggunakan model
    prediction = model.predict(text_tfidf)


    return prediction[0]
    
@app.route('/api/predict', methods=['POST'])    
def create_prediction():
    data = request.get_json()
    data = prep(data)
    prediction = predict_sentiment(data)
    
    if prediction == 0 :
        prediction = "Negatif"
    else : 
        prediction = "Positif"
    
    prediction_id = str(uuid.uuid4())
    response = {
        'id' : prediction_id,
        'prediction' : prediction
    }
    return response


if __name__ == '__main__':
    app.run(debug=True)