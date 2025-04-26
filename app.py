from flask import Flask, jsonify, request
from flask_cors import CORS
from database import get_connection
import uuid
import joblib
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory



app = Flask(__name__)
CORS(app)

model = joblib.load('Backend Python/model_multinomial_nb.pkl')


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


def slang_to_formal(text, slang_dict):
    text = str(text) 
    words = text.split()
    formal_words = [slang_dict.get(word, word) for word in words]
    return " ".join(formal_words)


def load_slang_dict_from_csv(csv_file):
    df = pd.read_csv(csv_file,encoding='latin-1')

    if 'anakjakartaasikasik' in df.columns and 'anak jakarta asyik asyik' in df.columns:
        slang_dict = dict(zip(df['anakjakartaasikasik'], df['anak jakarta asyik asyik']))
        return slang_dict
    else:
        raise ValueError("Kolom 'aamiin' dan 'amin' tidak ditemukan dalam file CSV")
    
    
def tokenize(text):
    text = word_tokenize(text)
    return text


factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()


def remove_stopwords(text):
    if isinstance(text, list):  
        text = " ".join(text)
    elif not isinstance(text, str):  
        return ""
    text = stopword_remover.remove(text)  
    return text.split()  


factory = StemmerFactory()

stemmer = factory.create_stemmer()

def stem_text(text):
    stemmed_tokens = [stemmer.stem(token) for token in text]
    return ' '.join(stemmed_tokens)

@app.route('/api/predict', methods=['POST'])    
def create_prediction():
    data = request.get_json()
    data = prep(data)
    slang_dict = load_slang_dict_from_csv('/Hackathon/dataset/new_kamusalay.csv')
    data = slang_to_formal(data, slang_dict)
    data = tokenize(data)
    data = remove_stopwords(data)
    data = stem_text(data)
    
    vectorizer = joblib.load("vectorizer.pkl")
    data = vectorizer.transform([data]).toarray()
    
    prediction = model.predict(data)
    
    if prediction == 0 :
        prediction = "Negatif"
    else : 
        prediction = "Positif"
    
    prediction_id = str(uuid.uuid4())
    prediction_results[prediction_id] = prediction
    response = {
        'id' : prediction_id,
        'prediction' : prediction
    }
    return jsonify(response), 201


if __name__ == '__main__':
    app.run(debug=True)