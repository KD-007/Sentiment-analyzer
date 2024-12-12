from flask import Flask , request , jsonify , render_template
import tensorflow as tf
import numpy as np
import re
import joblib

import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords



app = Flask(__name__)

model = None
vectorizer = None
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stop_words]
    return ' '.join(review)

@app.route('/' , methods=["GET", "POST"])
def index():
    return render_template('index.html')


@app.route("/predict", methods=["POST", "GET"])
def predict():
    data = request.get_json()
    print(data)
    
    preprocessed_text = preprocess_text(data['data'])  # Extract preprocessed text from the list
    preprocessed_text = vectorizer.transform(np.array([preprocessed_text])).toarray() 
    prediction = model.predict(preprocessed_text)
    
    prediction = (prediction > 0.5).astype(int)
    
    sentiment = np.argmax(prediction, axis=1)[0]

    # Map the sentiment to categories
    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    result = sentiment_mapping[sentiment]
    
    response = {
        'prediction': result
    }
    
    return jsonify(response)
    
    
if __name__ == '__main__':
    model = tf.keras.models.load_model('./keras_model_s.h5')
    vectorizer = joblib.load('./tfidf_vectorizer_s.pkl')
    print("app is running")
    
    app.run(debug=True)