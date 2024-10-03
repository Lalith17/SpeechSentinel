from flask import Flask, render_template, request
# from flask_cors import CORS
import pandas as pd
import numpy as np
import nltk
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split    
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')

app = Flask(__name__)
# CORS(app)

# Load and preprocess data
def preprocess_data():
    # Load data
    data = pd.read_csv("labeled_data.csv")
    data["labels"] = data["class"].map({0: "Hate speech", 1: "Offensive speech", 2: "No hate and offensive speech"})
    data = data[["tweet", "labels"]]

    # Preprocessing function
    stopword = set(stopwords.words('english'))
    stemmer = nltk.SnowballStemmer("english")

    def clean(text):
        text = str(text).lower()
        text = re.sub('[.?]', '', text)
        text = re.sub('https?://\\S+|www\\.\\S+', '', text)  
        text = re.sub('<.*?>', '', text)  
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\\w\\d\\w', '', text)  
        text = [word for word in text.split() if word not in stopword]
        text = " ".join(text)
        text = [stemmer.stem(word) for word in text.split()]
        text = " ".join(text)
        return text

    # Apply cleaning function
    data["tweet"] = data["tweet"].apply(clean)

    return data

# Train model
def train_model(data):
    x = np.array(data["tweet"]) 
    y = np.array(data["labels"])

    cv = CountVectorizer()
    X = cv.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    return model, cv

data = preprocess_data()
model, cv = train_model(data)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input')
def input():
    return render_template('input_page.html')

@app.route('/index1')
def index1():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if request.method == 'POST':
            text = request.form['text1']
            # Preprocess input text
            text = cv.transform([text]).toarray()
            # Predict using the model
            result = model.predict(text)
            return render_template('input_page.html', result=result)
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(debug=True)
