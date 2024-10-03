#importing the required modules
import pandas as pd
import numpy as np
import nltk
import re
import string  # Import the 'string' module

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split    
from sklearn.tree import DecisionTreeClassifier

#setting the language to english
stopword = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")

# reading the data using pandas
data = pd.read_csv("labeled_data.csv")
#setting the offensive level of the words
data["labels"] = data["class"].map({0: "Hate speech", 1: "Offensive speech", 2: "No hate and offensive speech"})

data = data[["tweet", "labels"]]

def clean(text):
    text = str(text).lower()
    text = re.sub('[.?]', '', text)
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)  # Fixed escape sequence
    text = re.sub('<.*?>', '', text)  # Fixed escape sequence
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\\w\\d\\w', '', text)  # Fixed escape sequence
    text = [word for word in text.split() if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split()]
    text = " ".join(text)
    return text


data["tweet"] = data["tweet"].apply(clean)

x = np.array(data["tweet"]) 
y = np.array(data["labels"])

cv = CountVectorizer()

X = cv.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
#print("The accuracy in the prediction of the hate speech is ", accuracy_score(y_test, y_pred))

i = input("Enter the text you feel offensive: ")
i = cv.transform([i]).toarray()
print(model.predict((i)))
