# SpeechSentinel

A web application that detects and classifies hate speech using machine learning. This project preprocesses user input, classifies it into categories such as hate speech, offensive speech, or non-offensive speech, and returns the result.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)
- [Model](#model)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Overview
SpeechSentinel uses natural language processing (NLP) and a machine learning model to classify text as hate speech, offensive speech, or non-offensive speech. The app preprocesses text input, applies a trained DecisionTreeClassifier, and provides a prediction.

## Features
- Input text and analyze for hate speech categories.
- NLP preprocessing of text (removal of stopwords, stemming, etc.).
- Classification into three categories: Hate speech, Offensive speech, or Non-offensive speech.
- Flask-based web interface with input and results pages.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SpeechSentinel.git
   ```

2. Navigate to the project directory:
   ```bash
   cd SpeechSentinel
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask app:
   ```bash
   python app.py
   ```

## Usage
- Visit `http://127.0.0.1:5000/` in your browser.
- Input text in the provided field.
- The app will classify the text as one of three categories:
  - Hate speech
  - Offensive speech
  - No hate and offensive speech

## Technologies
- **Python 3.12**
- **Flask**
- **Pandas**
- **NumPy**
- **NLTK** for text preprocessing
- **scikit-learn** for machine learning model (DecisionTreeClassifier)

## Model
The model used is a DecisionTreeClassifier trained on a dataset labeled for hate speech and offensive speech. The input text is preprocessed by:
- Converting to lowercase
- Removing URLs and special characters
- Tokenizing and removing stopwords
- Applying stemming

The model achieves basic classification based on word frequency features.

## Future Enhancements
- Improve the model by experimenting with more complex algorithms like Random Forest or SVM.
- Add more sophisticated NLP techniques like lemmatization and part-of-speech tagging.
- Expand the dataset for better generalization.
- Implement API endpoints for model prediction.
- Improve the user interface.

## Contributing
Feel free to contribute to this project by submitting pull requests or suggesting features. Please open an issue for discussions before major changes.

## License
This project is licensed under the MIT License.
