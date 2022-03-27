# import libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.externals import joblib
import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

import sys


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM RawData', engine)
    X = df.message
    Y = df.iloc[:,4:]
    # Remove NaN in Y.
    X_clean = X[~Y.isnull().any(axis = 1)]
    Y_clean = Y[~Y.isnull().any(axis = 1)]
    category_names = Y_clean.columns.tolist()
    return X_clean,  Y_clean, category_names


def tokenize(text):
    
    # Detect & Replace URL
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_regex, text)
    for url in urls:
        text = text.replace(url, "urlplaceholder")
        
    # normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    
    clean_words = []
    for word in words:
        clean_word = lemmatizer.lemmatize(word).strip()
        clean_words.append(clean_word)
        
    return clean_words


def build_model():
    # parameters from grid search: 'clf__estimator__n_estimators': 12, 'clf__estimator__n_jobs': 1, 'vect__max_df': 1.0
    pipeline_cv = Pipeline([
        ('vect',CountVectorizer(tokenizer = tokenize, max_df = 1.0)),
        ('tfidf',TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 12, n_jobs = 1)))
    ])
    return pipeline_cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f'Report for category: {col}:')
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()