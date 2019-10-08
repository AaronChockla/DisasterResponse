"""
Train Classifier for Disaster Response Pipeline Project

Arguments:
    1) CSV file containing messages (e.g., disaster_messages.csv)
    2) CSV file containing categories (e.g., disaster_categories.csv)
    3) SQLite destination database (e.g., DisasterResponse.db)

    1) SQLite database path (e.g., ../data/DisasterResponse.db)
    2) File name for pickle file of ML model (e.g., classifier.pkl)
    
Example:
    python train_classifier.py ../data/DisasterResponse.db classifier.pkl
"""

import sys
import pandas as pd
import numpy as np
import re
import time
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.grid_search import GridSearchCV
from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet','stopwords'])



def load_data(database_filepath):
    """
    Function to load database of disaster response messages

    Arguments:
        1) database_filepath: path to db containing messages and categories

    Output:
        1) X: database of feature variables
        2) Y: database of target variables
        3) categories: list of target category names
    """

    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df.message

    cols = [col for col in df.columns if col not in ['id','message','original','genre']]
    Y = df[cols]
    categories = Y.columns
    
    return X, Y, categories


def tokenize(text):
    """
    Function to process process and tokenize text data from a message
    
    Arguments:
        1) text: message to be processed
    
    Output:
        1) lemmed: text that has been tokenized, had stop-words removed, and lemmetized
    """
    
    # Normalize Text
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    
    # tokenize text
    words = word_tokenize(text)
    
    # remove stopwords
    words = [w for w in words if w not in stopwords.words('english')]
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w,pos='n').strip() for w in words]
    lemmed = [lemmatizer.lemmatize(w,pos='v').strip() for w in lemmed]

    return lemmed


def build_model():
    """
    Function to build ML pipeline model
    
    Arguments: None
    
    Output: ML pipeline model
    """
    
    pipeline = Pipeline([
    
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }
    
    model = GridSearchCV(pipeline,param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to fit the test data to the NLP model and evaluate performance
    
    Arguments:
        1) model: NLP model
        2) X_test: test data features
        3) Y_test: test data targets
        4) category_names: label names
    
    Output: None
    """
    
    # predict
    y_pred = model.predict(X_test)

    cols = Y_test.columns
    
    # print classification report
    for i in range(len(cols)):
        print('Column:  ',cols[i])
        print('Accuracy: %.2f' % accuracy_score(Y_test[cols[i]],y_pred[:,i]))
        print(classification_report(Y_test[cols[i]], y_pred[:,i]))
        print('')

        
def save_model(model, model_filepath):
    """
    Function to save model as pickel file
    
    Arguments:
        1) model: final ML model
        2) model_filepath: file path to save ML model
    
    Output: None
    """

    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        start = time.time()
        model.fit(X_train, Y_train)
        print('Run time: {}'.format(time.time()-start))
        
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