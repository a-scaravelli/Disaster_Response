import sys
import pandas as pd
import sqlite3
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    imports the df from a .db file
    creates X and y
    
    INPUT
        database_filepath - string containing the name of the .db file
        
    OUTPUT
        X        - Features of the df
        y        - target variables
        y.keys() - colums names
    '''
    conn = sqlite3.connect(database_filepath)
    cur = conn.cursor()
    df = pd.read_sql("SELECT * FROM df", con=conn)

    X = df['message'].values
    y = df.drop(['id','message','original','genre','related'],axis = 1).copy()
    
    return X,y,y.keys()


def tokenize(text):
    '''
    tokenize, lemmatize, lowered and stripped the text
    
    INPUT
        text - string to be tokenized
    
    OUTPUT
        clean_tokens - string cleaned
    
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    
    '''
    creates a pipeline with three components:
        Vectorizer
        tfidf
        multioutputclassifier
    
    initiate params for Gridsearch
    
    initiate the GridSearch object
    
    OUTPUT
        cv - model 
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {  
        #'clf__estimator__n_estimators': [25, 50, 100],
        #'clf__estimator__max_depth': [ 25, 50] 
        'clf__estimator__max_depth': [50]
    }
    
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)


    
    return cv
    
def evaluate_model(model, X_test, y_test, category_names):
    '''
    INPUT 
        pipeline: The model that is to be evaluated
        X_test: Input features, testing set
        y_test: Label features, testing set
        category_names: List of the categories 
    OUTPUT
        This method does not specifically return any data to its calling method.
        However, it prints out the precision, recall and f1-score
    '''
    
    y_pred_cv = model.predict(X_test)
    print(classification_report(y_test, y_pred_cv, target_names=category_names))


def save_model(model, model_filepath):
    '''
    Saves the model to disk
    INPUT 
        model: The model to be saved
        model_filepath: Filepath for where the model is to be saved
    OUTPUT
        While there is no specific item that is returned to its calling method, this method will save the model as a pickle file.
    '''  
    temp_pickle = open(model_filepath, 'wb')
    pickle.dump(model, temp_pickle)
    temp_pickle.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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