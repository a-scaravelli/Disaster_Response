# Disaster Response Pipeline Project.

## Project motivation:
Figure eight provided data related to messages that they have received during disasters/emergencies. The data was classified in 36 different categories.
in this project a model is trained in order to classify a new message.

## Library used
    NumPy
    Pandas
    Matplotlib
    Json
    Plotly
    Nltk
    Flask
    Sklearn
    Sqlalchemy
    Picke
    Re

## Project development:
    1. Data cleaning Pipeline: data ingestion and data cleaning (../data/process_data.py)
    2. training of the model and preiction were made (../models/train_classifier.py)
    3. Web app that can be used to make predictions and Visualizations (provided by Udacity)

## Instructions:
    run the followings commands in project directory:
    
    ELT: python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    Model creation: python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    app: python run.py
    
## file structure
    /app folder for web app
    /data folder containing the datasets and the data cleaning.py
    /models model.py and pickle file