# Disaster-Response-Pipelines
Udacity project


# This project includes the following files: 

ETL Pipeline Preparation.ipynb
ML Pipeline Preparation.ipynb
categories.csv dataset
messages.csv dataset
Then a workspace folder containing the .py files and the web app:
A-App : the run.py model
B-Data: datasets for messages and categories + .py file for the data preparation
C-Models: contains the train_classifier.py file



# Libraries to be installed

1- for the data preparation:

import pandas as pd
import numpy as np
from sqlalchemy import create_engine


2- for the Mashine Learning pipelines 

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix #not used
from sklearn.metrics import accuracy_score #not used
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')



