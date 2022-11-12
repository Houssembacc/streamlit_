from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from wordcloud import WordCloud
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import contractions
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

tokenizer = RegexpTokenizer(r'\w+')


def tokenize_text(text):
    text_processed = " ".join(tokenizer.tokenize(text))
    return text_processed


# import en_core_web_sm

# nlp = en_core_web_sm.load(disable=['parser', 'tagger', 'ner'])
lemmatizer = WordNetLemmatizer()


def lemmatize_text(text):
    tokens_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    lemmatized_text_list = list()

    for word, tag in tokens_tagged:
        if tag.startswith('J'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,
                                                             'a'))  # Lemmatise adjectives. Not doing anything since we remove all adjective
        elif tag.startswith('V'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word, 'v'))  # Lemmatise verbs
        elif tag.startswith('N'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word, 'n'))  # Lemmatise nouns
        elif tag.startswith('R'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word, 'r'))  # Lemmatise adverbs
        else:
            lemmatized_text_list.append(
                lemmatizer.lemmatize(word))  # If no tags has been found, perform a non specific lemmatisation

    return " ".join(lemmatized_text_list)


def normalize_text(text):
    return " ".join([word.lower() for word in text.split()])


def contraction_text(text):
    return contractions.fix(text)


negative_words = ['not', 'no', 'never', 'nor', 'hardly', 'barely']
negative_prefix = "NOT_"


def get_negative_token(text):
    tokens = text.split()
    negative_idx = [i + 1 for i in range(len(tokens) - 1) if tokens[i] in negative_words]
    for idx in negative_idx:
        if idx < len(tokens):
            tokens[idx] = negative_prefix + tokens[idx]

    tokens = [token for i, token in enumerate(tokens) if i + 1 not in negative_idx]

    return " ".join(tokens)


from spacy.lang.en.stop_words import STOP_WORDS


def remove_stopwords(text):
    english_stopwords = stopwords.words("english") + list(STOP_WORDS) + ["tell", "restaurant"]

    return " ".join([word for word in text.split() if word not in english_stopwords])


def preprocess_text(text):
    # Tokenize review
    text = tokenize_text(text)

    # Lemmatize review
    text = lemmatize_text(text)

    # Normalize review
    text = normalize_text(text)

    # Remove contractions
    text = contraction_text(text)

    # Get negative tokens
    text = get_negative_token(text)

    # Remove stopwords
    text = remove_stopwords(text)

    return text


print('The scikit-learn version is {}.'.format(sklearn.__version__))
import pickle

file_name = "vectorizer.bin"
with (open(file_name, "rb")) as f:
    while True:
        try:
            vectorizer = pickle.load(f)
        except EOFError:
            break
print(vectorizer)

file_name2 = "model_nmf.bin"
with (open(file_name2, "rb")) as f:
    while True:
        try:
            nmf_model = pickle.load(f)
        except EOFError:
            break
print(nmf_model)

text = preprocess_text("I am good")
print(text)

liste = []
liste.append("I am good")
x = vectorizer.transform(liste)
y = nmf_model.transform(x)
print(y)
ind = []
argsort = np.argsort(y, axis=1)
print(argsort)
for i in range(2):
    ind.append(argsort[0][len(argsort[0]) - (i + 1)])
print(ind)

# Fonction de prediction

from textblob import TextBlob
import pandas as pd

interpretation_topi = {
    'topic0': 'TICKET PAYEMENT',
    'topic1': 'OVERLOADED RESTAURANT',
    'topic2': 'PIZZA/DELIVERY',
    'topic3': 'PRICES/ONLINE-ORDERING',
    'topic4': 'FOOD/SERVICE',
    'topic5': 'TABLE/BOTHERING',
    'topic6': 'SERVICE',
    'topic7': 'TEMPS D ATTENTE',
    'topic8': 'CHICKEN',
    'topic9': 'BEER/BAR',
    'topic10': 'BAD PLACE TO EAT',
    'topic11': 'SUSHI',
    'topic12': 'LUNCH/SANDWICH',
    'topic13': 'AMBIANCE',
    'topic14': 'WAITER'
}


def prediction_text(vectorizer, nmf_model, n_topic, text):
    topic_list = ['TICKET PAYEMENT', 'OVERLOADED RESTAURANT', 'PIZZA/DELIVERY', 'PRICES/ONLINE-ORDERING',
                  'FOOD/SERVICE', 'TABLE/BOTHERING', 'SERVICE', 'TEMPS D ATTENTE', 'CHICKEN', 'BEER/BAR',
                  'BAD PLACE TO EAT', 'SUSHI',
                  'LUNCH/SANDWICH',
                  'AMBIANCE',
                  'WAITER']

    txt = TextBlob(text)
    polarity = txt.polarity
    liste = []
    text_cleaned = preprocess_text(text)
    liste.append(text_cleaned)
    idx_l = []
    if (polarity < 0):
        text_cleaned = preprocess_text(text)
        liste.append(text_cleaned)
        x = vectorizer.transform(liste)
        y = nmf_model.transform(x)
        ind = []
        argsort = np.argsort(y, axis=1)
        print(argsort)
        for i in range(n_topic):
            ind.append(argsort[0][len(argsort[0]) - (i + 1)])

        for i in ind:
            idx_l.append(topic_list[i])

    return polarity, idx_l

######################################################### Partie Streamlit (Front) ##################################################
from PIL import Image

st.title("Detection de sujet d'insatisfaction !")
image = Image.open('étapes.png')

st.image(image, caption='')
text_value= st.text_input("Entrez un texte:")
number = st.number_input('Insert a number of topics',min_value=1,max_value=15,step=1)
#st.write('The current number is ', number)
if st.button("Detecter le sujet d'insatisfaction"):
    p,l=prediction_text(vectorizer, nmf_model, number, text_value)
    if (p<0):  
        st.write('NEG:',p)
        st.write('Topics are :',l)
    if (p>0):
        st.write('Cet avis est positif !')
    if (p=0):
        st.write('Cet avis est neutre !')
        
