import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
from sklearn.feature_extraction import image
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from IPython import get_ipython
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from pymystem3 import Mystem

print('Выбирите стеммер 1 - Porter, 2 - Snowball ')
j = int(input())
if j == 2:
    stemmer = SnowballStemmer("russian")
elif j == 1:
    stemmer = PorterStemmer()

dataset = pd.read_csv(r"data.csv")
dataset = dataset['keyword']

i = 0
while i < len(dataset):
    dataset[i] = re.sub(r'(\<(/?[^>]+)>)', ' ', dataset[i])
    dataset[i] = re.sub('[^а-яА-Я ]', '', dataset[i])
    i += 1

def tns(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def to(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filtered_tokens.append(token)
    return filtered_tokens

totalvocab_stem = []
totalvocab_token = []
for i in dataset:
    allwords_stemmed = tns(i)
    totalvocab_stem.extend(allwords_stemmed)
    
    allwords_tokenized = to(i)
    totalvocab_token.extend(allwords_tokenized)

stopwords = nltk.corpus.stopwords.words('russian')
stopwords.extend([])

n_featur=200000
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000,
                                 min_df=0.01, stop_words=stopwords,
                                 use_idf=True, tokenizer=tns, ngram_range=(1,3))
ipython = get_ipython()

tfidf_matrix = tfidf_vectorizer.fit_transform(dataset)
num_clusters = 15

km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
idx = km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

mbk  = MiniBatchKMeans(init='random', n_clusters=num_clusters)
mbk.fit_transform(tfidf_matrix)
miniclusters = mbk.labels_.tolist()

dataout = dataset.tolist()
frame = pd.DataFrame(dataout, index = [clusters])
out = { 'words': dataout, '№ of cluster': clusters }
resultDF = pd.DataFrame(out, columns = ['words', '№ of cluster'])
resultDF.to_csv("clustkm.txt", sep='\t', encoding='utf-8')

dataout = dataset.tolist()
frame = pd.DataFrame(dataout, index = [miniclusters])
out = { 'words': dataout, '№ of cluster': miniclusters }
resultDF = pd.DataFrame(out, columns = ['words', '№ of cluster'])
resultDF.to_csv("clustmkm.txt", sep='\t', encoding='utf-8')


