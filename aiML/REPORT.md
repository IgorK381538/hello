# Отчет по лабораторной работе
## по курсу "Искусственый интеллект"
### Студент: Корнев Игорь 8о-304б

# Задание:   
Вариант 3. Реализовать алгоритм выявляющий взаимосвязанные сообщения на языке Python. Подобрать или создать датасет и обучить модель. Продемонстрировать зависимость качества кластеризации от объема, качества выборки и числа кластеров. Продемонстрировать работу вашего алгоритма. Обосновать выбор данного алгоритма машинного обучения. Построить облако слов для центров кластеров(wordcloud).   

# Решение:   
``` Python

import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpld3


dataset = pd.read_csv(r"data.csv")
dataset = dataset['keyword']

i = 0
for line in dataset:
    line = re.sub(r'(\<(/?[^>]+)>)', ' ', line)
    line = re.sub('[^а-яА-Я ]', '', line)
    dataset[i] = line
    i += 1

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("russian")

def token_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

token_and_stem('sad')

def token_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filtered_tokens.append(token)
    return filtered_tokens

totalvocab_stem = []
totalvocab_token = []
for i in dataset:
    allwords_stemmed = token_and_stem(i)
    totalvocab_stem.extend(allwords_stemmed)
    
    allwords_tokenized = token_only(i)
    totalvocab_token.extend(allwords_tokenized)

stopwords = nltk.corpus.stopwords.words('russian')
stopwords.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'к', 'на'])

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

n_featur=200000
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000,
                                 min_df=0.01, stop_words=stopwords,
                                 use_idf=True, tokenizer=token_and_stem, ngram_range=(1,3))
from IPython import get_ipython
ipython = get_ipython()

tfidf_matrix = tfidf_vectorizer.fit_transform(dataset)
num_clusters = 15

from sklearn.cluster import KMeans

km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
idx = km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
clusterkm = km.labels_.tolist()

dataout = dataset.tolist()
frame = pd.DataFrame(dataout, index = [clusterkm])
out = { 'words': dataout, '№ of cluster': clusterkm }
resultDF = pd.DataFrame(out, columns = ['words', '№ of cluster'])
resultDF.to_csv("clust.txt", sep='\t', encoding='utf-8')

```
# Пример вывода:   
```
    words	№ of cluster
345	снятое колесо 	6
346	под колесами 	6
347	интернет колесо 	6
348	сумка на колесах 	6
349	колесо года 	6
350	динозавры  	14
351	про динозавров 	14
352	динозавры мультфильмы 	14
353	динозавры фильмы 	14
354	динозавры мультики 	14
```

# Вывод   
Для анализа я взял случайнные подборы запросов с wordstat.yandex.ru, и проверил как поведят себя мой алгоритм.
Для начала я считал данные и нормализовал их используя стеммер Портера. Затем создаю матрицу весов TF-IDF. Затем применяю метод k-средних, т.к. он самый простой в реализаций и сгруппировываем в файл вывода. Результат весьма спорный и не претендует на истину, но чем больше мы берем кол-во кластеров, тем точнее получается результат.   
Источник информации:   
https://pythondigest.ru/view/32057/   
http://www.machinelearning.ru/wiki/images/archive/2/28/20150427184336%21Voron-ML-Clustering-slides.pdf   
http://www.nltk.org   
https://habr.com/company/mailru/blog/339496/
