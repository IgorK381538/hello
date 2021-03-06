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
print('Введите число кластеров')
nc = int(input())

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

km = KMeans(n_clusters=nc)
km.fit(tfidf_matrix)
idx = km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

mbk  = MiniBatchKMeans(init='random', n_clusters=nc)
mbk.fit_transform(tfidf_matrix)
miniclusters = mbk.labels_.tolist()

dataout = dataset.tolist()
frame = pd.DataFrame(dataout, index = [clusters])
out = { 'words': dataout, '№ of cluster': clusters }
resultDF = pd.DataFrame(out, columns = ['words', '№ of cluster'])
resultDF.to_csv("clustkm2.txt", sep='\t', encoding='utf-8')

dataout = dataset.tolist()
frame = pd.DataFrame(dataout, index = [miniclusters])
out = { 'words': dataout, '№ of cluster': miniclusters }
resultDF = pd.DataFrame(out, columns = ['words', '№ of cluster'])
resultDF.to_csv("clustmkm.txt", sep='\t', encoding='utf-8')
```
# Пример вывода для K-means:   
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
# Пример вывода для MiniBatchKMeans:  
```
642	картошка с мясом в духовке рецепт 	5
643	жарка картошки 	5
644	картошка с мясом фото 	5
645	тесто с картошкой 	5
646	сажу картошку 	5
647	котиков 	13
648	котики 	13
649	котиков котики 	13
650	котики картинки 	13
651	картинки котиков 	13
652	котики картинки котиков 	13
```

# Вывод   
Для анализа я взял случайнные подборы запросов с wordstat.yandex.ru, и проверил как поведят себя моя программа.
Для начала я считал данные и нормализовал их используя стеммер Портера или его дальнейшее развитие Snowball (snowball естественно работает лучше). Затем создаю матрицу весов TF-IDF. Затем применяю метод Kmeans и MiniBatchKMeans, т.к. они простые в реализации и сгруппировываем в файл вывода. Стоит отметить что MiniBatchKMeans работает значительно быстрее, но в той же мере с большей погрешностью, также для повышения точности, необходимо выбирать большее число кластеров (я взял 15). В итоге, глядя на вывод программы, можно с уверенностью сказать, что она кластеризует весьма точно.
