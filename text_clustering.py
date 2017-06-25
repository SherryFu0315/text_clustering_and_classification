import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
import os
from sklearn import feature_extraction
# import matplotlib.pyplot as plt
# import matplotlib as mpl

''' load data '''
def get_clean_list(texts, parser):
    result = []
    for text in texts:
        text = BeautifulSoup(text, parser).get_text()
        result.append(text)
    return result

titles = open('./data/title_list.txt').read().split('\n')
titles = titles[:100]

synopses_wiki = open('./data/synopses_list_wiki.txt').read().split('\n BREAKS HERE')
synopses_wiki = synopses_wiki[:100]
synopses_wiki = get_clean_list(synopses_wiki, 'html.parser')

genres = open('./data/genres_list.txt').read().split('\n')
genres = genres[:100]

synopses_imdb = open('./data/synopses_list_imdb.txt').read().split('\n BREAKS HERE')
synopses_imdb = synopses_imdb[:100]
synopses_imdb = get_clean_list(synopses_imdb, 'html.parser')

synopses = [wiki + imdb for wiki, imdb in zip(synopses_wiki, synopses_imdb)]

ranks = [i for i in range(len(titles))]

stopwords = nltk.corpus.stopwords.words('english')

'''tokenize and stemm'''
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

# only tokenize
def tokenize_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

totalvocab_stemmed = []
totalvocab_tokenized = []
for synopsis in synopses:
    totalvocab_stemmed.extend(tokenize_and_stem(synopsis))
    totalvocab_tokenized.extend(tokenize_only(synopsis))

# create DataFrame
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)

''' tf-idf: find document similarity'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0.2, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))
tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)

print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()

dist = 1 - cosine_similarity(tfidf_matrix)

'''k-means clustering'''
# from sklearn.cluster import KMeans
#
# num_clusters = 5
#
# km = KMeans(n_clusters=num_clusters)
# km.fit(tfidf_matrix)
#
# clusters = km.labels_.tolist()
#
# # using pickle to store intermediate result
# from sklearn.externals import joblib
#
# joblib.dump(km, 'doc_cluster.pkl')
# km = joblib.load('doc_cluster.pkl')
# clusters = km.labels_.tolist()
#
# '''transfor data to pandas DataFrame'''
# films = {'title': titles, 'rank': ranks, 'synopsis': synopses, 'cluster': clusters, 'genre': genres}
# frame = pd.DataFrame(films, index=[clusters] , columns=['rank', 'title', 'cluster', 'genre'])
# print(frame['cluster'].value_counts())
#
# grouped = frame['rank'].groupby(frame['cluster'])
# print(grouped.mean())
#
# # print clusters info
# print("Top terms per cluster:")
# order_centroids = km.cluster_centers_.argsort()[:, ::-1]
# for i in range(num_clusters):
#     print("Cluster %d words:" % i, end='')
#     for ind in order_centroids[i, :6]:
#         print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
#     print()
#     print("Cluster %d titles:" % i, end='')
#     for title in frame.ix[i]['title'].values.tolist():
#         print(' %s,' % title, end='')
#     print()
#
#
# # multidimensional scaling
#
# from sklearn.manifold import MDS
#
# mds = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
# pos = mds.fit_transform(dist)
# xs, ys = pos[:, 0], pos[:, 1]
#
# #strip NNP and NNPS
from nltk.tag import pos_tag
#
# def strip_POS(text):
#     tagged = pos_tag(text)
#     non_propernouns = [word for word, pos in tagged if pos != 'NNP' and pos != 'NNPS']
#     return non_propernouns
#
# '''Visualize clusters'''
# cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
# cluster_names = {0: 'Family, home, war',
#                  1: 'Police, killed, murders',
#                  2: 'Father, New York, brothers',
#                  3: 'Dance, singing, love',
#                  4: 'Killed, soldiers, captain'}
#
# df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))
# groups = df.groupby('label')
#
# fig, ax = plt.subplots(figsize=(17, 9))
# ax.margins(0.05)
#
# for name, group in groups:
#     ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')
#     ax.set_aspect('auto')
#     ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
#     ax.tick_params(axis='y', which='both', bottom='off', top='off', labelbottom='off')
#
# ax.legend(numpoints=1)
#
# for i in range(len(df)):
#     ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

# plt.show()

'''Hierarchical clustering'''
from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist)

# fig, ax = plt.subplots(figsize=(15, 20))
# ax = dendrogram(linkage_matrix, orientation='right', labels=titles)
#
# plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
#
# plt.tight_layout()
# plt.show()
# plt.close()

'''Latent Dirichlet Allocation'''
import string

def strip_propers(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.islower()]
    return ''.join([' ' + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()

def strip_proppers_POS(text):
    tagged = pos_tag(text.split())
    non_propernons = [word for word, pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernons

from gensim import corpora, models, similarities

preprocess = [strip_propers(doc) for doc in synopses]
tokenized_text = [tokenize_and_stem(text) for text in preprocess]
texts = [[word for word in text if word not in stopwords] for text in tokenized_text]

print(len(texts[0]))

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=1, no_above=0.8)
corpus = [dictionary.doc2bow(text) for text in texts]
print(len(corpus))

lda = models.LdaModel(corpus, num_topics=5, id2word=dictionary, update_every=5, chunksize=10000, passes=100)

topics = lda.print_topics(5, num_words=20)

print(topics)

topics_matrix = lda.show_topics(formatted=False, num_words=20)

print(topics_matrix)