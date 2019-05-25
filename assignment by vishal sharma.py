#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:07:06 2019

@author: vishal
"""

import pandas as pd 

import plotly
import matplotlib.pyplot as plt
from textblob import TextBlob 
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
%matplotlib inline
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
data = pd.read_csv("transcripts.csv") 
# Preview the first 5 lines of the loaded data 
data.head()
# Function for getting speech names for url 
def topicheading():
    newurllist = []
    i =0
    while i<len(data.url):
        e = data.url[i].replace('https://www.ted.com/talks/','')
        newurllist.append(e.replace('\n',''))
        i+=1
    return newurllist
# for Testing Purpose
topicheading()[1:10]
len(topicheading())
newtranscript= data.transcript

def preprocess():
    ReviewText = newtranscript.replace("\'re", " are")
    ReviewText = newtranscript.replace("\'ll", " will")
    ReviewText = newtranscript.replace("\'s", " is")
    ReviewText = newtranscript.replace("\'t", " not")     
    return ReviewText



data['polarity'] = data['transcript'].map(lambda text: TextBlob(text).sentiment.polarity)

# testing data['polarity']
data['polarity'].head()

data['review_len'] = data['transcript'].astype(str).apply(len)
data['word_count'] = data['transcript'].apply(lambda x: len(str(x).split()))

# Code for rpint highest neutral sentiment 
print('5 random sentence with the highest neutral sentiment polarity: \n')
cl = data.loc[data.polarity == 0, ['transcript']].sample(5).values
for k in cl:
    print(k[0])
# graph for  sentiment polarity distribution
data['polarity'].plot(
    kind='hist',
    bins=50,
    xTitle='polarity',
    linecolor='black',
    yTitle='count',
    title='Sentiment Polarity Distribution')
#graph for SPEECH Text Length Distribution
data['review_len'].plot(
    kind='hist',
    bins=100,
    xTitle='review length',
    linecolor='black',
    yTitle='count',
    title='SPEECH Text Length Distribution')
#graph for SPEECH Text WORD COUNT Distribution
data['word_count'].plot(
    kind='hist',
    bins=100,
    xTitle='word count',
    linecolor='black',
    yTitle='count',
    title='SPEECH Text Word Count Distribution')
# code for drawing bar chart for Top 20 words in review before removing stop words
corpus = data.transcript
def draw_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = draw_top_n_words(data['transcript'], 20)
for word, freq in common_words:
    print(word, freq)
df1 = pd.DataFrame(common_words, columns = ['transcript' , 'count'])
df1.groupby('transcript').sum()['count'].sort_values(ascending=False).plot(
kind='bar', title='Top 20 words in review before removing stop words')


# code for drawing bar chart for Top 20 words in review after removing stop words
def draw_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = draw_top_n_words(data['transcript'], 20)
for word, freq in common_words:
    print(word, freq)
df2 = pd.DataFrame(common_words, columns = ['transcript' , 'count'])
df2.groupby('transcript').sum()['count'].sort_values(ascending=False).plot(
kind='bar', title='Top 20 words in review after removing stop words')


# code for drawing bar chart for Top 20 trigrams in review after removing stop words 
def draw_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = draw_top_n_trigram(data['transcript'], 20)
for word, freq in common_words:
    print(word, freq)
df6 = pd.DataFrame(common_words, columns = ['transcript' , 'count'])
df6.groupby('transcript').sum()['count'].sort_values(ascending=False).plot(
kind='bar', title='Top 20 trigrams in review after removing stop words')


# code for drawing bar chart for top 20 part-of-speech tagging for review corpus
blob = TextBlob(str(data['transcript']))
pos_df = pd.DataFrame(blob.tags, columns = ['word' , 'pos'])
pos_df = pos_df.pos.value_counts()[:20]
pos_df.plot(
    kind='bar',
title='Top 20 Part-of-speech tagging for review corpus')

