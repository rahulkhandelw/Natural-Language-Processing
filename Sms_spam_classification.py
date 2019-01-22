# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:32:19 2019

@author: rahkhand
"""

import os
import pandas as pd
print(os.getcwd())

os.chdir('D:\\DataS\\NLP\\Inputs')

data = pd.read_csv('sms_spam.csv')

data.columns

data.info()

data.describe()

data['type'].unique()

data['type'].value_counts(normalize = True)

type(data['text'])


###Data Loaded in Data Frame###
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#data['words'] = word_tokenize(data['text'])
StWords = set (stopwords.words('english'))
data['words'] =data['text'].apply(nltk.word_tokenize)

def removeStWords(words):
    flt_words = []
    for w in words:
        if w not in StWords:
            flt_words.append(w)
            
    return flt_words

ps = PorterStemmer()
def Stemmer(words):
    stemmedWord = []
    for w in words:
        stemmedWord.append(ps.stem(w))
    
    return stemmedWord


def createFilterdSentence(words):
    str1 = ''
    str1 =  ' '.join(words)
    return str1
    
        
data['flt_words'] = data['words'].apply(removeStWords)

data['tokens'] = data['flt_words'].apply(Stemmer)

data['newText'] = data['tokens'].apply(createFilterdSentence)

del data['words']
del data['flt_words']
del data['text']

all_words = ' '.join(data['newText'])

all_words_tokenize = word_tokenize(all_words)

del data['newText']

#all_words_tokenize = set(all_words_tokenize)

#all_words_tokenize = list(all_words_tokenize)

flt_all_words_tokenize = []
for w in all_words_tokenize:
    if w not in StWords:
        flt_all_words_tokenize.append(w)
        
bagOfWords = flt_all_words_tokenize



bagOfWords = nltk.FreqDist(bagOfWords)

sortedBag = bagOfWords.most_common()

sortedBag[20:3000]

features = []
for i in range(21, 3000):
    features.append(sortedBag[i][0])
    


features

def find_features(textData):
#    print(textdata)
    words = set(textData)
    sms_features = {}
    for w in features:
        sms_features[w] = (w in words)
        
    return sms_features 
 

data['features']    = data['tokens'].apply(find_features)



data.info()

del data['tokens']

def createLabeledFeatures(obs):
    
    return (obs['features'],obs['type'])

data['featureset'] = data.apply(createLabeledFeatures, axis =1 )
    
#data['featureset'] = (data['features'],data['type'])

train_set = data[:5000]
test_set = data[5000:]



test_set.info()
print(train_set['featureset'][0])
classifier = nltk.classify.NaiveBayesClassifier.train(train_set['featureset'])

print("NB Classifier Accuracy", nltk.classify.accuracy(classifier,test_set['featureset']))



    
