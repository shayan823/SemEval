# In[]
import sys
import ast
import re
from itertools import groupby
import numpy as np
import collections
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import logging
import pickle
import pandas as pd
import json
import os
from sklearn.preprocessing import LabelBinarizer

# In[]

#logging setup
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

def read(directory,file):
    path = directory + '/' + file
    with open(path) as json_file:
        data = json.load(json_file)
        json_file.close()
    
    text = data['text']
    return text

class feature_extractor(object):
    
    def __init__(self,directory,embedding_size=512):

        #store records
        labels = []
        tokens = []
        maxsentlen = 0
        maxdoclen = 0
        

        data_train = pd.read_csv('train.csv', sep=',')

        #process json one line at a time
        for idx in range(data_train.shape[0])[:1000]:
            text = read(directory,data_train.Article_Id[idx])
                        
            #process text
            text = text.lower()
            text = re.sub("'", '', text)
            text = re.sub("\.{2,}", '.', text)
            text = re.sub('[^\w_|\.|\?|!]+', ' ', text)
            text = re.sub('\.', ' . ', text)
            text = re.sub('\?', ' ? ', text)
            text = re.sub('!', ' ! ', text)

            #tokenize
            text = text.split()
                
            #drop empty reviews
            if len(text) == 0:
                continue

            #split into sentences
            sentences = []
            sentence = []
            for t in text:
                if t not in ['.','!','?']:
                    sentence.append(t)
                else:
                    sentence.append(t)
                    sentences.append(sentence)
                    if len(sentence) > maxsentlen:
                        maxsentlen = len(sentence)
                    sentence = []
            if len(sentence) > 0:
                sentences.append(sentence)
                
                #add split sentences to tokens
            tokens.append(sentences)
            if len(sentences) > maxdoclen:
                maxdoclen = len(sentences)
                
            #add label 
            labels.append(data_train.bias[idx])
                
        print ('\nsaved %i records' % len(tokens))
                
        #generate Word2Vec embeddings
        print ("generating word2vec embeddings")

        #used all processed raw text to train word2vec
        self.allsents = [sent for doc in tokens for sent in doc]

        self.model = Word2Vec(self.allsents, min_count=100, size=embedding_size, workers=4, iter=5)
        self.model.init_sims(replace=True)
        
        #save all word embeddings to matrix
        print ("saving word vectors to matrix")
       # self.vocab = np.zeros((len(self.model.wv.vocab)+1,embedding_size))
        word2id = {}
                
        GLOVE_DIR = "."
        embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        
        print('Total %s word vectors.' % len(embeddings_index))
        
        
        #first row of embedding matrix isn't used so that 0 can be masked
        
        for key,val in self.model.wv.vocab.items():
            idx = val.__dict__['index'] + 1
            self.vocab[idx,:] = self.model[key]
            word2id[key] = idx
            
        #normalize embeddingso
        self.vocab -= self.vocab.mean()
        self.vocab /= (self.vocab.std()*2.5)

        #reset first row to 0
        self.vocab[0,:] = np.zeros((embedding_size))

        #add additional word embedding for unknown words
        self.vocab = np.concatenate((self.vocab, np.random.rand(1,embedding_size)))

        #index for unknown words
        unk = len(self.vocab)-1

        #convert words to word indicies
        print ("converting words to indices")
        self.data = {}
        for idx,doc in enumerate(tokens):
            sys.stdout.write('processing %i of %i records       \r' % (idx+1,len(tokens)))
            sys.stdout.flush()
            dic = {}
            dic['label'] = labels[idx]
            dic['text'] = doc
            indicies = []
            for sent in doc:
                indicies.append([word2id[word] if word in word2id else unk for word in sent])
            dic['idx'] = indicies
            self.data[idx] = dic
        
    def get_embeddings(self):
        #print(self.vocab.shape)
        return self.vocab

    def get_data(self):
       #print(len(self.data))
        return self.data
    
if __name__ == "__main__": 

    #get json filepath
    #args = (sys.argv)
    #if len(args) != 2:
     #   raise Exception("Usage: python feature_extraction.py <path to Yelp json file>")
    json_path = 'articles-training-20180831'
    
    #process json
    fe = feature_extractor(json_path,512)
    vocab = fe.get_embeddings()
    data = fe.get_data()
    
    #create directory for saved model
    if not os.path.exists('./data'):
        os.makedirs('./data')
    
    #save vocab matrix and processed documents
    np.save('./data/yelp_embeddings',vocab)
    with open('./data/yelp_data.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
