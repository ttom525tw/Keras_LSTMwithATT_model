# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(1337) 
import pickle
import pyodbc
import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from Lstm import  train_lstm
cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=your server ip;DATABASE=dataset name;UID=your id;PWD=your passwords;')
cursor = cnxn.cursor()
query = "SELECT [review_content],[y]FROM [movie].[dbo].[reviews]"
df = pd.read_sql(query, cnxn)
def sentence_to_wordlist(sentence,remove_stopwords=False):
    words=sentence.split()
    
    return(words)

def word_split(i):
    try:
        sentences =[]
        sentences.append(sentence_to_wordlist(i))
        return(sentences)
    except:
        print('nope')

sentences=[]
y=[]

for index, row in df.iterrows():
    try:
        #print(sentence_to_wordlist(row['review_content']))
        
        sentences+=word_split(row['review_content'])
        y.append(row['y'])
    except:
        print('no')

vocab_dim = 300  
maxlen = 50 
batch_size = 1024
n_epoch = 100
input_length = 50
def text_to_index_array(p_new_dic, p_sen):  
    new_sentences = []
    for sen in p_sen:
        new_sen = []
        for word in sen:
            try:
                new_sen.append(p_new_dic[word])  
            except:
                new_sen.append(0)  
        new_sentences.append(new_sen)

    return np.array(new_sentences)

sentences,y=shuffle(sentences,y)
#process
f = open("wordlist.pkl", 'rb')
index_dict = pickle.load(f)  
word_vectors = pickle.load(f)  
new_dic = index_dict
nb_classes=3
n_symbols = len(index_dict) + 1 
embedding_weights = np.zeros((n_symbols, 300))  
for w, index in index_dict.items():  
    embedding_weights[index, :] = word_vectors[w]  
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(sentences,y,test_size=0.2)
X_train = text_to_index_array(new_dic, X_train_l)
X_test = text_to_index_array(new_dic, X_test_l)
print ("traindata shape： ", X_train.shape)
print ("testshape： ", X_test.shape)
y_train1 = np.array(y_train_l)  
y_test1 = np.array(y_test_l)
y_train = np_utils.to_categorical(y_train1,nb_classes,dtype='float32')
y_test = np_utils.to_categorical(y_test1,nb_classes,dtype='float32')
print ("traindata shape： ", y_train.shape)
print ("testshape： ", y_test.shape)
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)




if __name__ == "__main__":
    train_lstm(n_symbols, embedding_weights, X_train, y_train, X_test, y_test,vocab_dim,input_length,batch_size,n_epoch)








