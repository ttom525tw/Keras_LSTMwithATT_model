# -*- coding: utf-8 -*-
import numpy as np
import pickle
import pyodbc
import pandas as pd

from keras.preprocessing import sequence
from double_BiLstm import train_double_BiLstm
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
np.random.seed(1337)  # For Reproducibility

#設定資料庫連接
cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=your server ip;DATABASE=dataset name;UID=your id;PWD=your passwords;')
cursor = cnxn.cursor()
query_t = "SELECT [review_content],[y]FROM [movie].[dbo].[reviews]"
df_t = pd.read_sql(query_t, cnxn)

#讀取review資料集
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

for index, row in df_t.iterrows():
    try:
        
        sentences+=word_split(row['review_content'])
        y.append(row['y'])
    except:
        print('no')


#讀取提取的emotion_words 資料
query_e= "SELECT [emotion_words]FROM [movie].[dbo].[emotion_data]"
df_e = pd.read_sql(query_e, cnxn)

def word_split_e(i):
    try:
        sentences =[]
        emotions=i.replace("[",'').replace(']','').replace("'",' ')
        
        sentences.append(sentence_to_wordlist(emotions))
        return(sentences)
    except:
        print('nope')

emotion_words=[]
for index, row in df_e.iterrows():
    try:
        
        emotion_words+=word_split_e(row['emotion_words'])
        
    except:
        print('no')
vocab_dim = 300 
maxlen,maxlen_e = 50,50
batch_size = 1024
n_epoch = 50
input_length,input_length_e = 50,50

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
#長短期記憶模型建置

#讀取單字向量 
f = open("wordlist.pkl", 'rb')
index_dict = pickle.load(f)  
word_vectors = pickle.load(f)  
new_dic = index_dict

nb_classes=3
n_symbols = len(index_dict) + 1 
embedding_weights = np.zeros((n_symbols, 300))  

for w, index in index_dict.items():  
    embedding_weights[index, :] = word_vectors[w] 


#loading data and reshape for model
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(sentences,y,test_size=0.2)
X_train_le, X_test_le = train_test_split(emotion_words,test_size=0.2)
X_train = text_to_index_array(new_dic, X_train_l)
X_test = text_to_index_array(new_dic, X_test_l)
X_train_e = text_to_index_array(new_dic, X_train_le)
X_test_e = text_to_index_array(new_dic, X_test_le)
print ("traindata shape： ", X_train.shape)
print ("testshape： ", X_test.shape)
print ("traindata shape： ", X_train_e.shape)
print ("testshape： ", X_test_e.shape)
y_train1 = np.array(y_train_l)  
y_test1 = np.array(y_test_l)
y_train = np_utils.to_categorical(y_train1,nb_classes,dtype='float32')
y_test = np_utils.to_categorical(y_test1,nb_classes,dtype='float32')
print ("traindata shape： ", y_train.shape)
print ("testshape： ", y_test.shape)
print('Pad sequences (samples x time)')

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
X_train_e = sequence.pad_sequences(X_train_e, maxlen=maxlen_e)
X_test_e = sequence.pad_sequences(X_test_e, maxlen=maxlen_e)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('X_train shape:', X_train_e.shape)
print('X_test shape:', X_test_e.shape)

if __name__ == "__main__":
    train_double_BiLstm(n_symbols, embedding_weights, X_train, y_train, X_test, 
                        y_test,X_train_e,X_test_e,vocab_dim,input_length,input_length_e,batch_size,n_epoch)








