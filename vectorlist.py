# -*- coding: utf-8 -*-
import pandas as pd
import pyodbc
import pickle
import logging
import numpy as np
np.random.seed(1337)  
# For Reproducibility
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
#設定資料庫連接
cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=your server ip;DATABASE=dataset name;UID=your id;PWD=your passwords;')
cursor = cnxn.cursor()
query = "SELECT [review_content]FROM [movie].[dbo].[reviews]"
df = pd.read_sql(query, cnxn)
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
for index, row in df.iterrows():
    try:
        sentences+=word_split(row['review_content'])
    except:
        print('no')

def create_dictionaries(p_model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(p_model.wv.vocab.keys(), allow_update=True)
    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  
    #詞彙的索引
    w2vec = {word: model[word] for word in w2indx.keys()}  
    #詞彙的向量
    return w2indx, w2vec

#main function
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
print ('model tranning')
model = Word2Vec(sentences,
                 size=300,  # 詞向量維度
                 min_count=5,  # 詞頻閥值
                 window=5)  # 窗口大小


model.save('wordlist')  # 保存模型

#建立索引，詞向量字典
index_dict, word_vectors= create_dictionaries(model)

# 儲存pkl檔案

output = open("wordlist.pkl", 'wb')
pickle.dump(index_dict, output)  
pickle.dump(word_vectors, output)  
output.close()

if __name__ == "__main__":
    pass
