# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(1337) 
import numpy as np


from keras.models import Sequential
from keras.layers.embeddings import Embedding

import keras
from keras.layers.recurrent import LSTM 
from keras.layers import  Bidirectional
from keras.layers.core import Dense, Dropout
import matplotlib.pyplot as plt
from attention import SelfAttention
from ConfusionMetrics import precision,f1_score,recall
def train_BiLstm(p_n_symbols, p_embedding_weights, p_X_train, p_y_train,
                 p_X_test, p_y_test,vocab_dim,input_length,batch_size,n_epoch):
    """
    模型建置
    """
    print ('build model...')
    model = Sequential()
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=p_n_symbols,
                        mask_zero=True,
                        weights=[p_embedding_weights],
                        input_length=input_length))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(output_dim=50)))
    model.add(Dropout(0.5))
    model.add(SelfAttention())
    model.add(Dense(32,activation='tanh'))
    model.add(Dense(3,activation='softmax'))
   
    
    print ('processing...') 
    #調整學習率
    adam=keras.optimizers.Adam(lr=0.00003)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy', precision, recall, f1_score])

    print ("training...")
    history=model.fit(p_X_train, p_y_train, batch_size=batch_size, nb_epoch=n_epoch,
              validation_data=(p_X_test, p_y_test))
    #透過matplot繪圖顯示訓練過程
    print ("counting...")
    score, acc,prec,re,f1 = model.evaluate(p_X_test, p_y_test, batch_size=batch_size)
    print ('Test score:', score)
    print ('Test accuracy:', acc)
    print ('Test precision:', prec)
    print ('Test recall:', re)
    print ('Test f1-score:', f1)
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
   
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()
    plt.savefig('BiLstm')
    #儲存模型
    model.save('BiLstm.h5')
    print('model is saved')