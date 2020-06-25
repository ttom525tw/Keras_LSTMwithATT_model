# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(1337) 
import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
import keras
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
import matplotlib.pyplot as plt
from attention import SelfAttention
from ConfusionMetrics import precision,f1_score,recall
from keras.models import Model
def train_double_Lstm(p_n_symbols, p_embedding_weights, p_X_train, p_y_train, p_X_test, 
                        p_y_test,p_X_train_e,p_X_test_e,vocab_dim,input_length,
                        input_length_e,batch_size,n_epoch):
    """
    模型建置
    """
    print ('build model...')
    #text_Lstm 
    model_t = Sequential()
    model_t.add(Embedding(output_dim=vocab_dim,
                        input_dim=p_n_symbols,
                        mask_zero=True,
                        weights=[p_embedding_weights],
                        input_length=input_length))

    model_t.add(LSTM(output_dim=50))
    model_t.add(Dropout(0.5))
    model_t.add(SelfAttention())
    

    #emotion_lstm   
    model_e = Sequential()
    model_e.add(Embedding(output_dim=vocab_dim,
                        input_dim=p_n_symbols,
                        mask_zero=True,
                        weights=[p_embedding_weights],
                        input_length=input_length_e))
    model_e.add(LSTM(output_dim=50))
    model_e.add(Dropout(0.5))
    model_e.add(SelfAttention())
    #merge text 跟 emotion 兩部分模型
    mergedOut = keras.layers.Add()([model_t.output,model_e.output])
    mergedOut = Dense(3, activation='softmax')(mergedOut)
    model = Sequential()
    model=Model([model_t.input,model_e.input],mergedOut)
    print ('processing...') 
    #調整學習率
    adam=keras.optimizers.Adam(lr=0.0001)
   
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy', precision, recall, f1_score])

    print ("training...")
    history=model.fit([p_X_train,p_X_train_e], p_y_train, batch_size=batch_size, nb_epoch=n_epoch,
              validation_data=([p_X_test,p_X_test_e], p_y_test))
    
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
    #儲存模型
    plt.savefig('Bilstm')
    model.save('double_Lstm.h5')
    print('model is saved')