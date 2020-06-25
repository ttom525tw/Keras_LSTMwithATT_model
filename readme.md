# Combining Long Short-term Memory and Attention Models in Sentiment Analysis
---
This project will show how to use IMDb review data and Combining Long Short-term Memory and Attention Models in Sentiment Analysis.

+ ### Step1. Data collection
    + #### crawler
    在執行IMDb_crawler.py時，依序對照事先儲存的電影ID訪問User reviews頁面，主要是透過使用selenium模擬瀏覽IMDb的頁面並獲取網頁中需要的欄位值 ，最後輸出成json檔。
    + #### 資料儲存與整理
    將包含個電影評論文本資料依序儲存到 MSSQL Server中，在過程進行資料前處理(移除標點符號，調整為小寫，移除表情符號...)
    + #### 提取情緒特徵單詞
    使用[NRC-Emotion-Lexicon](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) 所提供的情緒特徵辭典，在基於情緒輪理論[Plutchik-wheel](https://zh.wikipedia.org/wiki/File:Plutchik-wheel.svg)中的8種基本情緒(生氣、噁心、害怕、傷心、期待、喜悅、驚喜和信任)所歸類的類別的詞語提取文本當中的情緒單詞，作為情緒特徵部分資料的輸入資料。
+ ### Step2. Word2vec
    執行vectorlist.py ,透過Word2vec對於評論句子中的單詞進行詞向量的訓練，最後輸出成wordlist.pkl的詞彙索引表。

+ ### Step3. Build models
    + #### 資料提取
        從MSSQL Server中提取儲存資料集當中的評論資料以及情緒特徵資料，轉換為詞向量並輸入到模型中進行訓練。
    + #### 模型訓練
    在模型建置的部分主要是透過keras進行建置，範例中包括使用以下幾點：
        + 長短期記憶模型 
        + 雙向長短期記憶模型
        + 注意力模型
        + 提取評論文本當中的情緒特徵
    + #### 衡量指標
        + 準確率(accuracy)
        + F1-score
        + 精確率(precision)
        + 召回率(recall)
    
    在模型訓練後，通過測試集可得到衡量指標以及訓練過程圖示。
    # 模型範例執行
    + ## 使用評論文本資料(model_main.py)
        ### Lstm
        
       ```train_lstm(n_symbols, embedding_weights, X_train, y_train, X_test, y_test,vocab_dim,input_length,batch_size,n_epoch)```
       ### BiLstm
        
       ```train_biLstm(n_symbols, embedding_weights, X_train, y_train, X_test, y_test,vocab_dim,input_length,batch_size,n_epoch)```
    + ## 使用評論文本資料與情緒特徵(double_model_main.py)
       ### Lstm_t,Lstm_e
        
       ```train_double_Lstm(n_symbols, embedding_weights, X_train, y_train, X_test, y_test,X_train_e,X_test_e,vocab_dim,input_length,input_length_e,batch_size,n_epoch)```
       
       ### Lstm_t,BiLstm_e
       ```train_Lstm_BiLstm_e(n_symbols, embedding_weights, X_train, y_train, X_test, y_test,X_train_e,X_test_e,vocab_dim,input_length,input_length_e,batch_size,n_epoch)```
       
       ### BiLstm_t,Lstm_e
       ```train_BiLstm_Lstm_e(n_symbols, embedding_weights, X_train, y_train, X_test, y_test,X_train_e,X_test_e,vocab_dim,input_length,input_length_e,batch_size,n_epoch)```
       
       ### BiLstm_t,BiLstm_e
       ```train_Lstm_BiLstm_e(n_symbols, embedding_weights, X_train, y_train, X_test, y_test,X_train_e,X_test_e,vocab_dim,input_length,input_length_e,batch_size,n_epoch)```
    + ## 注意力模型(attention.py)
        ```from attention import SelfAttention``` 
        
        ```model.add(SelfAttention())```
        
       
        
        
    
    
        
    




