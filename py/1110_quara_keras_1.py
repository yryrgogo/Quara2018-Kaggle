
# coding: utf-8
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
sys.exit()
# In[1]:

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime
import sys
import re
import gc
import glob

import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from utils import logger_func, get_categorical_features, get_numeric_features, pararell_process
logger = logger_func()
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)

key = 'qid'
qt = 'question_text'


# In[2]:


import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


# In[3]:


#  train = utils.read_pkl_gzip(path='../input/1116_train_wordnet_lemma_dict.gz')
#  train_df = utils.read_df_pkl('../input/train*.p')[['qid', 'target']]
#  print(train_df.set_index('qid').head())


seed = 1208
#  tmp = train.copy()

#  # for uid, value in tmp.items():
#  def pararell_val_join(args):
#      uid = args[0]
#      value = args[1]
#      df_dict = {}
#      np.random.seed(seed)
#      val_len = int(len(value)*0.5)
#      try:
#          value = np.random.choice(a=value, size=val_len)
#      except ValueError:
#          pass
#      df_dict[uid] = " ".join(value)
#      return df_dict
#  tmp_dict = {}
#  p_list = pararell_process(pararell_val_join, tmp.items())
#  [tmp_dict.update(p) for p in p_list]
#  tmp_train = pd.Series(tmp_dict).to_frame()
#  tmp_train = tmp_train.join(train_df.set_index('qid'))
#  tmp_train.rename(columns={0:qt}, inplace=True)

#  utils.to_df_pkl(df=tmp_train, path='../input/', fname='wn_bagging_train')
#  sys.exit()

tmp_train = utils.read_df_pkl(path='../input/wn_bagging_train*.p')
tmp_train = tmp_train.head(50000)

## split to train and val
train_df, val_df = train_test_split(tmp_train, test_size=0.2, random_state=seed)

## some config values 
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

## fill up the missing values
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
# test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
# test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
# test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values


### add for TensorBoard
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

old_session = KTF.get_session()

session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)
### 


# In[5]:
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
# x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())


### add for TensorBoard
tb_cb = keras.callbacks.TensorBoard(log_dir="../tflog/", histogram_freq=1)
cbks = [tb_cb]
###


# In[6]:


## Train the model 
model.fit(train_X, train_y, batch_size=512, epochs=2,
          validation_data=(val_X, val_y))


### add for TensorBoard
KTF.set_session(old_session)
###


pred_noemb_val_y = model.predict([val_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_noemb_val_y>thresh).astype(int))))
# del model, inp, x
# import gc; gc.collect()
# time.sleep(10)

