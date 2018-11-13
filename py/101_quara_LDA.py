feat_no = 101
from main_quara import quara_load_data, cleansing_text
from tqdm import tqdm
import numpy as np
import pandas as pd
import datetime
import sys
import re
import gc
import glob
import pickle as pkl
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from utils import logger_func, get_categorical_features, get_numeric_features, pararell_process
logger = logger_func()

# NLP
from wordcloud import WordCloud, STOPWORDS
from gensim.corpora import Dictionary
from gensim import corpora, matutils, models

qid = 'qid'
qt = 'question_text'
target = 'target'

logger.info("Load Data...")
train, test = quara_load_data()
df = pd.concat([train, test], axis=0)

#========================================================================
# Texts Cleansing & 2dim_list sample * word list
#========================================================================
logger.info("Cleansing Text...")
def pararell_cleansing(tx):
    return cleansing_text(tx).split()
texts = pararell_process(pararell_cleansing, df[qt].values)

#========================================================================
# Make Dictionary
#========================================================================
# 出現頻度が10以下の単語は無視/文書への出現頻度が8割以上の単語は無視
logger.info("Making Dictionary...")
dictionary = Dictionary(texts)
no_below = 10
no_above = 0.8
topics = 20
dictionary.filter_extremes(no_below=no_below, no_above=no_above)
logger.info("Making Corpus...")
corpus = [dictionary.doc2bow(text) for text in tqdm(texts)]

#========================================================================
# LDA Calculate
#========================================================================
logger.info("LDA Calculation...")
lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=topics)

#========================================================================
# Model Save
#========================================================================
logger.info("Dictionary & LDA Model Save...")
dictionary.save_as_text('../model/1111_gensim_dict_below10_above08')
with open('../model/1111_LDA_20topics_gensim__below10_above08', mode='wb') as f:
    pkl.dump(lda, f)
for topic in lda.show_topics(num_topics=-1):
    print(f'topics: {topic}\n')


# LDA Value write to Train & Test
mx = np.zeros((len(texts), topics))

# Get LDA Topic Value from corpus
logger.info("Get LDA Value from corpus...")
arg_list = []
for i, bow in tqdm(enumerate(corpus)):

# Pararell ===
#     arg_list.append([i, bow])
# def pararell_wrapper(args):
#     return pararell_write_lda_feat(*args)
# def pararell_write_lda_feat(i, bow):
#     tmp = np.zeros(topics+1)
# ===
    topic = lda.get_document_topics(bow)
    for (tp_no, prob) in topic:
        mx[i][tp_no] = prob
# Pararell
#         tmp[tp_no] = prob
#     tmp[topics+1] = i
#     return tmp
# p_list = pararell_process(pararell_wrapper, arg_list)

cols = [f"{feat_no}_topic{i}@" for i in range(20)]
df_lda = pd.DataFrame(mx, columns=cols)

train_idx = train.index
test_idx = test.index
lda_train = df_lda.loc[train_idx, :]
lda_test = df_lda.loc[test_idx, :]

#========================================================================
# Save Feature
#========================================================================
logger.info("Save Features...")
for col in lda_train.columns:
    utils.to_pkl_gzip(obj=lda_train[col].values, path=f'../features/1_first_valid/train_{col}')
    utils.to_pkl_gzip(obj=lda_test[col].values, path=f'../features/1_first_valid/test_{col}')
