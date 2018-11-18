feat_no = 102
is_tfidf = True
is_svd = True
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

qid = 'qid'
qt = 'question_text'
target = 'target'

logger.info("Load Data...")
train, test = quara_load_data()

def get_tfidf(text_list):
    '''
    Explain:
        TFIDFを出力したいテキストリストを作成する
    Args:
        text_list(list): split前のテキストリスト
    Return:
        sparse csr_matrix: TFIDF値が入ったスパースな行列
    '''

    # Get the tfidf
    logger.info("Calculate TFIDF...")
    tfidf_vec = TfidfVectorizer(
        max_features = 100000,
        min_df=3,
        max_df=0.8,
        stop_words="english",
        analyzer='word',
        #  analyzer='char',
        strip_accents='unicode',
        ngram_range=(1,3),
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    ).fit(text_list)

    df_tfidf = tfidf_vec.transform(text_list)
    utils.to_pkl_gzip(obj=df_tfidf, path='./df_tfidf')

# Load Text List
train_text_list = list(train[qt].values)
test_text_list = list(test[qt].values)

# 並列処理でクレンジング
logger.info("Cleansing Text...")
def pararell_cleansing(tx):
    return cleansing_text(tx)
train_text_list = pararell_process(pararell_cleansing, train_text_list)
test_text_list = pararell_process(pararell_cleansing, test_text_list)
text_list = train_text_list + test_text_list
# TFIDF
get_tfidf(text_list)
df_tfidf = utils.read_pkl_gzip(path='./df_tfidf.gz')


if is_svd:
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=100, random_state=1208)
    svd_tfidf = svd.fit_transform(df_tfidf)
    col_names = [f"{feat_no}_svd100_tfidf100k_{i}@" for i in range(100)]
    df_svd = pd.DataFrame(svd_tfidf, columns=col_names)
    train_idx = train.index
    test_idx = test.index
    svd_train = df_svd.loc[train_idx, :]
    svd_test = df_svd.loc[test_idx, :]
    print(svd_train.shape)
    print(svd_test.shape)

    #========================================================================
    # Save Feature
    #========================================================================
    logger.info("Save Features...")
    for col in df_svd.columns:
        utils.to_pkl_gzip(obj=svd_train[col].values, path=f'../features/1_first_valid/train_{col}')
        utils.to_pkl_gzip(obj=svd_test[col].values, path=f'../features/1_first_valid/test_{col}')
