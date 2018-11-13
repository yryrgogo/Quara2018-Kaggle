from main_quara import quara_load_data, cleansing_text
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
import gc
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from utils import logger_func, get_categorical_features, get_numeric_features, pararell_process
logger = logger_func()

import string
from wordcloud import STOPWORDS

qid = 'qid'
qt = 'question_text'
target = 'target'

logger.info("Load Data...")
train, test = quara_load_data()

## Feature Engineering ##
train["num_words@"] = train["question_text"].apply(lambda x: len(str(x).split()))
test["num_words@"] = test["question_text"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ## 
train["num_unique_words@"] = train["question_text"].apply(lambda x: len(set(str(x).split())))
test["num_unique_words@"] = test["question_text"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##
train["num_chars@"] = train["question_text"].apply(lambda x: len(str(x)))
test["num_chars@"] = test["question_text"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##
train["num_stopwords@"] = train["question_text"].apply(lambda x: len([w for w in str(x).split() if w in STOPWORDS ]))
test["num_stopwords@"] = test["question_text"].apply(lambda x: len([w for w in str(x).split() if w in STOPWORDS ]))

## Number of punctuations in the text ##
train["num_punctuations@"] = train["question_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
test["num_punctuations@"] = test["question_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

## Number of title case words in the text ##
train["num_words_upper@"] = train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["num_words_upper@"] = test["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##
train["num_words_upper@"] = train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["num_words_upper@"] = test["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##
train["num_words_lower@"] = train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.islower()]))
test["num_words_lower@"] = test["question_text"].apply(lambda x: len([w for w in str(x).split() if w.islower()]))

## Number of upper chars in the text ##
train["num_chars_upper@"] = train["question_text"].apply(lambda x: len([w for w in str(x) if w.isupper()]))
test["num_chars_upper@"] = test["question_text"].apply(lambda x: len([w for w in str(x) if w.isupper()]))

## Number of lower chars in the text ##
train["num_chars_lower@"] = train["question_text"].apply(lambda x: len([w for w in str(x) if w.islower()]))
test["num_chars_lower@"] = test["question_text"].apply(lambda x: len([w for w in str(x) if w.islower()]))

## Number of title case words in the text ##
train["num_words_title@"] = train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test["num_words_title@"] = test["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Average length of the words in the text ##
train["mean_word_len@"] = train["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test["mean_word_len@"] = test["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

## Max length of the words in the text ##
train["max_word_len@"] = train["question_text"].apply(lambda x: np.max([len(w) for w in str(x).split()]))
test["max_word_len@"] = test["question_text"].apply(lambda x: np.max([len(w) for w in str(x).split()]))

## Min length of the words in the text ##
train["min_word_len@"] = train["question_text"].apply(lambda x: np.min([len(w) for w in str(x).split()]))
test["min_word_len@"] = test["question_text"].apply(lambda x: np.min([len(w) for w in str(x).split()]))


#========================================================================
# Save Feature
#========================================================================
logger.info("Save Features...")
for col in train.columns:
    if col.count('@'):
        utils.to_pkl_gzip(obj=train[col].values, path=f'../features/1_first_valid/train_{col}')
        utils.to_pkl_gzip(obj=test[col].values, path=f'../features/1_first_valid/test_{col}')
