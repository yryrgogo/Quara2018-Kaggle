import numpy as np
import sys
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk import word_tokenize
brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')


is_sample=True
is_sample=False

#========================================================================
# Similarity Value
#========================================================================
if is_sample:
    dog = wn.synsets('dog')[0]
    cat = wn.synsets('cat')[0]
    # 単語間の最短グラフ距離？数字は直感的でない
    print(dog.path_similarity(wn.synsets('cat')[0]))
    # 意味の近さについては直感的な数字が出る
    print(dog.wup_similarity(wn.synsets('cat')[0]))
    # information contentsの種類によって結果が変わるがまあまあ使いやすい
    print(dog.lin_similarity(cat, semcor_ic))
    # よくわからん
    print(dog.jcn_similarity(other=cat, ic=brown_ic))
    #  2つの単語の共通上位概念？
    print(dog.lowest_common_hypernyms(cat)[0].lemma_names())


def get_wordnet_similarity_path_wup(w1, w2):
    w1_sets = wn.synsets(w1)
    w2_sets = wn.synsets(w2)
    path_val = []
    wup_val = []
    for w1, w2 in zip(w1_sets, w2_sets):
        path_val.append(w1.path_similarity(w2))
        wup_val.append(w1.wup_similarity(w2))
    return np.max(path_val), np.max(wup_val)


#========================================================================
# Get Synonym List
#========================================================================
def get_wordnet_lemma(word):
    lemma_list = []
    for w in wn.synsets(word):
        lemma_list+=w.lemma_names()
    return list(set(lemma_list) - set([word]))
