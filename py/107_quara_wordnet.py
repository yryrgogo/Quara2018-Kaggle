from nltk.corpus import wordnet as wn
from nltk import word_tokenize

for i in wn.synsets('America'):
    print(i.lemma_names())
