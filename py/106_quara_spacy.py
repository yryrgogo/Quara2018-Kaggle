import pandas as pd
import spacy
import sys

nlp = spacy.load('en_core_web_sm')
train = pd.read_csv('../input/train.csv')
doc_list = train['question_text'].values
doc = nlp(f'{doc_list[0]}')


def spacy_to_lemma_pos_tag_dep(doc):
    doc = nlp(doc)
    return [(token.lemma_, token.pos_, token.tag_, token.dep_) for token in doc]


def spacy_to_ent_lemma_pos_tag_dep(doc):
    doc = nlp(doc)
    return [(ent.text, ent.label_) for token in doc.ents]


# 途中
def spacy_get_similary(doc):
    doc = nlp(doc)
    return [(ent.text, ent.label_) for token in doc.ents]

# Basic
#  for token in doc:
#      print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#            token.shape_, token.is_alpha, token.is_stop)

# Entity
#  for ent in doc.ents:
#      print(ent.text, ent.start_char, ent.end_char, ent.label_)

# Similarity    
#  for token1 in doc:
#      for token2 in doc:
#          print(token1.text, token2.text, token1.similarity(token2))
