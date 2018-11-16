import pandas as pd
import spacy
import sys

# Spacy default model
nlp = spacy.load('en_core_web_sm')
ENTITY_ENUM = {
    '': '',
    'PERSON': 'person',
    'NORP': 'nationality',
    'FAC': 'facility',
    'ORG': 'organization',
    'GPE': 'country',
    'LOC': 'location',
    'PRODUCT': 'product',
    'EVENT': 'event',
    'WORK_OF_ART': 'artwork',
    'LANGUAGE': 'language',
    'DATE': 'date',
    'TIME': 'time',
#     'PERCENT': 'percent',
#     'MONEY': 'money',
#     'QUANTITY': 'quantity',
#     'ORDINAL': 'ordinal',
#     'CARDINAL': 'cardinal',
    'PERCENT': 'number',
    'MONEY': 'number',
    'QUANTITY': 'number',
    'ORDINAL': 'number',
    'CARDINAL': 'number',
    'LAW': 'law'
}

NUMERIC_TYPES = set([
    'DATE',
    'TIME',
    'PERCENT',
    'MONEY',
    'QUANTITY',
    'ORDINAL',
    'CARDINAL',
])



def spacy_to_lemma_pos_tag_dep(doc):
    doc = nlp(doc)
    return [(token.lemma_, token.pos_, token.tag_, token.dep_) for token in doc]


def spacy_to_entity(doc):
    doc = nlp(doc)
    return [(ent.text, ent.label_) for ent in doc.ents]


# 途中
def spacy_get_similary(doc):
    doc = nlp(doc)
    return [(ent.text, ent.label_) for token in doc.ents]


def spacy_get_entity_dict(doc, replace=False):
    '''
    Explain:
        textを受け取り、eitityに置換したtextと含まれていたentityを
        カウントして辞書を返す.
        entity: 14 pattern
    Args:
        doc(str): text
    Return:
    '''
    ent_dict = {
        'person':0,
        'nationality':0,
        'facility':0,
        'organization':0,
        'country':0,
        'location':0,
        'product':0,
        'event':0,
        'artwork':0,
        'language':0,
        'date':0,
        'time':0,
        'number':0,
        'law':0
    }
    # textをentity_tupleリストで返す(space split)
    ent_tuple_list = spacy_to_ent_lemma_pos_tag_dep(doc)

    for ent_tuple in ent_tuple_list:
        # Replace to Entity
        if replace:
            doc = doc.replace(ent_tuple[0], ent_tuple[1])
        ent_dict[ENTITY_ENUM[ent_tuple[1]]] +=1

    if replace:
        return ent_dict, doc
    else:
        return ent_dict


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
