import spacy
import sys
nlp = spacy.load('en_core_web_sm')
#  doc_list = train['question_text'].values
#  doc = nlp(f'{doc_list[0]}')

#  for token in doc:
#      print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#            token.shape_, token.is_alpha, token.is_stop)

#  for ent in doc.ents:
#      print(ent.text, ent.start_char, ent.end_char, ent.label_)

#  # Similarity    
#  for token1 in tokens:
#      for token2 in tokens:
#          print(token1.text, token2.text, token1.similarity(token2))
