# quara
kaggle competition Quara

## Try

* Preprocessing  
https://github.com/MajorTal/DeepSpell  
  
### Feature Engineering
#### Basic Feature
* N-gram  
word unigram/bigram/trigram  
char bigram/trigram/four-gram  
* ~~BoW~~  
* ~~TFIDF~~  
* ~~word_cnt~~  
* ~~unique_word_cnt~~  
* ~~char_cnt~~  
* ~~stopword_cnt~~  
* ~~punctuations_cnt~~  
* ~~upper_word_cnt~~  
* ~~word_title_cnt~~  
* ~~word_len_mean~~  
* ~~word_len_max~~  
* Entity  
* POS(part of speech) @NLTK/SpaCy  
https://qiita.com/kei_0324/items/400f639b2f185b39a0cf  


#### Advanced Feature
* WordNet  
https://gist.github.com/XericZephyr/462043957ee2a23ace73  
* NER-based feature  
https://toshipedia.jp/201803073553/


#### Target Encoding  
* oof_ratio_sincere/oof_ratio_insincere  
Out of Foldのデータセットにおけるその単語のsincere/insincereテキストに含まれていた割合  
* oof_tfidf_sincere/oof_tfidf_insincere  
Out of Foldのデータセットにおけるその単語のsincere/insincere内でのTFIDF  
* oof_cos_similar_sincere/oof_cos_similar_sincere(euclid, some scipy.dist...)  
Out of Foldのデータセットにおけるsincere/insincereとのcos similar avg  
* oof_lda_sincere/oof_lda_sincere  
Out of Foldのデータセットにおけるsincere/insincere別LDAでのtopic value  
* oof_wordvector_sincere/oof_wordvector_sincere  
Out of Foldのデータセットにおけるsincere/insincere別word vector similar/diff/ratio  

#### Word Embedding  
* LDA  
* LSI  
* sense2vec  
* Inception  
* Attention  
* word vector avg/min/max/std  
* word vector absolute avg/min/max/std  


#### NN
* 自動要約seq2seqの重要seq抽出部分のコードを読みたい  
https://techblog.exawizards.com/entry/2018/08/23/121437  
* SCDV  
https://qiita.com/fufufukakaka/items/a7316273908a7c400868  
https://www.pytry3g.com/entry/text-classification-SCDV  
* Word mover distance  


#### Other
* TF-IDF/word2vec/seq2seq...combine dataset solution  
http://datanerd.hateblo.jp/entry/2018/06/30/170121  
https://github.com/Cisco-Talos/fnc-1/tree/master/tree_model  

* Mercari 18th LGBM + wordbatch + Naive Bayes  
discussion  
https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/50252  
kernel  
https://www.kaggle.com/peterhurford/lgb-and-fm-18th-place-0-40604  
Multinomial naive Bayes  
https://qiita.com/fujin/items/bd58fc7a93dc6e001045  

* wordbatch  
creater  
https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/47295  
kernel  
https://www.kaggle.com/anttip/wordbatch-1-3-3-fm-ftrl-lb-0-9812  
https://www.kaggle.com/konohayui/attempt-with-wordbatch  
mercari kernel  
https://www.kaggle.com/anttip/wordbatch-ftrl-fm-lgb-lbl-0-42555  
qiita  
https://qiita.com/sh-tatsuno/items/0b0fbc15c03f3d54df90  
https://github.com/sh-tatsuno/WordBatch_Example/blob/master/WordBatch_Example.ipynb  

* Sentiment Analysis  
qiita  
https://qiita.com/yukinoi/items/c3c4e4e24a66b88d8215  

## Other Kaggle Competitoin

* Mercari  
2nd  
some model Ensemble. sprase NN.  
https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/50499  

## Not Try

* BERT   
https://www.kaggle.com/sergeykalutsky/introducing-bert-with-tensorflow/notebook  
