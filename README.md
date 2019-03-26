# Quara
kaggle competition Quara.
途中でEloが来たので結局やらなかった。
  
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
* SVD  
BoW/TFIDF(N-gram, below:, max_feature:50000/100000/200000/300000/400000/500000)  
* ~~word_cnt~~  
* ~~unique_word_cnt~~  
* ~~char_cnt~~  
* ~~stopword_cnt~~  
* ~~punctuations_cnt~~  
* ~~upper_word_cnt~~  
* ~~word_title_cnt~~
* sentence_cnt
* ~~word_len_mean~~  
* ~~word_len_max~~  


#### Advanced Feature
* WordNet  
lemma  
similarity  
* spacy  
entity  
pos(part of speech)  
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
* entity-embedding-rossmann  
https://github.com/entron/entity-embedding-rossmann  
* preprocessing for embedding (stemming, stopword...is not best for embedding)
https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings/notebook  


#### NN
* 自動要約seq2seqの重要seq抽出部分のコードを読みたい  
https://techblog.exawizards.com/entry/2018/08/23/121437  
* SCDV  
https://qiita.com/fufufukakaka/items/a7316273908a7c400868  
https://www.pytry3g.com/entry/text-classification-SCDV  
* Word mover distance  
* About Overfitting/Underfitting  
https://www.kaggle.com/ratthachat/handle-overfitting-error-analysis-of-glove-gru/notebook  


#### Ensemble  
* oof_top50_similar_psuedo_label  
CVでlossの最も小さいTOP50のsincere/insincereのテキストとのsimilar(cos/wordvector)  
* Ridge  
* Wordbatch  
* FFM  
* Naive Bayes  
https://www.kaggle.com/jpmiller/drunk-anteaters-logistic-regression  

  
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


## 雑メモ
### Feature Engineering
Levenshtein distance 
Fixed misspellings by finding word vector neighborhoods. Fasttext tool can create vectors for out-of-dictionary words which is really nice. 
Text Normalization. 表記揺れの是正と似通った単語の統一、数字の置き換え, Fixed some miss spellings
Vowpal Wabbit model
Quara 2017 1st place solution
Not Try
Classical text mining features

Similarity measures on LDA and LSI embeddings.
Similarity measures on bag of character n-grams ( TFIDF reweighted or not) from 1 to 8 grams.
Abhishek's and owl’s kindly shared features.
Edit and sequence matching distances, percentage of common tokens up to 1, 2, …, 6 when question ends the same, or starts the same
Length of questions, diff of length
Number of capital letters, question marks etc…
Indicators for Question 1/2 starting with "Are", "Can", "How" etc… and all mathematical engineering corresponding
We also used stanford corenlp to tokenizer, postagger and ner to preprocessing text input for some deep learning models.


### Reference Discussion
#### 2018 Quara
- CPMP F1 Optimization https://www.kaggle.com/cpmpml/f1-score-expectation-maximization-in-o-n/code
- Augument Text https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/71083
- SOTA https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/70821
- 2018 Toxic Discussion  
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion  

#### Winner's
- 1st place solution https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557
- 2nd place solution https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52612
- 3rd place solution https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52762
- 3rd place solution https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52692
- 3rd place solution(single model) https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52644
- 5th place solution https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52630
- 11th place solution https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52526
- 12th place solution https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52702
- 15th place solution https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52563
- 25th place solution https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52647
- 27th place solution https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52719
- 33rd place solution https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52666
- 34th place solution https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52645

#### Others
- LR words and char n-grams https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams/code
- For Beginners Keras https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras/notebook 
- LSTM baseline https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout/comments
- NB-SVM strong linear baseline https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline/comments
- keras LSTM attention globe840b https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043/code
- You should use out-of-fold data https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52224
- adversarial_validation https://www.kaggle.com/ogrellier/adversarial-validation-and-lb-shakeup/notebook
- spell checker using word embeddings(Quara) https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/51426
- 2017 Quara Discussion  
https://www.kaggle.com/c/quora-question-pairs/discussion

- 28th place solution https://www.kaggle.com/c/quora-question-pairs/discussion/34560
- 33th place solution https://www.kaggle.com/c/quora-question-pairs/discussion/34292

- tkmさんのmagic feature (networkx) https://www.kaggle.com/tkm2261/my-magic-feature/notebook
- CPMP spell checker https://www.kaggle.com/cpmpml/spell-checker-using-word2vec/notebook
- CPMP Good or Bad? https://www.kaggle.com/c/quora-question-pairs/discussion/33801
- LSTM with Glove https://www.kaggle.com/lystdo/lb-0-18-lstm-with-glove-and-magic-features
- https://www.kaggle.com/c/quora-question-pairs/discussion/31284
- benchmark https://www.kaggle.com/c/quora-question-pairs/discussion/31019
- external interesting solution https://www.kaggle.com/c/quora-question-pairs/discussion/30260
- Jackさん Ensemble https://www.kaggle.com/c/quora-question-pairs/discussion/34267
- funny feature https://www.kaggle.com/c/quora-question-pairs/discussion/34335
- Graph Feature https://www.kaggle.com/c/quora-question-pairs/discussion/34295
- LSTM with word2vec https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
#### 可視化系
- Temporal pattern in train response rates?  
https://www.kaggle.com/ashhafez/temporal-pattern-in-train-response-rates/notebook

####  その他
- メルカリ eli5 使えそう？ https://www.kaggle.com/lopuhin/eli5-for-mercari
- mercari 4th place solution(NN. Code is too long.) https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/49678
LSTM Stackingがなぜうまくいかないか？
- Explanations about why stacking LSTMs often leads to worse LB and one question about Stacking procedures
https://www.kaggle.com/c/quora-question-pairs/discussion/34294
- Jackさん Modeling Flow
- グラフ理論　クリーク構造

