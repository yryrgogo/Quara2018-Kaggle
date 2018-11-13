win_path = f'../features/4_winner/*.gz'
stack_name='add_nest'
fname=''
xray=False
#  xray=True
#========================================================================
# argv[1] : model_type 
# argv[2] : learning_rate
# argv[3] : early_stopping_rounds
# argv[4] : pred_type
#========================================================================
import gc
import sys

try:
    model_type=sys.argv[1]
except IndexError:
    model_type='lgb'
try:
    learning_rate = float(sys.argv[2])
except IndexError:
    learning_rate = 0.1
try:
    early_stopping_rounds = int(sys.argv[3])
except IndexError:
    early_stopping_rounds = 150
num_boost_round = 10000
seed = 1208

try:
    pred_type = int(sys.argv[4])
except IndexError:
    pred_type = 1

import numpy as np
import pandas as pd
import datetime
import glob
import os
HOME = os.path.expanduser('~')

sys.path.append(f'{HOME}/kaggle/data_analysis/model')
from params_lgbm import params_quara
from xray_wrapper import Xray_Cal
sys.path.append(f'{HOME}/kaggle/data_analysis')
from model.lightgbm_ex import lightgbm_ex as lgb_ex

sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from preprocessing import get_ordinal_mapping
from utils import logger_func
logger=logger_func()

if model_type=='lgb':
    params = params_quara()
    params['learning_rate'] = learning_rate
elif model_type=='xgb':
    params = xgb_params_0814()
    params['eta'] = learning_rate
elif model_type=='extra':
    params = extra_params()

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

key = 'qid'
target = 'target'
ignore_list = [key, target, 'question_text']

#========================================================================
# Data Load
#========================================================================

base = utils.read_df_pkl('../input/base*')
win_path_list = glob.glob(win_path)
train_path_list = []
test_path_list = []
for path in win_path_list:
    if path.count('train'):
        train_path_list.append(path)
    elif path.count('test'):
        test_path_list.append(path)

base_train = base[~base[target].isnull()].reset_index(drop=True)
base_test = base[base[target].isnull()].reset_index(drop=True)
train_feature_list = utils.pararell_load_data(path_list=train_path_list)
test_feature_list = utils.pararell_load_data(path_list=test_path_list)
train = pd.concat(train_feature_list, axis=1)
train = pd.concat([base_train, train], axis=1)
test = pd.concat(test_feature_list, axis=1)
test = pd.concat([base_test, test], axis=1)

key = ''

#========================================================================
# Args
#========================================================================
metric = 'logloss'
fold=5
fold_type='stratified'
group_col_name=''
dummie=1
oof_flg=True

#========================================================================
# Train & Prediction Start
#========================================================================
if len(key)>0:
    LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)
    train, test, drop_list = LGBM.data_check(train=train, test=test, target=target)
    if len(drop_list):
        train.drop(drop_list, axis=1, inplace=True)
        test.drop(drop_list, axis=1, inplace=True)

    LGBM = LGBM.cross_prediction(
        train=train
        ,test=test
        ,key=key
        ,target=target
        ,fold_type=fold_type
        ,fold=fold
        ,group_col_name=group_col_name
        ,params=params
        ,num_boost_round=num_boost_round
        ,early_stopping_rounds=early_stopping_rounds
        ,oof_flg=oof_flg
    )
else:
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.metrics import log_loss, confusion_matrix, f1_score, accuracy_score

    # Dataset Setting
    train_ = utils.read_pkl_gzip('../py/train_tfidf.gz')
    test_ = utils.read_pkl_gzip('../py/test_tfidf.gz')
    from scipy.sparse import hstack, csr_matrix
    y = train[target]
    prediction = np.array()
    train = hstack((csr_matrix(train.drop(['qid', target], axis=1)), train_))
    test = hstack((csr_matrix(test.drop(['qid', target], axis=1)), test_))

    ' KFold '
    if fold_type == 'stratified':
        folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)  # 1
        kfold = folds.split(train, y)

    for n_fold, (trn_idx, val_idx) in enumerate(kfold):
        x_train, x_val, y_train, y_val = train_test_split(train, y, test_size=0.2, random_state=42)

        # Dataset
        lgb_train = lgb.Dataset(data=x_train, label=y_train)
        lgb_eval = lgb.Dataset(data=x_val, label=y_val)

        lgb = lgb.train(
            train_set=lgb_train,
            valid_sets=lgb_eval,
            params=params,
            verbose_eval=200,
            early_stopping_rounds=early_stopping_rounds,
            num_boost_round=num_boost_round
        )

        y_pred = lgb.predict(x_val)
        score = log_loss(y_val, y_pred)
        logger.info(f'Fold No: {n_fold} | {metric}: {score}')

        test_pred = lgb.predict(test)

        if len(prediction) == 0:
            prediction = test_pred
        else:
            prediction += test_pred

        #  ' Feature Importance '
        #  feim_name = f'{n_fold}_importance'
        #  feim = self.df_feature_importance(feim_name=feim_name)

        #  if len(self.cv_feim) == 0:
        #      cv_feim = feim.copy()
        #  else:
        #      cv_feim = self.cv_feim.merge(feim, on='feature', how='inner')

        break

#========================================================================
# Result
#========================================================================
for thresh in np.arange(0.1, 0.301, 0.01):
    thresh = np.round(thresh, 2)
    f1 = f1_score(y_val, (y_pred>thresh).astype(int))
    print(f"F1 score at threshold {thresh} is {f1}")
sys.exit()

cv_score = LGBM.cv_score
result = LGBM.prediction
cv_feim = LGBM.cv_feim
feature_num = len(LGBM.use_cols)

cv_feim.to_csv(f'../valid/{start_time[4:12]}_{model_type}_{fname}_feat{feature_num}_CV{cv_score}_lr{learning_rate}.csv', index=False)

#========================================================================
# X-RAYの計算と出力
# Args:
#     model    : 学習済のモデル
#     train    : モデルの学習に使用したデータセット
#     col_list : X-RAYの計算を行うカラムリスト。指定なしの場合、
#                データセットの全カラムについて計算を行うが、
#                計算時間を考えると最大30カラム程度を推奨。
#========================================================================
if xray:
    train.reset_index(inplace=True)
    train = train[LGBM.use_cols]
    result_xray = pd.DataFrame()
    N_sample = 500000
    max_point = 30
    for fold_num in range(fold):
        model = LGBM.fold_model_list[fold_num]
        if fold_num==0:
            xray_obj = Xray_Cal(logger=logger, ignore_list=ignore_list, model=model)
        xray_obj, tmp_xray = xray_obj.get_xray(base_xray=train, col_list=train.columns, fold_num=fold_num, N_sample=N_sample, max_point=max_point, Pararell=True)
        tmp_xray.rename(columns={'xray':f'xray_{fold_num}'}, inplace=True)

        if len(result_xray):
            result_xray = result_xray.merge(tmp_xray.drop('N', axis=1), on=['feature', 'value'], how='inner')
        else:
            result_xray = tmp_xray.copy()
        del tmp_xray
        gc.collect()

    xray_col = [col for col in result_xray.columns if col.count('xray')]
    result_xray['xray_avg'] = result_xray[xray_col].mean(axis=1)
    result_xray.to_csv(f'../output/{start_time[4:10]}_xray_{model_type}_CV{LGBM.cv_score}.csv', index=False)
    sys.exit()

submit = pd.read_csv('../input/sample_submission.csv')
#  submit = []

#========================================================================
# STACKING
#========================================================================
if len(stack_name)>0:
    logger.info(f'result_stack shape: {LGBM.result_stack.shape}')
    utils.to_pkl(path=f"../stack/{start_time[4:12]}_{stack_name}_{model_type}_CV{str(cv_score).replace('.', '-')}_{feature_num}features.fp", obj=LGBM.result_stack)
logger.info(f'FEATURE IMPORTANCE PATH: {HOME}/kaggle/home-credit-default-risk/output/cv_feature{feature_num}_importances_auc_{cv_score}.csv')

#========================================================================
# Submission
#========================================================================
if len(submit)>0:
    if stack_name=='add_nest':
        test[target] = result
        test = test.reset_index()[[key, target]].groupby(key)[target].mean().reset_index()
        submit = submit[key].to_frame().merge(test, on=key, how='left')
        submit[target].fillna(0, inplace=True)
        submit_path = f'../submit/{start_time[4:12]}_submit_{fname}_{model_type}_rate{learning_rate}_{feature_num}features_CV{cv_score}_LB.csv'
        submit.to_csv(submit_path, index=False)
        utils.submit(file_path=submit_path)

    else:
        submit[target] = result
        submit.to_csv(f'../submit/{start_time[4:12]}_submit_{model_type}_rate{learning_rate}_{feature_num}features_CV{cv_score}_LB.csv', index=False)
