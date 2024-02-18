import os, gc
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
import pickle
import catboost as cat
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold, GroupKFold

print('CatBoost version',cat.__version__)

VER = 3

TARS = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}


train = pd.read_parquet('dataset/train.parquet')
print(train.head())
FEATURES = train.columns[12:]
print(train.loc[0, FEATURES].values)
# for col in FEATURES:
#     print(col)

gkf = GroupKFold(n_splits=5)
for i, (train_index, valid_index) in enumerate(gkf.split(train, train.target, train.patient_id)):
    model = CatBoostClassifier(task_type='GPU', loss_function='MultiClass')
    train_pool = Pool(
        data = train.loc[train_index, FEATURES],
        label = train.loc[train_index, 'target'].map(TARS),
    )
    valid_pool = Pool(
        data = train.loc[valid_index,FEATURES],
        label = train.loc[valid_index, 'target'].map(TARS),
    )
    model.fit(train_pool, verbose=100, eval_set=valid_pool)
    model.save_model(f'CAT_v{VER}_f{i}.cat')