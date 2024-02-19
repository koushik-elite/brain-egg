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

train = pd.read_parquet('dataset/dataset.parquet')
print(train.head())

FEATURES = list(train.columns)
del FEATURES[1]
del FEATURES[3]
print(FEATURES)

gkf = GroupKFold(n_splits=5)
for i, (train_index, valid_index) in enumerate(gkf.split(train, train.target, train.case_id)):
    model = CatBoostClassifier(task_type='GPU', loss_function='MultiClass')
    train_pool = Pool(
        data = train.loc[train_index, FEATURES],
        label = train.loc[train_index, 'target'],
    )
    valid_pool = Pool(
        data = train.loc[valid_index,FEATURES],
        label = train.loc[valid_index, 'target'],
    )
    model.fit(train_pool, verbose=100, eval_set=valid_pool)
    model.save_model(f'CAT_v{VER}_f{i}.cat')