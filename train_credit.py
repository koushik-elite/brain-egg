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
from sklearn.model_selection import train_test_split

# def set_table_dtypes(df: pl.DataFrame) -> pl.DataFrame:
#     # implement here all desired dtypes for tables
#     # the following is just an example
#     for col in df.columns:
#         # last letter of column name will help you determine the type
#         if col[-1] in ("P", "A"):
#             df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))

#     return df

def convert_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:  
        if df[col].dtype.name in ['object', 'string']:
            df[col] = df[col].astype("string").astype('category')
            current_categories = df[col].cat.categories
            new_categories = current_categories.to_list() + ["Unknown"]
            new_dtype = pd.CategoricalDtype(categories=new_categories, ordered=True)
            df[col] = df[col].astype(new_dtype)
    return df

train = pd.read_parquet('dataset/dataset.parquet')
print(train.head())

cols_pred = []
for col in train.columns:
    if col[-1].isupper() and col[:-1].islower():
        cols_pred.append(col)

print(cols_pred)

case_ids = train["case_id"].unique()
# case_ids_train, case_ids_test = train_test_split(case_ids, train_size=0.6, random_state=1)
# case_ids_valid, case_ids_test = train_test_split(case_ids_test, train_size=0.5, random_state=1)

X_train = train[cols_pred]
y_train = train["target"]

print(X_train.loc[0].values)

for df in [X_train]:
    df = convert_strings(df)

print(X_train.loc[0].values)

# gkf = GroupKFold(n_splits=5)
# for i, (train_index, valid_index) in enumerate(gkf.split(train, train.target, train.case_id)):
#     model = CatBoostClassifier(task_type='GPU', loss_function='MultiClass')
#     train_pool = Pool(
#         data = train.loc[train_index, cols_pred],
#         label = train.loc[train_index, 'target'],
#     )
#     valid_pool = Pool(
#         data = train.loc[valid_index,cols_pred],
#         label = train.loc[valid_index, 'target'],
#     )
#     model.fit(train_pool, verbose=100, eval_set=valid_pool)
#     model.save_model(f'CAT_v{VER}_f{i}.cat')