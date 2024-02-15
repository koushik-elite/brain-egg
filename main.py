import os, gc
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
import pickle

VER = 3

df = pd.read_csv('dataset/train.csv')
TARGETS = df.columns[-6:]
print('Train shape:', df.shape )
print('Targets', list(TARGETS))

train = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg({'spectrogram_id':'first','spectrogram_label_offset_seconds':'min'})
train.columns = ['spec_id','min']

tmp = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg({'spectrogram_label_offset_seconds':'max'})
train['max'] = tmp

tmp = df.groupby('eeg_id')[['patient_id']].agg('first')
train['patient_id'] = tmp

tmp = df.groupby('eeg_id')[TARGETS].agg('sum')
for t in TARGETS:
    train[t] = tmp[t].values
    
y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1,keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')
train['target'] = tmp

train = train.reset_index()
print('Train non-overlapp eeg_id shape:', train.shape )
# train.head()

# PATH = 'dataset/train_spectrograms/'
# files = os.listdir(PATH)
# print(f'There are {len(files)} spectrogram parquets')

spectrograms = np.load('dataset/specs.npy',allow_pickle=True)
# spectrograms = pickle.load(open('dataset/specs.npy','rb'))

all_eegs = np.load('dataset/eeg_specs.npy',allow_pickle=True)
# all_eegs = pickle.load(open('dataset/eeg_specs.npy','rb'))

print('Train spectrograms and all_eegs loaded')

# FEATURE NAMES
# PATH = '/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/'
# SPEC_COLS = pd.read_parquet(f'{PATH}1000086677.parquet').columns[1:]
# FEATURES = [f'{c}_mean_10m' for c in SPEC_COLS]
# FEATURES += [f'{c}_min_10m' for c in SPEC_COLS]
# FEATURES += [f'{c}_mean_20s' for c in SPEC_COLS]
# FEATURES += [f'{c}_min_20s' for c in SPEC_COLS]
# FEATURES += [f'eeg_mean_f{x}_10s' for x in range(512)]
# FEATURES += [f'eeg_min_f{x}_10s' for x in range(512)]
# FEATURES += [f'eeg_max_f{x}_10s' for x in range(512)]
# FEATURES += [f'eeg_std_f{x}_10s' for x in range(512)]
# print(f'We are creating {len(FEATURES)} features for {len(train)} rows... ',end='')

data = np.zeros((len(train),len(FEATURES)))
for k in tqdm(range(len(train))):
    if k%100==0: print(k,', ',end='')
    row = train.iloc[k]
    r = int( (row['min'] + row['max'])//4 ) 

    # 10 MINUTE WINDOW FEATURES (MEANS and MINS)
    x = np.nanmean(spectrograms[row.spec_id][r:r+300,:],axis=0)
    data[k,:400] = x
    x = np.nanmin(spectrograms[row.spec_id][r:r+300,:],axis=0)
    data[k,400:800] = x

    # 20 SECOND WINDOW FEATURES (MEANS and MINS)
    x = np.nanmean(spectrograms[row.spec_id][r+145:r+155,:],axis=0)
    data[k,800:1200] = x
    x = np.nanmin(spectrograms[row.spec_id][r+145:r+155,:],axis=0)
    data[k,1200:1600] = x

    # RESHAPE EEG SPECTROGRAMS 128x256x4 => 512x256
    eeg_spec = np.zeros((512,256),dtype='float32')
    xx = all_eegs[row.eeg_id]
    for j in range(4): eeg_spec[128*j:128*(j+1),] = xx[:,:,j]

    # 10 SECOND WINDOW FROM EEG SPECTROGRAMS 
    x = np.nanmean(eeg_spec.T[100:-100,:],axis=0)
    data[k,1600:2112] = x
    x = np.nanmin(eeg_spec.T[100:-100,:],axis=0)
    data[k,2112:2624] = x
    x = np.nanmax(eeg_spec.T[100:-100,:],axis=0)
    data[k,2624:3136] = x
    x = np.nanstd(eeg_spec.T[100:-100,:],axis=0)
    data[k,3136:3648] = x

train[FEATURES] = data
print("--------------------------------------------------------"); 
print('New train shape:',train.shape)

# FREE MEMORY
del all_eegs, spectrograms, data
gc.collect()

import catboost as cat
from catboost import CatBoostClassifier, Pool
print('CatBoost version',cat.__version__)

all_oof = []
all_true = []
TARS = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}

train_pool = Pool(
    data = train[FEATURES],
    label = train['target'].map(TARS),
)

print(train_pool.data)
print("----------label------------------------------------------"); 
print(train_pool.label)