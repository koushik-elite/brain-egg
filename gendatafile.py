import os, gc
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import pandas as pd, numpy as np
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import mediainfo

import scipy
from scipy.io import wavfile

import tensorflow as tf 

# import pickle
# import catboost as cat
# from catboost import CatBoostClassifier, Pool
# from sklearn.model_selection import KFold, GroupKFold

desired_length = 10000
desired_sample_rate = 16000
columns = [f"col_{i}" for i in range(desired_length)]
# columns.insert(0, "label")
print(len(columns))

original = pd.read_csv("/mnt/d/Data/birds/train_metadata_1.csv")
datalen = original.shape[0]
labels = []
data = np.zeros((datalen, desired_length))
index = 0
for _, row in tqdm(original.iterrows()):
    label_col = row["primary_label"]
    filename = row["filename"]
    filepath = f"/mnt/d/Data/birds/train_audio/{filename}"
    # Load your audio.
    song = AudioSegment.from_ogg(filepath)
    chunks = split_on_silence (
        # Use the loaded audio.
        song, 
        # Specify that a silent chunk must be at least 2 seconds or 2000 ms long.
        min_silence_len = 1000,
        # Consider a chunk silent if it's quieter than -16 dBFS.
        # (You may want to adjust this parameter.)
        silence_thresh = -35,
        keep_silence = 500
    )
    final_audio = None
    # Process each chunk with your parameters 63435
    for i, chunk in enumerate(chunks):        
        if final_audio:
            final_audio = final_audio + chunk
        else:
            final_audio = chunk
        # End
    
    if not final_audio:
        final_audio = song

    final_audio = final_audio.set_frame_rate(desired_sample_rate)
    y = np.array(final_audio.get_array_of_samples())
    if final_audio.channels == 2:
        y = y.reshape((-1, 2))

    y = scipy.signal.resample(y, desired_length)
    y = np.float32(y) / tf.int16.max
    data[index, :] = y
    labels.append(label_col)
    index = index + 1
    # data = pd.DataFrame([label].extend(y), columns)
    # print(row["firstname"])

print(data.shape)

dataset = pd.DataFrame(data, columns=columns)
# dataset[columns] = data
dataset["label"] = labels

dataset.to_parquet('train.parquet')