import os 
import csv

# import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import tensorflow as tf 
import tensorflow_hub as hub 
import tensorflow_io as tfio
import scipy
from scipy.io import wavfile
import soundfile as sf

# Utility functions for loading audio files and making sure the sample rate is correct.

@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    print(sample_rate)
    wav = tf.squeeze(wav, axis=-1)
    print(wav)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    print(sample_rate)
    # wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def load_wav_16k_data(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    testing_wav_data_1 = tfio.audio.AudioIOTensor(filename, dtype=tf.float32)
    audioTensor = tf.squeeze(testing_wav_data_1.to_tensor(), axis=-1)
    print(audioTensor)
    audioTensor = tf.cast(audioTensor, tf.float32) / (32768.0)
    audioTensor = tfio.audio.resample(
        audioTensor,
        rate_in=tf.cast(testing_wav_data_1.rate, tf.int64),
        rate_out=16000
    )
    return audioTensor

def load_wav_32k_data(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    testing_wav_data_1 = tfio.audio.AudioIOTensor(filename, dtype=tf.float32)
    print(testing_wav_data_1.rate)
    audioTensor = tf.squeeze(testing_wav_data_1.to_tensor(), axis=-1)
    audioTensor = tf.cast(audioTensor, tf.float32) / 65536.0
    audioTensor = tfio.audio.resample(
        audioTensor,
        rate_in=tf.cast(testing_wav_data_1.rate, tf.int64),
        rate_out=16000
    )
    return audioTensor

def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform
  
def load_waveform(filename):
    sample_rate, wav_data = wavfile.read(filename, 'rb')
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
    print(tf.int16.max)
    waveform = wav_data / tf.int16.max
    return waveform

# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])

    return class_names

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1' 
yamnet_model = hub.load(yamnet_model_handle)
# yamnet_model = hub.KerasLayer(yamnet_model_handle, trainable=True)
print(yamnet_model)
print("------------------------------------------------------------------------------")
# print(yamnet_model.get_signature_names())

# testing_wav_file_name = tf.keras.utils.get_file('miaow_16k.wav', 'https://storage.googleapis.com/audioset/miaow_16k.wav', cache_dir='./', cache_subdir='test_data')


# testing_wav_file_name = "/home/koushik/brain-egg/test_data/1-34495-A-14.wav"
testing_wav_file_name = "spli_audio/final_audio.wav"
print(testing_wav_file_name)

testing_wav_data = load_waveform(testing_wav_file_name)
print(testing_wav_data)
print(len(testing_wav_data))
print("------------------------------------------------------------------------------")
# wav_file_name = "/mnt/d/Data/birds/train_audio/abethr1/XC128013.ogg"
# testing_wav_data = load_wav_16k_data(testing_wav_file_name)
# print(testing_wav_data)

# print("------------------------------------------------------------------------------")


# testing_wav_data_1 = tfio.audio.AudioIOTensor(testing_wav_file_name, dtype=tf.int16)
# print(testing_wav_data_1)
# chanalsIn = testing_wav_data_1.shape[1]
# sample_rate = testing_wav_data_1.rate.numpy()
# sample_rate = tf.cast(sample_rate, dtype=tf.int64)
# audioTensor = tf.cast(tf.squeeze(testing_wav_data_1.to_tensor()), tf.float32) / 32768.0
# audioTensor = testing_wav_data_1[0:]
# audioTensor = tf.squeeze(audioTensor, axis=-1)
# print(audioTensor)
# print(sample_rate)
# audioTensor = tfio.audio.resample(audioTensor, rate_in=sample_rate, rate_out=16000)
# print(audioTensor)
# class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
# class_names = list(pd.read_csv(class_map_path)['display_name'])

class_map_path = yamnet_model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

# for name in class_names:
#   print(name)
# print('...')

scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
scores_np = scores.numpy()
spectrogram_np = spectrogram.numpy()
infered_class = class_names[scores_np.mean(axis=0).argmax()]

# class_scores = tf.reduce_mean(scores, axis=0)
# print(class_scores)
# top_class = tf.math.argmax(class_scores)
# inferred_class = class_names[top_class]

print(f'The main sound is: {infered_class}')
print(f'The embeddings shape: {embeddings.shape}')