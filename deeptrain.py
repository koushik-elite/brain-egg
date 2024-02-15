import os 

# import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import tensorflow as tf 
import tensorflow_hub as hub 
import tensorflow_io as tfio

# Utility functions for loading audio files and making sure the sample rate is correct.

@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    # audio = tfio.audio.AudioIOTensor(filename, dtype=tf.float32)
    # waveform = tf.contrib.ffmpeg.decode_audio(audio_binary, file_format='mp3', samples_per_second=44100, channel_count=2)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    # sample_rate = audio.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    print(sample_rate)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def load_wav_16k_data(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    testing_wav_data_1 = tfio.audio.AudioIOTensor(filename, dtype=tf.float32)
    audioTensor = tf.squeeze(testing_wav_data_1.to_tensor(), axis=-1)
    audioTensor = tf.cast(audioTensor, tf.float32) / 32768.0
    audioTensor = tfio.audio.resample(
        audioTensor,
        rate_in=tf.cast(testing_wav_data_1.rate, tf.int64),
        rate_out=16000
    )
    return audioTensor

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1' 
# yamnet_model = hub.load(yamnet_model_handle)
yamnet_model = hub.KerasLayer(yamnet_model_handle, trainable=True)
print(yamnet_model)
print("------------------------------------------------------------------------------")
# print(yamnet_model.get_signature_names())

# testing_wav_file_name = tf.keras.utils.get_file('miaow_16k.wav', 'https://storage.googleapis.com/audioset/miaow_16k.wav', cache_dir='./', cache_subdir='test_data')
# print(testing_wav_file_name)

# testing_wav_data = load_wav_16k_mono(testing_wav_file_name)
# print(testing_wav_data)
print("------------------------------------------------------------------------------")
# wav_file_name = "/mnt/d/Data/birds/train_audio/abethr1/XC128013.ogg"
# testing_wav_data = load_wav_16k_data(wav_file_name)
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

# for name in class_names[:20]:
#   print(name)
# print('...')

# scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
# class_scores = tf.reduce_mean(scores, axis=0)
# top_class = tf.math.argmax(class_scores)
# inferred_class = class_names[top_class]

# print(f'The main sound is: {inferred_class}')
# print(f'The embeddings shape: {embeddings.shape}')