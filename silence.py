# Import the AudioSegment class for processing audio and the 
# split_on_silence function for separating out silent chunks.
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import mediainfo
import numpy as np
import tensorflow as tf 
import tensorflow_hub as hub 
import tensorflow_io as tfio
import scipy
from scipy.io import wavfile

# Define a function to normalize a chunk to a target amplitude.
def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)

# Load your audio.
song = AudioSegment.from_ogg("/mnt/d/Data/birds/train_audio/abethr1/XC128013.ogg")

original_bitrate = mediainfo("/mnt/d/Data/birds/train_audio/abethr1/XC128013.ogg")
sample_rate = int(original_bitrate["sample_rate"])
desired_sample_rate = 16000

print(original_bitrate)

# Split track where the silence is 2 seconds or more and get chunks using 
# the imported function.
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
    # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
    silence_chunk = AudioSegment.silent(duration=500)
    
    if final_audio:
        final_audio = final_audio + chunk
    else:
        final_audio = chunk

    # Add the padding chunk to beginning and end of the entire chunk.
    audio_chunk = silence_chunk + chunk + silence_chunk

    # Normalize the entire chunk.
    # normalized_chunk = match_target_amplitude(audio_chunk, -20.0)
    normalized_chunk = audio_chunk

    # Export the audio chunk with new bitrate.
    print("Exporting chunk{0}".format(i))
    normalized_chunk.export(
        "spli_audio/chunk{0}.wav".format(i),
        bitrate = "63k",
        format = "wav"
    )
# 63435
final_audio = final_audio.set_frame_rate(desired_sample_rate)
final_audio.export("spli_audio/final_audio.wav", bitrate = "196k", format = "wav")
y = np.array(final_audio.get_array_of_samples())
if final_audio.channels == 2:
    y = y.reshape((-1, 2))

desired_length = int(round(float(len(y)) / sample_rate * desired_sample_rate))
desired_length = 10000

print(desired_length)
y = scipy.signal.resample(y, desired_length)
y = np.float32(y) / tf.int16.max
print(tf.int16.max, 2**15)
print(final_audio.frame_rate, y.shape, y)