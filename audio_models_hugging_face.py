### Audio Processing using Python

import numpy as np
import matplotlib.pyplot as plt
import librosa # Standard python lib to process audio files
from IPython.display import Audio

audio_path = "audio.mp3"
y , sr = librosa.load(audio_path,sr=None)
# Formula for calc audio file length in s = File Size / Sampling Rate (sr)
print(f"{len(y) / sr} seconds")

plt.figure(figsize=(14,5),dpi=150)
plt.plot(y);
plt.xlabel("Time - Samples")
plt.ylabel("Amplitude")

# Audio(data=y, rate=sr) # To listen to audio in output
# DFT - Discrete Fourier Transform
window = np.hanning(len(y))
windowed_input = y * window
dft = np.fft.rfft(windowed_input)

dft
plt.plot(dft)
plt.title("Discrete Fourier Transform")

amplitude = np.abs(dft)
plt.plot(amplitude)

amplitude_db = librosa.amplitude_to_db(amplitude,ref=np.max)
frequency = librosa.fft_frequencies(sr=sr, n_fft=len(y))

plt.figure(figsize=(15,4) , dpi=150)
plt.plot(frequency, amplitude_db)
plt.xlabel("Freq Hz")
plt.ylabel("Amp dB")
plt.xscale("log")

D = librosa.stft(y)
# D
D_db = librosa.amplitude_to_db(np.abs(D),ref=np.max)

plt.figure(figsize=(14,5) , dpi=150)
librosa.display.specshow(D_db,sr=sr,x_axis="time",y_axis="log")
plt.colorbar(format="%+2.0f dB")

S = librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128,fmax=8000)
S_dB = librosa.power_to_db(S,ref=np.max)

plt.figure(figsize=(14,5) , dpi=150)
librosa.display.specshow(S_dB,sr=sr,x_axis="time",y_axis="log",fmax=8000)
plt.colorbar(format="%+2.0f dB")

"""### Audio Classification"""

import transformers, torch, torchaudio
import librosa
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from IPython.display import Audio

audio_path = "audio.mp3"
y, sr = librosa.load(audio_path,sr=None)

model = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = AutoFeatureExtractor.from_pretrained(model)

result = feature_extractor(y,return_tensors="pt")
result["input_values"]

model1 = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

prediction_logits = model1(result["input_values"]).logits
# prediction_logits

# returns highest prob. value from the logits
predicted_class_ids = torch.argmax(prediction_logits,dim=-1)
model1.config.id2label   # returns the classed of ast/model , 0 - speech
predicted_class_ids.item()

model1.config.id2label[predicted_class_ids.item()]

"""### Converting Audio to Text"""

from transformers import pipeline
pipe = pipeline("automatic-speech-recognition")

pipe("audio.mp3")

"""### Convert Text to Audio"""

from transformers import pipeline
pipe = pipeline("text-to-speech")

text = "I like coding in Javascript and I am a professional web developer"
output = pipe(text)

output

plt.plot(output["audio"].squeeze())

Audio(data=output["audio"], rate=output["sampling_rate"])

from pydub import AudioSegment

audio_seg = AudioSegment(output["audio"].tobytes(),
                        frame_rate=output["sampling_rate"],
                        sample_width=output["audio"].dtype.itemsize,
                        channels=1)

audio_seg.export("my_audio_saved.mp3",format="mp3")