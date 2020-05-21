import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank #this is the library by jameslyones from practicalcryptography blog instead of librosa
import librosa

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean() #window size and min periods
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask




def calc_fft(y,rate):
    n = len(y) #y is the signal
    freq = np.fft.rfftfreq(n, d = 1/rate)
    Y = abs(np.fft.rfft(y)/n) #Y is the magnitued of the fft which is complex number
    return (Y, freq)

df = pd.read_csv('instruments.csv') #loading csv
df.set_index('fname',inplace = True)

for f in df.index:
    rate, signal = wavfile.read('wavfiles/'+f) #reading the wavfile
    df.at[f, 'length'] = signal.shape[0]/rate #in df at index f put value size of signal divided by sampling rate

classes = list(np.unique(df.label)) #getting classes from df
class_dict = df.groupby(['label'])['length'].mean() #group the files by label and find the mean leangth of the classes

# to display the number of examples for every class in pie chart
fig, ax = plt.subplots() #creating plot
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dict, labels=class_dict.index, autopct = '%1.1f%%',
       shadow = False, startangle = 90)
ax.axis('equal')
plt.show()
df.reset_index(inplace=True)

#making dictionaries for each class
signals = {}
fft = {}
fbank = {}
mfccs = {}


#to link csv file to audio files
for c in classes:
    wav_file = df[df.label == c].iloc[0,0] #whatever in this df at 0 index and 0 column
    signal, rate = librosa.load('wavfiles/'+wav_file, sr=44100) #the second parameter is our sampling rate
    #if you wish to find out the sampling rate we use here livrary (see imported) wav_file from scipi_io
    #it detects it for us
    mask = envelope(signal, rate, 0.0005) #can try changing this
    signal = signal[mask]
    signals[c] = signal
    fft[c] = calc_fft(signal, rate)

    bank = logfbank(signal[:rate], rate, nfilt=26, nfft = 1103).T
    fbank[c] = bank
    mel = mfcc(signal[:rate],rate,numcep=13, nfilt=1103).T
    mfccs[c] = mel

plot_signals(signals)
plt.show()

plot_fft(fft)
plt.show()

plot_fbank(fbank)
plt.show()

plot_mfccs(mfccs)
plt.show()

#this is to downsample are audio to 16khz per second (16000 samples per second)
if len(os.listdir('clean')) == 0:
    for f in tqdm(df.fname):
        signal, rate = librosa.load('wavfiles/'+f,sr=16000) #this will load the audio file and we are mentioning the sample rate
        mask = envelope(signal, rate, 0.0005) #this will get rid of the dead space
        #the envelope of an oscillating signal is a smooth curve outlining its extremes. The envelope thus generalizes the concept of a constant amplitude

        wavfile.write(filename='clean/'+f,rate = rate, data = signal[mask]) #this will write audio to clean folder and the name of the file is file, which are 300
                                                                            #audio file which are now cleaned, for processing. (deadspace is removed)




