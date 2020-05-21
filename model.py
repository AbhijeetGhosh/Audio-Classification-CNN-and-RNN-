import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import os
import pickle
from keras.callbacks import ModelCheckpoint # this will help save model from keras to implement on new predictions
from cfg import Config #this file will contain the Config class that we created earlier

def check_data(): #this will look in pickles folder and see there is an existing file in there
    if os.path.isfile((config.p_path)):
        print('Loading existing data for {} model'.format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

#the function thats gonna build all of our data so it will be preprocessed to push through our model
def build_rand_feat():
    tmp = check_data()
    if tmp:
        return  tmp.data[0], tmp.data[1]
    X = []
    y = []
    _min, _max = float('inf'), -float('inf') #maximum float value and minimum float value
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p = prob_dist)
        file = np.random.choice(df[df.label == rand_class].index) #this is another level of randomnes apart from above
        rate, wav = wavfile.read('clean/'+file) #reading the file
        label = df.at[file, 'label']
        rand_index = np.random.randint(0, wav.shape[0] - config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample, rate,
                        numcep=config.nfeat, nfilt = config.nfilt, nfft = config.nfft)
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample)
        y.append(classes.index(label)) #we are associating integer index to our string labels or string classe
    config.min = _min
    config.max = _max
    X,y = np.array(X), np.array(y) #converting in numpy array
    X = (X-_min)/(_max - _min) #normalizing
    if config.mode == 'conv': #for solving through cnn
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time': #for solving through rnn
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes=10)
    config.data = (X, y)

    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)


    return X,y

def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', strides=(1,1),
                     padding='same', input_shape=input_shape))#usually we use alternate convolutional layer and pooling layer but since our data deals with only 1/10 of a second we will do pooling only once
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))  #
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    return  model

def get_recurrent_model():
    #shape of data for RNN is (n, time, feat)
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape = input_shape)) #LSTM is just like a Dense layer, here it's just like cnn's neurals but they have LSTM built into them
    #return_sequences is the output of LSTMs.
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    return  model

# we are storing this class in cfg.py instead
#class to configure the parameters which has two modes, cnn and rnn. Config will help us switch between cnn and rnn
# class Config:
#     def __init__(self, mode='conv',nfilt=26, nfeat=13,nfft=512, rate=16000):#here nfilt is no of filter, nfeat is number of features and nfft is number of fft which is half of what was before because we have downsampled the audio
#         self.mode = mode
#         self.nfilt = nfilt
#         self.nfeat = nfeat
#         self.nfft = nfft
#         self.rate = rate
#         self.step = int(rate/10) #This is the 10th of a data file, we shoudn't just take a step and reach the end of the file

#this portion is from eda.py file, the only difference is we are getting audio from clean file
#in clean folder we downsampled our audio and removed dead spots
df = pd.read_csv('instruments.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('clean/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

n_samples = 2*int(df['length'].sum()/0.1) #we are taking a sample every 10th of a second
prob_dist = class_dist/class_dist.sum() #this will convert everything between 0 and 1 (probability)
choices = np.random.choice(class_dist.index, p = prob_dist) #no clue what this does

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()


config = Config(mode='time') #use mode = 'conv' for cnn and mode = 'time' for rnn


#what happens if it runs in convolution
if config.mode == 'conv':
    X,y = build_rand_feat() #this will form the feature set from the random sampling done above
    y_flat = np.argmax(y, axis=1) #we are returning the hot encoded index value of labels to the original string labels
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model()

elif config.mode == 'time': #for reccurent neural network
    X,y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)  # we are returning the hot encoded index value of labels to the original string labels
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()

class_weight = compute_class_weight('balanced',
                                    np.unique(y_flat),
                                    y_flat)

checkpoint = ModelCheckpoint(config.model_path, monitor = 'val_acc', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=1)

model.fit(X, y, epochs=10, batch_size=32,
          shuffle=True, validation_split=0.1,
          callbacks=[checkpoint]) #class weight calculated above

model.save(config.model_path)



