import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from sklearn.metrics import accuracy_score
from keras.models import load_model

# the function looks in the directory passed in parameter and makes prediction on all
# audio files present in the directory
def build_predictions(audio_dir):
    y_true = [] #true class
    y_pred = [] #the predictions from the neural network
    fn_prob = {}

    print('Extracting features from audio') #this will actually do the prediction
    for fn in tqdm(os.listdir(audio_dir)): #this will itereate through the audio directory
        rate, wav = wavfile.read(os.path.join(audio_dir, fn)) #getting the specific audio file
        label = fn2class[fn] #will calculate the accuracy metric
        c = classes.index(label)
        y_prob = []
        print(c)

    #     for i in range(0, wav.shape[0]-config.step, config.step):
    #         sample = wav[i:i+config.step]
    #         x = mfcc(sample, rate,numcep=config.nfeat,
    #                  nfilt = config.nfilt, nfft = config.nfft)
    #         x = (x - config.min) / (config.max - config.min)
    #
    #         if config.mode == 'conv':
    #             x = x.reshape(1,x.shape[0], x.shape[1],1)
    #         elif config.mode == 'time':
    #             x = np.expand_dims(x, axis = 0)
    #         y_hat = model.predict(x)
    #         y_prob.append(y_hat)
    #         y_prob.append(np.argmax(y_hat))
    #         y_true.append(c)
    #
    #     fn_prob[fn] = np.mean(y_prob, axis = 0).flatten()
    #
    # return y_true, y_pred, fn_prob



df = pd.read_csv('instruments.csv') #reading the csv file
classes = list(np.unique(df.label)) #creating classes because we are testing on training data
                                    #remove this if we don't know the classes.
fn2class = dict(zip(df.fname,df.label)) #calculates accuracy metric
p_path = os.path.join('pickles', 'conv.p') #this will decide what model do we want to run

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)  #this will store our config

model = load_model(config.model_path) #this will load our model from keras

# y_true, y_pred, fn_pro = build_predictions('clean') #this will pull audio data from clean directory
# acc_store = accuracy_score(y_true = y_true, y_pred=y_pred)
#
# y_probs = []
# for i, row in df.iterrows():
#     y_prob = fn_prob[row.fname]
#     y_prob.append(y_prob)
#     for c, p in zip(classes, y_prob):
#         df.at[i, c] =  p
#
# y_pred = [classes[np.argmax(y)] for y in y_probs]
# df['y_pred'] = y_pred
#
# df.to_csv('predictions.csv', index = False)