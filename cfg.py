import os

class Config:
    def __init__(self, mode='conv',nfilt=26, nfeat=13,nfft=512, rate=16000):#here nfilt is no of filter, nfeat is number of features and nfft is number of fft which is half of what was before because we have downsampled the audio
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10) #This is the 10th of a data file, we shoudn't just take a step and reach the end of the file
        self.model_path = os.path.join('models', mode + '.model') #.model extension for pickles
        self.p_path = os.path.join('pickles', mode + '.p') #.p extension for pickles