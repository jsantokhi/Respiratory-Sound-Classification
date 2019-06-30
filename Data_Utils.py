'''
File    : Data_Utils.py
Author  : Jay Santokhi (jks1g15)
Brief   : Contains functions for loading data and pre-processing
Usage   : python Data_Utils.py
'''
from python_speech_features import logfbank
from python_speech_features import delta
from python_speech_features import mfcc

from scipy.io import wavfile
from matplotlib import cm
from scipy import signal

import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import math
import sys
import os


def load_wav_files(path):
    ''' Takes file path to wav audio, extracts no of files, filenames and class

    Arguments:
        path -- File path to audio

    Returns:
        n_wav_files -- The number of wav files
        wav_file_names -- The names of the wav files
        class_labels -- The class labels of the wav files
    '''
    wav_file_names = []
    class_labels = []
    n_wav_files = 0

    # Assigns number values to the class labels
    class_name_to_id = {"Asthma": 0, "Bronchiectasis": 1, "Bronchiolitis": 2,
                        "COPD": 3, "Healthy": 4, "LRTI": 5, "Pneumonia": 6,
                        "URTI": 7}

    nclasses = len(class_name_to_id.keys())

    # Extract relevant information
    for root, dirs, files in os.walk(path):
        c = '_'
        for file in files:
            if file.endswith('.wav'):
               index = [pos for pos, char in enumerate(file) if char == c]
               base_file_name = file.rstrip(".wav")
               class_label = base_file_name[0:index[0]]
               # print(class_label)
               class_labels.append(class_name_to_id[class_label])
               wav_file_names.append(os.path.join(root, file))

               n_wav_files += 1

    return n_wav_files, wav_file_names, class_labels


def get_training_seg(n_wav_files, wav_file_names, seg_size):
    ''' Calculates the no of training segments given the segment size

    Arguments:
        n_wav_files -- The number of wav files
        wav_file_names -- The names of the wav files
        class_labels -- The class labels of the wav files
        seg_size -- Desired segment size

    Returns:
        i -- The number of training segments

    Note: Only use to find the number and then hardcode this number to other
          function arguments afterwards.
    '''
    i = 0
    index = 0
    size = seg_size

    for idx, wavfname in enumerate(wav_file_names):
        data, rate = sf.read(wavfname)
        l = int(len(data)/size)
        i =  i + l
        length = len(data)
        nseg = int(len(data)/size)
        # print(idx, wavfname, class_labels[idx],length, l, i,index)
    return i


def graph_spectrogram(wav_file, seg_size):
    ''' Take wav audio file and graph the spectrogram

    Arguments:
        wav_file -- Desired audio to turn into a spectrogram (wav format)

    Returns:
        pxx -- Spectrum (Columns are the periodograms of successive segments)
    '''
    data, rate = sf.read(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequency
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, t, im = plt.specgram(data[0:seg_size], nfft, fs, noverlap=noverlap, mode='psd')
    elif nchannels == 2:
        pxx, freqs, t, im = plt.specgram(data[:, 0:seg_size], nfft, fs, noverlap=noverlap, mode='psd')

    plt.xlabel('Time in seconds')
    plt.ylabel('Frequency Hz')
    plt.title('Spectrogram')
    plt.show()
    return pxx


def wav_to_spectrogram(n_wav_files, wav_file_names, class_labels, seg_size, training_segs):
    ''' Create spectrograms for all wav files specified

    Arguments:
        n_wav_files -- The number of wav files
        wav_file_names -- The names of the wav files
        class_labels -- The class labels of the wav files
        seg_size -- Desired segment size
        training_segs -- No of segments (value from 'get_training_seg()')

    Returns:
        X -- Data with shape [training_segs, 101, shape, 1]
        Y -- Class labels of X
    '''
    # This value is determined from 'pxx.shape[1]' of graph_spectrogram for a
    # single audio recording of the desired segment size
    shape = 149
    X = np.zeros([training_segs, 101, shape, 1])
    Y = np.zeros([training_segs])

    i = 0
    index = 0
    size = seg_size

    for idx, wavfname in enumerate(wav_file_names):
       data, rate = sf.read(wavfname)
       l = int(len(data)/size)
       i =  i + l

       length = len(data)
       nseg = int(len(data)/size)

       if(nseg > 1):
           seg = 0
           for seg in range(nseg):
               segment = data[seg:seg+size]
               _, _, Sxx = signal.spectrogram(segment, fs=8000, nperseg=200,
                                              noverlap=100, mode='magnitude')
               # Sxx, _, _, _ = plt.specgram(segment, 200, 8000, noverlap=100,
               #                             mode='psd')
               sxx = Sxx.reshape(101, shape, 1)
               #graph_spectrogram(segment)
               X[index,:] = sxx
               Y[index] =  class_labels[idx]

               seg = seg + size
               index = index + 1
       else:
           segment = data[0:size]
           _, _, Sxx = signal.spectrogram(segment, fs=8000, nperseg=200,
                                          noverlap=100,mode='magnitude')
           # Sxx, _, _, _ = plt.specgram(segment, 200, 8000, noverlap=100,
           #                             mode='psd')
           sxx = Sxx.reshape(101,shape,1)

           X[index,:] = sxx
           Y[index] =  class_labels[idx]

           index = index + 1

       idx += 1
    return X, Y


def graph_MFCC(wav_file, seg_size):
    ''' Take wav audio file and graph the MFCC heat map

    Arguments:
        wav_file -- Desired audio to turn into a spectrogram (wav format)

    Returns:
        mfcc_feat -- MFCC Features (Frames x MFCC Coefficients)
    '''
    data, rate = sf.read(wav_file)
    winlen = 0.025
    window_step = 0.01
    fs = 8000
    nchannels = data.ndim
    if nchannels == 1:
        mfcc_feat = mfcc(data[0:seg_size], fs, winlen, window_step, numcep=13)
        # print(mfcc_feat.shape)
    elif nchannels == 2:
        mfcc_feat = mfcc(data[:,0:seg_size], fs, winlen, window_step, numcep=13)

    # print(mfcc_feat.shape)
    ig, ax = plt.subplots()
    mfcc_data= np.swapaxes(mfcc_feat, 0 ,1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.jet,
                    origin='lower', aspect='auto')
    plt.ylabel('MFCC-Coefficientsc')
    plt.xlabel('Frames')
    plt.title('Mel Spectrogram (MFCC Heat Map)')
    plt.show()
    # plt.plot(mfcc_feat)
    # plt.show()
    return mfcc_feat


def wav_to_MFCC(n_wav_files, wav_file_names, class_labels, seg_size, training_segs):
    ''' Create MFCC heat maps for all wav files specified

    Arguments:
        n_wav_files -- The number of wav files
        wav_file_names -- The names of the wav files
        class_labels -- The class labels of the wav files
        seg_size -- Desired segment size
        training_segs -- No of segments (value from 'get_training_seg()')

    Returns:
        X -- Data with shape [training_segs, 186, 13, 1]
        Y -- Class labels of X
    '''
    shape = 186
    X = np.zeros([training_segs, shape, 13, 1])
    Y = np.zeros([training_segs])

    winlen = 0.025
    window_step = 0.01

    fs = 8000
    duration = 10
    current_frame = 0

    i = 0
    index = 0
    size = seg_size

    for idx, wavfname in enumerate(wav_file_names):
       data, rate = sf.read(wavfname)
       l = int(len(data)/size)
       i =  i + l

       length = len(data)
       nseg = int(len(data)/size)

       if(nseg > 1):
           seg = 0
           for seg in range(nseg):
               segment = data[seg:seg+size]
               mfcc_feat = mfcc(segment, fs, winlen, window_step, numcep=13)
               mfcc_feat = mfcc_feat.reshape(shape, 13, 1)

               X[index,:] = mfcc_feat
               Y[index] = class_labels[idx]

               seg = seg + size
               index = index + 1
       else:
           segment = data[0:size]
           mfcc_feat = mfcc(segment, fs, winlen, window_step, numcep=13)
           mfcc_feat = mfcc_feat.reshape(shape, 13, 1)

           X[index,:] = mfcc_feat
           Y[index] = class_labels[idx]

           index = index + 1

       idx += 1
    return X, Y


# Check everything is working
if __name__ == "__main__":
    seg_size = 15000

    # train_path = 'Dataset/train/'
    # train_path = 'Dataset_edit/train/'
    # # train_path = 'ALL_TEST_Cycle_based_Train_Test_Val_Split/train'
    # # train_path = 'NO_ASTHMA_TEST_Cycle_based_Train_Test_Val_Split/train'
    # # train_path = 'NO_ASTHMA_LRTI_TEST_Cycle_based_Train_Test_Val_Split/train'
    # n_wav_files, wav_files, class_labels = load_wav_files(train_path)
    # # t = print(get_training_seg(n_wav_files, wav_files, seg_size))
    # # print(t)
    # trainX,trainY = wav_to_spectrogram(n_wav_files, wav_files, class_labels, seg_size, 3894)
    # # trainX,trainY = wav_to_MFCC(n_wav_files, wav_files, class_labels, seg_size, 3894)
    # print(trainX.shape)
    # print(trainY.shape)

    # valid_path = 'Dataset/valid/'
    valid_path = 'Dataset_edit/valid/URTI_0.wav'
    # valid_path = 'ALL_TEST_Cycle_based_Train_Test_Val_Split/valid'
    # valid_path = 'NO_ASTHMA_TEST_Cycle_based_Train_Test_Val_Split/valid'
    # valid_path = 'NO_ASTHMA_LRTI_TEST_Cycle_based_Train_Test_Val_Split/valid'

    pxx = graph_spectrogram(valid_path, seg_size)
    mfcc_feat = graph_MFCC(valid_path, seg_size)
    # print(pxx.shape)
    # print(mfcc_feat.shape)
    # n_wav_files, wav_files, class_labels = load_wav_files(valid_path)
    # # t = get_training_seg(n_wav_files, wav_files, seg_size)
    # # print(t)
    # validX, validY = wav_to_spectrogram(n_wav_files, wav_files, class_labels, seg_size, 2788)
    # # validX, validY = wav_to_MFCC(n_wav_files, wav_files, class_labels, seg_size, 2788)
    # print(validX.shape)
    # print(validY.shape)

    # test_path = 'Dataset/test/'
    # test_path = 'Dataset_edit/test/'
    # n_wav_files, wav_files, class_labels = load_wav_files(test_path)
    # # t = get_training_seg(n_wav_files, wav_files, seg_size)
    # # print(t)
    # testX,testY = wav_to_spectrogram(n_wav_files, wav_files, class_labels, seg_size, 5234)
    # # testX,testY = wav_to_MFCC(n_wav_files, wav_files, class_labels, seg_size, 5234)
    # print(testX.shape)
    # print(testY.shape)
