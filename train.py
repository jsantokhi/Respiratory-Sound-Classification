'''
File    : train.py
Author  : Jay Santokhi (jks1g15)
Brief   : Carries out training using either Spectrogram or MFCC inputs
Usage   : python train.py -p <'Spectrogram' or 'MFCC'> -c <'BoVW' or 'NN'>
'''
from Data_Utils import wav_to_spectrogram
from Data_Utils import load_wav_files
from Data_Utils import wav_to_MFCC
from models import NN_classifier
from models import BoVW_classifier

import argparse
import sys


def define_args():
    ''' Defines the script arguments.
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--PreProcessing", required=True,
                    help=" 'Spectrogram' or 'MFCC' ")
    ap.add_argument("-c", "--Classifier", required=True,
                    help=" 'BoVW' or 'NN' ")
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = define_args()
    pp = args["PreProcessing"]
    classifier = args["Classifier"]

    # train_path = 'Dataset/train/'
    # valid_path = 'Dataset/valid/'

    train_path = 'Dataset_edit/train/'
    valid_path = 'Dataset_edit/valid/'
    test_path = 'Dataset_edit/test/'

    # train_path = 'NO_ASTHMA_LRTI_TEST_Cycle_based_Train_Test_Val_Split/train'
    # valid_path = 'NO_ASTHMA_LRTI_TEST_Cycle_based_Train_Test_Val_Split/valid'

    # trainX,trainY = wav_to_MFCC(n_wav_files_t, wav_files_t, class_labels_t, 15000, 2966)
    # validX, validY = wav_to_MFCC(n_wav_files_v, wav_files_v, class_labels_v, 15000, 642)
    # trainX,trainY = wav_to_spectrogram(n_wav_files_t, wav_files_t, class_labels_t, 15000, 2966)
    # validX, validY = wav_to_spectrogram(n_wav_files_v, wav_files_v, class_labels_v, 15000, 642)

    if pp == 'MFCC':
        print('Using MFCC')
        print('Loading training set')
        n_wav_files_t, wav_files_t, class_labels_t = load_wav_files(train_path)
        trainX, trainY = wav_to_MFCC(n_wav_files_t, wav_files_t, class_labels_t, 15000, 3894)
        print(trainX.shape)
        print(trainY.shape)

        print('Loading validation set')
        if classifier == 'BoVW':
            n_wav_files_v, wav_files_v, class_labels_v = load_wav_files(test_path)
            testX, testY = wav_to_MFCC(n_wav_files_v, wav_files_v, class_labels_v, 15000, 5234)
            print(testX.shape)
            print(testY.shape)
        elif classifier == 'NN':
            n_wav_files_v, wav_files_v, class_labels_v = load_wav_files(valid_path)
            validX, validY = wav_to_MFCC(n_wav_files_v, wav_files_v, class_labels_v, 15000, 2788)
            print(validX.shape)
            print(validY.shape)

    elif pp == 'Spectrogram':
        print('Using Spectrogram')
        print('Loading training set')
        n_wav_files_t, wav_files_t, class_labels_t = load_wav_files(train_path)
        trainX, trainY = wav_to_spectrogram(n_wav_files_t, wav_files_t, class_labels_t, 15000, 3894)
        print(trainX.shape)
        print(trainY.shape)

        print('Loading validation set')
        if classifier == 'BoVW':
            n_wav_files_v, wav_files_v, class_labels_v = load_wav_files(test_path)
            testX, testY = wav_to_spectrogram(n_wav_files_v, wav_files_v, class_labels_v, 15000, 5234)
            print(testX.shape)
            print(testY.shape)
        elif classifier == 'NN':
            n_wav_files_v, wav_files_v, class_labels_v = load_wav_files(valid_path)
            validX, validY = wav_to_spectrogram(n_wav_files_v, wav_files_v, class_labels_v, 15000, 2788)
            print(validX.shape)
            print(validY.shape)

    if classifier == 'BoVW':
        BoVW_classifier(trainX, trainY, testX, testY)
    elif classifier == 'NN':
        NN_classifier(trainX, trainY, validX, validY, pp)
