'''
File    : test.py
Author  : Jay Santokhi (jks1g15)
Brief   : Carries out predictions on test set using a specified model
Usage   : python test.py
'''
from Data_Utils import wav_to_spectrogram
from Data_Utils import load_wav_files
from Data_Utils import wav_to_MFCC
from keras.models import load_model
from sklearn.metrics import classification_report

if __name__ == "__main__":
    test_path = 'Dataset_edit/test/'
    n_wav_files, wav_files, class_labels = load_wav_files(test_path)

    testX, testY = wav_to_spectrogram(n_wav_files, wav_files, class_labels, 15000, 5234)
    # testX, testY = wav_to_MFCC(n_wav_files, wav_files, class_labels, 15000, 5234)
    print(testX.shape)
    print(testY.shape)

    # model = load_model('Model.hdf5')
    model = load_model('Model-20-0.67.hdf5')
    print(model.summary())

    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY, predictions.argmax(axis=1)))
