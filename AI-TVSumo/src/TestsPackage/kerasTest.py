from keras.datasets import reuters
from keras import models
from keras import layers

import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical


def main():
    (train_data, train_labels),(test_data, test_labels) = reuters.load_data(num_words=10000)
    
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    ohTrainLabels = to_categorical(train_labels)
    ohTestLables = to_categorical(test_labels)
    
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    y_val = ohTrainLabels[:1000]
    partial_y_train = ohTrainLabels[1000:]
        
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(partial_x_train, partial_y_train, epochs=9, 
                        batch_size=512, validation_data=(x_val, y_val))
    
    print(model.evaluate(x_test, ohTestLables))
    plotHistory(history)    


def plotHistory(history):
    historyDict = history.history
    lossValues = historyDict['loss']
    valLossValues = historyDict['val_loss']
    
    epochs = range(1, len(lossValues)+1)
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, lossValues, 'bo', label='Training Loss')
    plt.plot(epochs, valLossValues, 'b', label='Validation Loss')
    plt.title("Training and Validation Loss and Accuracy")
#     plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    acc = historyDict['acc']
    valAcc = historyDict['val_acc']
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'bo', label='Training Accuracy')
    plt.plot(epochs, valAcc, 'b', label='Validation Accuracy')
#     plt.title("Training and Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()


def decodeToText(textSequence):
    wordIndex = reuters.get_word_index()
    reverseWordIndex = dict([(value,key) for (key,value) in wordIndex.items()])
    decodedText = ' '.join([reverseWordIndex.get(i-3, '?') for i in textSequence])
    return decodedText


def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


if __name__ == '__main__':
    main()