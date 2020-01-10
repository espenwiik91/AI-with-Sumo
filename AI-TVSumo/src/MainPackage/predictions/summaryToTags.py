import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import keras_metrics as km #Denne ser ikke ut til aa funke saerlig bra
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense


def main():
    df1 = pd.read_csv("../../../Data/SumoData.csv", sep=";")
    df2 = pd.read_csv("../../../Data/SumoAssetMetaData.csv", sep=",")
    print(f"Shape of first dataframe: {df1.shape}")
    print(f"Shape of second dataframe: {df2.shape}")
    df3 = pd.concat([df1, df2], ignore_index="true", join="inner")
    print(f"Shape of combined dataframe: {df3.shape}")
    print("Dropping duplicate summaries and shuffling dataframe...")
    df3 = df3.drop_duplicates(subset= 'summary')
    df3 = df3.sample(frac=1).reset_index(drop=True)
    print(f"Shape of dataframe with duplicate summaries removed: {df3.shape}")
    summaries = df3['summary']
    print(f"Number of summaries: {summaries.shape}")
    
    #Input dataframe har noen nan-verdier paa summery av en eller annen grunn,
    #disse fikk jeg ikke til aa endre, saa vi tar dem heller bare bort    
    index = 0
    toDrop = []
    for summary in summaries:
        if type(summary) != str:
            toDrop.append(index)
        index = index+1
    df3 = df3.drop(toDrop, axis=0)
    print(f"Removed {len(toDrop)} unclean summary entries")
    #Reinstansier summaries etter at vi har fjernet missing values
    summaries = df3['summary']
    print(f"New number of summaries {summaries.shape}")
    
    #Tokenizing for summary data
    suma_tokenizer = Tokenizer()
    print("Fitting Tokenizer on input summary texts...")
    suma_tokenizer.fit_on_texts(summaries)
    print("Transforming summary texts to sequences...")
    sequences = suma_tokenizer.texts_to_sequences(summaries)
    word_index = suma_tokenizer.word_index
    print(f'Found {len(word_index)} unique tokens for summary corpus')
    #Padding slik at alle sequences er like lange 
    print("Padding sequences...")
    summary_seq = pad_sequences(sequences)
    print('Shape of summary sequences:', summary_seq.shape)
    input_length = len(summary_seq[0])
    
    #Tokenizer for tags, denne er ulik fordi den splitter paa komma istedetfor mellomrom
    raw_tags = df3['tagsList']
    print(f"Number of tag sets: {raw_tags.shape}")
    print("Fitting special tokenizer for tag data including only the top 100 words...")
    #Vi tar bare med de 100 mest vanlige tagsene
    tag_tokenizer = Tokenizer(num_words=100, split=',')
    tag_tokenizer.fit_on_texts(raw_tags)
    print(f"Found {len(tag_tokenizer.word_index)} unique tags")
    #texts_to_matrix i binary-mode lager multi-hot, dvs. at vi mister rekkefoelgen paa ordene,
    #men vi slipper unna med kun en vector per sample
    print("Creating Multi-Hot matrix...")
    multi_hot_tags = tag_tokenizer.texts_to_matrix(raw_tags, mode='binary')
    print(f"Result is matrix of shape: {multi_hot_tags.shape}")
    output_length = len(multi_hot_tags[0])
    
#     nrAllZeroes = 0
#     for sequence in multi_hot_tags:
#     allZero = True
#     for i in sequence:
#         if i == 1:
#             allZero = False
#     if allZero:
#         nrAllZeroes +=1
    
    #16'000 samples til trening, 10 til manuell visuell kontroll, resten til testdata
    #Hardkodet antall fordi jeg ikke gidder aa implementere en utregning av passende trening/test-split
    x_train = summary_seq[:16000]
    y_train = multi_hot_tags[:16000]
    topredict = 10
    x_test = summary_seq[16000:-topredict]
    y_test = multi_hot_tags[16000:-topredict]
    
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 64, input_length=input_length))
    model.add(Flatten())
    #1xDense a 64 dimensjoner fungerer greit nok, men vi boer utforske mer
    #avanserte alternativer som RNN/1Dconvnet
    model.add(Dense(64, activation='relu'))
    #Output-layer med antall dimensjoner lik antall unike tags
    model.add(Dense(output_length, activation='sigmoid'))
    #loss= kan settes til 'mse' eller 'binary_crossentropy'
    #precision and recall implementert fra gammel Keras-kildekode
    model.compile(optimizer='rmsprop', loss='mse', metrics=[precision, recall])
    #Epochs er satt til 20 for aa teste ulike modell-arkitekturer, det holder nok med faerre
    history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_split=0.2)
    
    #Evaluering paa testdata minus 10 samples som kommer under
    print("Evalating model on test data...")
    print(model.evaluate(x_test, y_test))
    print('')
    
    #Printer prediksjoner av 10 usette samples fra testdata, med targets til sammenligning 
    print("Predictions on 10 unused entries from test set:")
    predict_set = x_test[-topredict:]
    answer_set = y_test[-topredict:]
    predictions = model.predict(predict_set)
    
    index = 0
    for sequence in predict_set:
        print('')
        print("Input text:")
        print(decodeSequenceToText(sequence, suma_tokenizer.word_index))
        (words, confidences) = decodePredToTags(predictions[index], tag_tokenizer.word_index)
        print("Confidence:")
        print(confidences)
        print("Predicted tags:")
        print(words)
        (targets, ones) = decodePredToTags(answer_set[index], tag_tokenizer.word_index)
        print("Target tags:")
        print(targets)
        index +=1
    #My pyplots bring all the bois to the yard
    plotHistory(history)

#Tar en integer-sekvens som representerer en tekst pluss tekstens tilhoerende wordIndex
#og returnerer teksten oversatt fra integer-sekvens til tekst-streng
def decodeSequenceToText(sequence, wordIndex):
    reverseWordIndex = dict([(value,key) for (key,value) in wordIndex.items()])
    words = []
    for i in sequence:
        #0 er brukt til padding
        if i !=0:
            words.append(reverseWordIndex.get(i) + " ")
    decodedText = ''.join(words)
    return decodedText

#Tar en sekvens med tag-prediksjoner og tilhoerende wordIndex og
#returnerer en string med ord pluss en liste med tilhoerende confidence scores
#avhengig av en threshold som definerer hvor hoey confidence maa vaere for at ordet skal telles
def decodePredToTags(sequence, wordIndex):
    reverseWordIndex = dict([(value,key) for (key,value) in wordIndex.items()])
    words = []
    confidences = []
    index = 0
    for value in sequence:
        #Tallet her presenterer hvor sikker prediksjonen er
        #Jo lavere, jo flere forskjellige tags inkluderes i "Predicted tags:" i konsollen
        if value > 0.3 :
            words.append(reverseWordIndex.get(index) + " ")
            confidences.append(value)
        index = index+1
    decodedText = ''.join(words)
    return decodedText, confidences

#plotHistory produserer en 1x3 plot med loss, precision, og recall, for trening og validation
def plotHistory(history):
    historyDict = history.history
    
    lossValues = historyDict['loss']
    valLossValues = historyDict['val_loss']
    pre = historyDict['precision']
    valPre = historyDict['val_precision']
    rec = historyDict['recall']
    valRec = historyDict['val_recall']
    
    epochs = range(1, len(pre)+1)
    
    plt.subplot(3, 1, 1)
    plt.plot(epochs, lossValues, 'bo', label='Training')
    plt.plot(epochs, valLossValues, 'b', label='Validation')
    plt.title("Training and Validation Metrics")
#     plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(epochs, pre, 'bo', label='Training')
    plt.plot(epochs, valPre, 'b', label='Validation')
    plt.ylabel('Precision')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(epochs, rec, 'bo', label='Training')
    plt.plot(epochs, valRec, 'b', label='Validation')
    plt.ylabel('Recall')
    plt.legend()
    
    plt.xlabel('Epochs')
    plt.show()

#Precision og Recall kopiert fra en commit fra fChollet paa Github
#disse fantes innebygd i Keras tidligere, men ble fjernet i versjon 2.0
#fordi de kan vaere misledende, fordi ett eller annet om batch-wise og global metrics blabla
def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


if __name__ == '__main__':
    main()
