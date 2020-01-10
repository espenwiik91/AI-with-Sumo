import pandas as pd
from keras import layers
from keras import models
import numpy as np
import re
import matplotlib.pyplot as plt

df = pd.read_csv("../../../Data/SumoData.csv", sep=";")
df_shuffled = df.sample(frac=1).reset_index(drop=True)

dur_temp = df_shuffled['duration']

dur_in_sec = []
for line in dur_temp:
    line /= 1000
    dur_in_sec.append(line)

duration = pd.DataFrame({'duration': dur_in_sec})

playingTime = df_shuffled['playingTime_sec']

df_playing_duration = pd.concat([playingTime, duration], axis=1)

normalized_playing_time = []
leif = []
for row in df_playing_duration.itertuples():
    if 0 < row.playingTime_sec and 0 < row.duration:
        # print('This checked out: ', row)
        if row.playingTime_sec > row.duration:
            res = 1
        else:
            res = row.playingTime_sec / row.duration
        # print(res)
        normalized_playing_time.append(res)
    else:
        normalized_playing_time.append(np.NaN)

df_labels = pd.DataFrame({'fraction_seen': normalized_playing_time})
df_genreCode = df_shuffled['genreCode']
df_fylke = df_shuffled['fylke']
df_dev = df_shuffled['deviceCategory']
genre_dummies = pd.get_dummies(df_genreCode)
fylke_dummies = pd.get_dummies(df_fylke)
device_dummies = pd.get_dummies(df_dev)
df_input = pd.concat([genre_dummies, fylke_dummies, device_dummies], axis=1)

toDrop = []
index = 0
for row in df_labels.itertuples():
    isNan = np.isnan(row.fraction_seen)
    if isNan:
        toDrop.append(index)
    index += 1

df_labels = df_labels.drop(toDrop, axis=0)

df_input = df_input.drop(toDrop, axis=0)

# divides the dataset into train_set and test_set on a fraction of 0.8
train_size = len(df) * 0.8
temp_size = str(train_size)
a = re.search(r'\b(.)\b', temp_size)
if (int(temp_size[a.start() + 1])) > 4:
    train_size = round(train_size)
train_size = int(train_size)

train_data = df_input[:train_size]
test_data = df_input[train_size:]

train_labels = df_labels[:train_size]
test_labels = df_labels[train_size:]

x_train = np.asarray(train_data)
x_test = np.asarray(test_data)

y_train = np.asarray(train_labels)
y_test = np.asarray(test_labels)

# Skjønner ikke helt hvorfor input-shape skal være 1 her
# Burde det ikke være likt som shapen til train-data, altså 21?
model = models.Sequential()
model.add(layers.Dense(64, activation='relu',
                       input_shape=(train_data.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# Setting aside an evaluation set
x_val = x_train[train_size:]
partial_x_train = x_train[:train_size]

y_val = y_train[train_size:]
partial_y_train = y_train[:train_size]

history = model.fit(train_data,
                    train_labels,
                    epochs=7,
                    batch_size=16,
                    validation_data=(x_val, y_val))

results = model.evaluate(test_data, test_labels)
print(results)

"""
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# summarize history for accuracy
plt.plot(history.history['val_mean_absolute_error'])
plt.plot(history.history['mean_absolute_error'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
"""
print(history.history.keys())

"""
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()

acc = history.history['mae']
val_acc = history.history['val_mae']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
"""