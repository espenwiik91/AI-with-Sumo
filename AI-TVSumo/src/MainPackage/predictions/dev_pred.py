import pandas as pd
from keras import layers
from keras import models
import numpy as np
import matplotlib.pyplot as plt

# This program classifies watching device with genreCode, playing time and start time
# as input. It performs significantly better with binary cross entropy as loss functions,
# compared to other loss functions. So far, the best model architecture has three hidden layers,
# and there is no big difference between sigmoid and softmax as a last layer actication
# function

# Reads csv file and shuffles rows
df = pd.read_csv("../../../Data/SumoData.csv", sep=";")
df_shuffled = df.sample(frac=1).reset_index(drop=True)

df = df[['playingTime_sec', 'genreCode', 'deviceCategory', 'startTime']]
df_genre = df['genreCode']
df_playing = df['playingTime_sec']
df_dev = df['deviceCategory']
start_temp = df['startTime']

temp = []
for t in start_temp:
    i = t.index(":")
    hour = t[(i - 2):i]
    hour = float(hour)
    temp.append(hour)

df_start = pd.DataFrame({'startTime': temp})

# One-hot encoding of genre code and device category
dev_dummies = pd.get_dummies(df_dev)
genre_dummies = pd.get_dummies(df_genre)

# Devides playing time into train and test, and transforms them into numpy arrays
train_data_playing = df_playing[:200000]
test_data_playing = df_playing[200000:]
train_data_start = df_start[:200000]
test_data_start = df_start[200000:]
# train_playing_np, test_playing_np = train_data_playing.values, test_data_playing.values

# Converts data types from ints to floats, so that mean and std operations can be done
train_data_playing = train_data_playing.astype(dtype=float)
test_data_playing = test_data_playing.astype(dtype=float)

# Normalizing data
mean = train_data_playing.mean(axis=0)
train_data_playing -= mean
std = train_data_playing.std(axis=0)
train_data_playing /= std

test_data_playing -= mean
test_data_playing /= std

mean = train_data_start.mean(axis=0)
train_data_start -= mean
std = train_data_start.std(axis=0)
train_data_start /= std

test_data_start -= mean
test_data_start /= std

# Creating train and test labels from the genre dummies
train_labels = dev_dummies[:200000]
test_labels = dev_dummies[200000:]

genre_train = genre_dummies[:200000]
genre_test = genre_dummies[200000:]

# Merges device and playing time data into train and test data
train_data = pd.concat([genre_train, train_data_playing, train_data_start], axis=1)
test_data = pd.concat([genre_test, test_data_playing, test_data_start], axis=1)

# print(len(train_data.columns))

x_train = np.asarray(train_data)
x_test = np.asarray(test_data)

y_train = np.asarray(train_labels)
y_test = np.asarray(test_labels)

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(51,)))
# model.add(layers.Dropout(0.2, input_shape=(8,)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Setting aside an evaluation set
x_val = x_train[:40000]
partial_x_train = x_train[40000:]

y_val = y_train[:40000]
partial_y_train = y_train[40000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=4,
                    batch_size=16,
                    validation_data=(x_val, y_val))

results = model.evaluate(x_test, y_test)
print(results)

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

acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
