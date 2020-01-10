import pandas as pd
from keras import layers
from keras import models
import numpy as np
import matplotlib.pyplot as plt
import datetime

# This program classifies watch sessions into different genreCodes based on
# playing time (measured in seconds), device category and start time.
# Start time values have been modified to hours, instead of date, hour and minute.
# GenreCode and device category have been one-hot encoded. Playing time and start
# time have been normalized around 0. After a lot of playing around, most parameters
# were insignificantly effecting the model when being tuned, except when setting the
# loss function to binary_crossentropy. Most parameter tuning predicts genreCode with
# an accuracy of 35% on the training set and slightly less on the test set.
# Binary-crossentropy has an accuracy around 98% both on the training and test set.

# Reads csv file and shuffles rows
df = pd.read_csv("../../../Data/SumoData.csv", sep=";")
df_shuffled = df.sample(frac=1).reset_index(drop=True)

df = df_shuffled[['playingTime_sec', 'genreCode', 'deviceCategory', 'startTime']]
df_genre = df_shuffled['genreCode']
df_playing = df_shuffled['playingTime_sec']
df_dev = df_shuffled['deviceCategory']
start_temp = df_shuffled['startTime']

# Modifies startTime values to floats with values between 0 and 23

week_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

day_temp_list = []
for t in start_temp:
    day = int(t[0:2])
    month = int(t[3:5])
    year = 2000 + int(t[6:8])
    day_of_week = datetime.date(year, month, day)
    day_of_week = day_of_week.weekday()
    day_of_week = week_days[day_of_week]
    day_temp_list.append(day_of_week)

df_day_of_week = pd.DataFrame({'day': day_temp_list})

hour_temp_list = []
for t in start_temp:
    i = t.index(":")
    hour = t[(i - 2):i]
    hour = float(hour)
    hour_temp_list.append(hour)

df_start_hour = pd.DataFrame({'startTime': hour_temp_list})

# One-hot encoding of genre code and device category
dev_dummies = pd.get_dummies(df_dev)
genre_dummies = pd.get_dummies(df_genre)
day_dummies = pd.get_dummies(df_day_of_week)

# Divides playing time and start time into train and test, and transforms them into numpy arrays
train_data_playing = df_playing[:200000]
test_data_playing = df_playing[200000:]
train_data_start = df_start_hour[:200000]
test_data_start = df_start_hour[200000:]


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
train_labels = genre_dummies[:200000]
test_labels = genre_dummies[200000:]

dev_train = dev_dummies[:200000]
dev_test = dev_dummies[200000:]

day_train = day_dummies[:200000]
day_test = day_dummies[200000:]

# Merges device and playing time data into train and test data
train_data = pd.concat([dev_train, train_data_playing, train_data_start, day_train], axis=1)

test_data = pd.concat([dev_test, test_data_playing, test_data_start, day_test], axis=1)

print("train_data_columns", train_data.shape, "test_data_columns", test_data.shape)

x_train = np.asarray(train_data)
x_test = np.asarray(test_data)

y_train = np.asarray(train_labels)
y_test = np.asarray(test_labels)

model = models.Sequential()
model.add(layers.Dense(216, activation='relu', input_shape=(9,)))
# model.add(layers.Dropout(0.2, input_shape=(8,)))
# model.add(layers.Dense(216, activation='relu'))
# model.add(layers.Dense(216, activation='relu'))
model.add(layers.Dense(216, activation='relu'))
model.add(layers.Dense(49, activation='sigmoid'))

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
                    epochs=2,
                    batch_size=16,
                    validation_data=(x_val, y_val))

results = model.evaluate(x_test, y_test)
print(results)

loss = history.history['loss']
val_loss = history.history['val_loss']

print(history.history.keys())

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
