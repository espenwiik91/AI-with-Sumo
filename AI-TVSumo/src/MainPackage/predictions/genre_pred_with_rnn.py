import pandas as pd
import calendar
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras.optimizers import SGD
from keras.layers import LSTM, GRU
from keras.layers import Reshape

df = pd.read_csv("../../../Data/SumoData2.csv", sep=";")
# df = df.sample(frac=1).reset_index(drop=True)

print(df.shape)

# Removes rows from df where genreCode has value NaN
df = df[pd.notnull(df['genreCode'])]

# Removes noisy values from the device category column
devices = ['Set Top Box', 'Desktop', 'TV', 'Tablet', 'Mobile Phone', 'Games Console']
df = df[df['deviceCategory'].isin(devices)]

df = df[
    df.userId == "5e853b1701d4ba068e90e08ac6a215c7bc94b6e51b59b64007d11cc1fba43ae1:f358bc80a7b0491aa3a53b81c6765cc8"]
print(df['userId'].value_counts())

print(df.shape)


# Function that splits startTime values into year, day and month and passes these values into the
# weekday function from the calendar lib. It returns DataFrame object containing a column with weekdays
# represented as numbers from 0 (monday) to 6 (sunday)
def weekday_extractor():
    df_day_of_week = df['startTime']
    weekday_list = []
    for e in df_day_of_week:
        year = int(e[:4])
        month = int(e[5:7])
        day = int(e[8:10])
        weekday_list.append(calendar.weekday(year, month, day))
    df_day = pd.DataFrame({'weekday': weekday_list})
    return df_day


# Function that splits startTime values into hours, and appends values into a DataFrame according to
# four hour intervals. Returns the DataFrame
def hour_extractor():
    df_day_of_week = df['startTime']
    hour_list = []
    for e in df_day_of_week:
        hour = int(e[11:13])
        if 0 <= hour < 4:
            hour_list.append(0)
        elif 4 <= hour < 8:
            hour_list.append(1)
        elif 8 <= hour < 12:
            hour_list.append(2)
        elif 12 <= hour < 16:
            hour_list.append(3)
        elif 16 <= hour < 20:
            hour_list.append(4)
        elif 20 <= hour < 24:
            hour_list.append(5)
    df_start = pd.DataFrame({'start_hour': hour_list})
    return df_start


# One hot encoding of categorical values
# df_ids = pd.get_dummies(df['userId'])
df_genres = pd.get_dummies(df['genreCode'])
df_devices = pd.get_dummies(df['deviceCategory'])

print("df:genres", df_genres.shape)

day_data_frame = weekday_extractor()
# print("day_data_frame", day_data_frame.shape)
hour_data_frame = hour_extractor()
# print("hour_data_frame", hour_data_frame.shape)
# print("df_ids", df_ids.shape)
# print("df_dev", df_devices.shape)
# Normalizes startTime values into floats between 0 and 1
start_temp = hour_data_frame.values
min_max_scaler = preprocessing.MinMaxScaler()
start_temp_minmax = min_max_scaler.fit_transform(start_temp)
normalized_hour_data_frame = pd.DataFrame(data=start_temp_minmax, columns=['hour'])

# Normalizes weekday values into floats between 0 and 1
day_temp = day_data_frame.values
day_temp_minmax = min_max_scaler.fit_transform(day_temp)
normalized_day_data_frame = pd.DataFrame(data=day_temp_minmax, columns=['day'])

# Resets indexes. Was needed as DF concatination went wrong without it
normalized_hour_data_frame.reset_index(drop=True, inplace=True)
normalized_day_data_frame.reset_index(drop=True, inplace=True)
df_devices.reset_index(drop=True, inplace=True)
# df_ids.reset_index(drop=True, inplace=True)

# Concatinates DataFrames into the a final DataFrame that will be the input data to the neural network
x = pd.concat([normalized_day_data_frame, normalized_hour_data_frame, df_devices], axis=1)
# x = pd.concat([normalized_day_data_frame, normalized_hour_data_frame], axis=1)
print("mjejejejejeeee", x.shape)

# Splits input and output data into train and test sets
x_train, x_test = train_test_split(x, test_size=0.2)
y_train, y_test = train_test_split(df_genres, test_size=0.2)

# Transforms DataFrames into numpy arrays for model processing
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

print(y_train.shape)
print(x_train.shape)

# Instansiates the model and adds layers to it
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(6,)))
#model.add(Reshape((1, 4)))
model.add(layers.SimpleRNN(32, input_shape=))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(256, activation='relu'))
# model.add(LSTM(256, input_shape=(1357, 6)))
model.add(LSTM(256))
# model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(48, activation='softmax'))

# Compiling the model
model.compile(optimizer="rmsprop",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Divides x and y train into train and evaluation sets
partial_x_train, x_val = train_test_split(x_train, test_size=0.2)
partial_y_train, y_val = train_test_split(y_train, test_size=0.2)

history = model.fit(partial_x_train, partial_y_train, epochs=10, batch_size=216,
                    validation_data=(x_val, y_val))

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
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

results = model.evaluate(x_test, y_test)
print(results)
