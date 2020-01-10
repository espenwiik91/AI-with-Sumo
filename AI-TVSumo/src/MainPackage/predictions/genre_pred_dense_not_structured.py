import pandas as pd
import calendar
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras import models, Sequential
from keras import layers
from keras.optimizers import SGD
from keras import backend as K
from sklearn.model_selection import StratifiedKFold


def start(number, shuffle, df, epochs, batchsize):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    df = df[
        df.userId == number]

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

    hour_data_frame = hour_extractor()
    # hour_data_frame = pd.get_dummies(hour_data_frame['start_hour'])

    print("df:genres", df_genres.shape)

    day_data_frame = weekday_extractor()
    print("day_data_frame", day_data_frame.shape)

    print("hour_data_frame", hour_data_frame.shape)
    # print("df_ids", df_ids.shape)
    print("df_dev", df_devices.shape)
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
    hour_data_frame.reset_index(drop=True, inplace=True)
    normalized_day_data_frame.reset_index(drop=True, inplace=True)
    df_devices.reset_index(drop=True, inplace=True)
    # df_ids.reset_index(drop=True, inplace=True)

    # Concatinates DataFrames into the a final DataFrame that will be the input data to the neural network
    # x = pd.concat([normalized_day_data_frame, hour_data_frame, df_devices, df_ids], axis=1)
    # x = pd.concat([normalized_day_data_frame, normalized_hour_data_frame, df_devices, df_ids], axis=1)
    x = pd.concat([normalized_day_data_frame, normalized_hour_data_frame, df_devices], axis=1)
    # x = pd.concat([normalized_day_data_frame, normalized_hour_data_frame], axis=1)

    # Splits input and output data into train and test sets
    x_train, x_test = train_test_split(x, test_size=0.2, random_state=12, shuffle=False)
    y_train, y_test = train_test_split(df_genres, test_size=0.2, random_state=12, shuffle=False)

    x_input_size = x_train.shape[1]
    y_input_size = y_train.shape[1]

    # Transforms DataFrames into numpy arrays for model processing
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    # Instansiates the model and adds layers to it
    def build_model(x_size, y_size):
        model = models.Sequential()
        model.add(layers.Dense(8, activation='relu', input_shape=(x_size,)))
        # model.add(layers.Dropout(0.2))
        # model.add(layers.Dense(8, activation='relu'))
        # model.add(layers.Dropout(0.2))
        # model.add(layers.Dense(128, activation='relu'))
        # model.add(layers.Dense(128, activation='relu'))
        # model.add(layers.Dense(128, activation='relu'))
        # model.add(layers.Dense(4, activation='relu'))
        model.add(layers.Dense(y_size, activation='softmax'))

        # Compiling the model
        opt = SGD(lr=0.01)
        model.compile(optimizer="rmsprop",
                      loss='categorical_crossentropy',
                      metrics=[precision, recall, 'acc'])
        return model

    model = build_model(x_input_size, y_input_size)

    def k_fold_val(model, epochs, batchsize):
        k = 4
        num_validation_samples = len(x_train) // k

        loss_scores = []
        precision_scores = []
        recall_scores = []
        acc_scores = []
        all_loss_histories = []

        for i in range(k):
            print('processing fold #', i)
            validation_data = x_train[i * num_validation_samples:
                                      num_validation_samples * (i + 1)]
            validation_targets = y_train[i * num_validation_samples: (i + 1) * num_validation_samples]
            partial_train_data = np.concatenate(
                [x_train[:i * num_validation_samples],
                 x_train[(i + 1) * num_validation_samples:]])
            partial_train_targets = np.concatenate(
                [y_train[:i * num_validation_samples],
                 y_train[(i + 1) * num_validation_samples:]],
                axis=0)
            model.fit(partial_train_data, partial_train_targets,
                      epochs=epochs, batch_size=batchsize, verbose=0)
            val_loss, val_precision, val_recall, val_acc = model.evaluate(validation_data, validation_targets,
                                                                          verbose=0)
            loss_scores.append(val_loss)
            precision_scores.append(val_precision)
            recall_scores.append(val_recall)
            acc_scores.append(val_acc)
            loss_history = history.history['val_loss']
            all_loss_histories.append(loss_history)

        # print(loss_scores, precision_scores, recall_scores, acc_scores)
        loss_scores = np.mean(loss_scores)
        precision_scores = np.mean(precision_scores)
        recall_scores = np.mean(recall_scores)
        acc_scores = np.mean(acc_scores)

        #print("Mean loss: ", loss_scores, ". Mean precision: ", precision_scores, ". Mean Recall: ", recall_scores,
         #     ". Mean accuracy: ", acc_scores)

        avarage_precision_history = [np.mean([x[i] for x in all_loss_histories]) for i in range(epochs)]
        plt.plot(range(1, len(avarage_precision_history) + 1), avarage_precision_history)
        plt.xlabel('Epochs')
        plt.ylabel('Validation loss')
        plt.show()

        # results = model.evaluate(x_test, y_test)
        # print(results)

        # Xnew = scalar.transform(Xnew)
        # make a prediction
        ynew = model.predict_classes(x_test)
        # show the inputs and predicted outputs
        n_right = 0
        n_wrong = 0
        for i in range(len(x_test)):
            # print("prediction:%s, label=%s" % (ynew[i], y_test[i]))

            if y_test[i][ynew[i]] == 1:
                n_right += 1
            else:
                n_wrong += 1
        # print("There were: ", n_right, "right predictions and ", n_wrong, " wrong predictions on test set")
        predict_acc = n_right / (n_right + n_wrong)
        # print("Test set gives ", predict_acc, "% corrrect precision")
        return precision_scores, recall_scores, acc_scores, predict_acc


    # Divides x and y train into train and evaluation sets
    partial_x_train, x_val = train_test_split(x_train, test_size=0.2)
    partial_y_train, y_val = train_test_split(y_train, test_size=0.2)

    history = model.fit(partial_x_train, partial_y_train, epochs=epochs, batch_size=batchsize,
                        validation_data=(x_val, y_val))
    """
    historyDict = history.history

    lossValues = historyDict['loss']
    valLossValues = historyDict['val_loss']
    pre = historyDict['precision']
    valPre = historyDict['val_precision']
    rec = historyDict['recall']
    valRec = historyDict['val_recall']
    acc = historyDict['acc']
    valAcc = historyDict['val_acc']
    epochsplot = range(1, len(pre) + 1)

    plt.subplot(4, 1, 1)
    plt.plot(epochsplot, lossValues, 'bo', label='Training')
    plt.plot(epochsplot, valLossValues, 'b', label='Validation')
    plt.title("Training and Validation Metrics")
    #     plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(epochsplot, pre, 'bo', label='Training')
    plt.plot(epochsplot, valPre, 'b', label='Validation')
    plt.ylabel('Precision')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(epochsplot, rec, 'bo', label='Training')
    plt.plot(epochsplot, valRec, 'b', label='Validation')
    plt.ylabel('Recall')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(epochsplot, acc, 'bo', label='Training')
    plt.plot(epochsplot, valAcc, 'b', label='Validation')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.xlabel('Epochs')
    plt.show()
    """
    results = model.evaluate(x_test, y_test)
    print(results)

    res_prec, res_rec, res_acc, res_test = k_fold_val(model, epochs, batchsize)
    res_dict = {"kfold_precision": res_prec, "kfold_recall: ": res_rec, "kfold_acc : ": res_acc,
                "kfold_test_precision : ": res_test}
    return res_dict


def main(startpoint, stoppoint, shuffle, epochs, batchsize):
    df = pd.read_csv("../../../Data/SumoData2.csv", sep=";")
    user_series = df['userId'].value_counts()
    # Removes rows from df where genreCode has value NaN
    df = df[pd.notnull(df['genreCode'])]

    # Removes noisy values from the device category column
    devices = ['Set Top Box', 'Desktop', 'TV', 'Tablet', 'Mobile Phone', 'Games Console']
    df = df[df['deviceCategory'].isin(devices)]

    main_genres = ['DR10', 'UH70', 'UH60', 'IN31', 'DR30', 'UH20', 'BA00', 'IN23']
    df = df[df['genreCode'].isin(main_genres)]
    user_list = []
    cnt = startpoint
    for vc in user_series.iteritems():
        print(vc)
        user_list.append(vc[0])
        if cnt == stoppoint:
            break
        cnt += 1
    res_list = []
    for e in user_list:
        s = start(str(e), shuffle, df, epochs, batchsize)
        res_list.append(s)
    mean_dict = {1: [], 2: [], 3: [], 4: []}
    for dic in res_list:
        res_string = ""
        dict_count = 1
        for k, v in dic.items():
            mean_dict[dict_count].append(v)
            res_string += str("{} {} ".format(k, v))
            dict_count += 1
        print(res_string)
    print("")
    for k, v in mean_dict.items():
        mean = sum(v) / len(v)
        v = mean
        mean_dict[k] = v

    output_list = ["Precision", "Recall", "Accuracy", "Test score"]
    output_counter = 0
    for v in mean_dict.values():
        out = output_list[output_counter]
        print("{}, {}".format(out, v))
        output_counter += 1


main(1, 10, True, 24, 32)
