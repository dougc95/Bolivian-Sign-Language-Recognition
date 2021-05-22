import numpy as np
import pandas as pd
import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, ConvLSTM2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn import preprocessing
from collections import deque

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "LTC-USD"

# Model params
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

main_df = pd.DataFrame()
ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


def preproccess_df(df):
    df = df.drop("future", 1)
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()  # Normalize
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)  # Scale
    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    for i in df.values:  # get only values
        prev_days.append([n for n in i[:-1]])  # remove target values
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
    random.shuffle(sequential_data)
    buys = []
    sells = []
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    random.shuffle(buys)
    random.shuffle(sells)
    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    x = []
    y = []

    for seq, target in sequential_data:
        x.append(seq)
        y.append(target)

    return np.array(x), np.array(y)


for ratio in ratios:
    dataset = f"crypto_data/{ratio}.csv"

    df = pd.read_csv("crypto_data/LTC-USD.csv", names=["time", "low", "high", "open", "close", "volume"])
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

    df.set_index("time", inplace=True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df["future"] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)

# print(main_df[[f"{RATIO_TO_PREDICT}_close", "future"]].head())
# print(main_df.head())
# for c in main_df.columns:
#     print(c)

main_df["target"] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))
# print(main_df[[f"{RATIO_TO_PREDICT}_close", "future", "target"]].head())

times = sorted(main_df.index.values)
last_5percent = times[-int(0.05 * len(times))]

validation_main_df = main_df[(main_df.index >= last_5percent)]
main_df = main_df[(main_df.index < last_5percent)]

train_x, train_y = preproccess_df(main_df)
validation_x, validation_y = preproccess_df(validation_main_df)

# print(f"train data: {len(train_x)} validation: {len(validation_x)}")
# print(f"Dont buys: {train_y.count(0)} buys: {train_y.count(1)}")
# print(f"VALIDATION Dont buys: {validation_y.count(0)} buys: {validation_y.count(1)}")

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

tensorboard = TensorBoard(log_dir=f"logs\\{NAME}")
filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}"  # file that will have the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                                      mode='max'))  # saves only the best ones

history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    #validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint])
