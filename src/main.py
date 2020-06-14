from raycasts import *
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
import reader
import tensorflow as tf
import pandas as pd
import numpy as np
import os

# Use python 3.6.10

def main():
    #Set Variables
    COUNT_BINS_AGENTS = 21
    COUNT_RAYS_WALLS = 15
    RADIUS_FIELD_OF_VIEW_WALLS = 180
    RADIUS_FIELD_OF_VIEW_AGENTS = 300
    MAX_VIEW_RANGE = 600
    COUNT_FISHES = 3
    N_CLUSTERS = 20

    #get raycast data (input)
    df_ray = pd.read_csv("data/raycast_data.csv", sep = ";")
    arr_ray = df_ray.to_numpy()

    #get locomotion data (input/output)
    df_loc = pd.read_csv("data/locomotion_data.csv", sep = ";")
    arr_loc = df_loc.to_numpy()

    #create dictionaries for locomotion clusters
    df_clusters = pd.read_csv("data/clusters.csv", sep = ";")
    dict_mov = {i: df_clusters["clusters_mov"] for i in range(0, len(df_clusters["clusters_mov"]))}
    dict_pos = {i: df_clusters["clusters_pos"] for i in range(0, len(df_clusters["clusters_pos"]))}
    dict_ori = {i: df_clusters["clusters_ori"] for i in range(0, len(df_clusters["clusters_ori"]))}

    #convert locomotion data into bin representation
    for i in range(0, arr_loc.shape[0]):
        for j in range(0, int(arr_loc.shape[1]/3)):
            #todo
            pass

    #create input/output data to feed to the network
    X = None
    y = None
    for i in range(0, COUNT_FISHES):
        if i == 0:
            X = np.append(np.append(arr_loc[:-1, i*3:(i+1)*3], arr_ray[1:-1, i*COUNT_BINS_AGENTS:(i+1)*COUNT_BINS_AGENTS], axis = 1), arr_ray[1:-1, COUNT_FISHES*COUNT_BINS_AGENTS+i*COUNT_RAYS_WALLS:COUNT_FISHES*COUNT_BINS_AGENTS+(i+1)*COUNT_RAYS_WALLS], axis = 1)
            y = arr_loc[1:, i*3:(i+1)*3]
        else:
            X = np.append(X, np.append(np.append(arr_loc[:-1, i*3:(i+1)*3], arr_ray[1:-1, i*COUNT_BINS_AGENTS:(i+1)*COUNT_BINS_AGENTS], axis = 1), arr_ray[1:-1, COUNT_FISHES*COUNT_BINS_AGENTS+i*COUNT_RAYS_WALLS:COUNT_FISHES*COUNT_BINS_AGENTS+(i+1)*COUNT_RAYS_WALLS], axis = 1), axis = 0)
            y = np.append(y, arr_loc[1:, i*3:(i+1)*3], axis = 0)


    #split data into train and test
    X_train = X[:int(len(X)*0.8)]
    y_train = y[:int(len(X)*0.8)]

    X_test = X[int(len(X)*0.8):]
    y_test = y[int(len(X)*0.8):]

    #reshape it for network input
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    #create model and fit it on training data
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, X_train.shape[2]), dropout = 0.1, return_sequences = True))
    model.add(LSTM(32, input_shape=(1, X_train.shape[2]), dropout = 0.1))
    model.add(Dense(16))
    model.add(Dense(2))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=100, batch_size=128, verbose=2)

    #predict test data and evaluate results
    predict_test = model.predict(X_test)
    eval_df = pd.DataFrame(data = predict_test, columns = ["pred_linear_movement", "pred_angle_radians"])
    eval_df.insert(0, "linear_movement", y_test[:, 0], False)
    eval_df.insert(0, "angle_radians", y_test[:, 1], False)
    eval_df.insert(0, "VERROR_movement", abs(eval_df["linear_movement"] - eval_df["pred_linear_movement"]), False)
    eval_df.insert(0, "temp1", abs(eval_df["angle_radians"] - (eval_df["pred_angle_radians"] + math.pi*2)), False)
    eval_df.insert(0, "temp2", abs(eval_df["angle_radians"] - eval_df["pred_angle_radians"]), False)
    eval_df.insert(0, "VERROR_angle", eval_df[["temp1", "temp2"]].min(axis = 1), False)
    eval_df.drop(["temp1", "temp2"], axis = 1)
    eval_df.insert(0, "ERROR_movement", eval_df["VERROR_movement"]/eval_df["linear_movement"], False)
    eval_df.insert(0, "ERROR_angle", eval_df["VERROR_angle"]/eval_df["angle_radians"], False)

    

    print("Mean VError of linear movement: " + str(eval_df["VERROR_movement"].mean()))
    print("Median VError of linear movement: " + str(eval_df["VERROR_movement"].median()))
    print("Mean VError of angle (radians): " + str(eval_df["VERROR_angle"].mean()))
    print("Median VError of angle (radians): " + str(eval_df["VERROR_angle"].median()))
    print("Mean Error of linear movement: " + str(eval_df["ERROR_movement"].mean()))
    print("Median Error of linear movement: " + str(eval_df["ERROR_movement"].median()))
    print("Mean Error of angle (radians): " + str(eval_df["ERROR_angle"].mean()))
    print("Median Error of angle (radians): " + str(eval_df["ERROR_angle"].median()))

    # Wir brauchen noch eine Funktion die zur runtime raycasts bestimmt später für die analyse

if __name__ == "__main__":
    main()