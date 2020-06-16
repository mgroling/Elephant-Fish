from functions import *
from raycasts import *
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
import reader
import tensorflow as tf
import pandas as pd
import numpy as np
import os

# Use python 3.6.10

class Simulation:
    def __init__(self, count_bins_agents, count_rays_walls, radius_fov_walls, radius_fov_agents, max_view_range, count_fishes, clusters_counts):
        self._count_bins = count_bins_agents
        self._count_rays = count_rays_walls
        self._fov_walls = radius_fov_walls
        self._fov_agents = radius_fov_agents
        self._view = max_view_range
        self._count_agents = count_fishes
        self._clusters_counts = clusters_counts

    def setModel(self, model):
        self._model = model

    def trainNetwork(self, locomotion_path, raycasts_path, subtrack_length, batch_size, epochs):
        X = []
        y = []
        for i in range(0, self._count_agents):
            #get last locomotion for input and next for output
            df = pd.read_csv(locomotion_path, sep = ";")
            loc_size = sum(list(self._clusters_counts))
            locomotion = df.to_numpy()

            df = pd.read_csv(raycasts_path, sep = ";")
            raycasts = df.to_numpy()
            for subtrack in range(0, int(len(locomotion)/subtrack_length)+1):
                #get locomotion
                X.append(locomotion[subtrack*subtrack_length : min((subtrack+1)*subtrack_length-1, len(locomotion)-2), i*loc_size : (i+1)*loc_size])
                y.append(locomotion[subtrack*subtrack_length+1 : min((subtrack+1)*subtrack_length, len(locomotion)-1), i*loc_size : (i+1)*loc_size])

                #get raycasts
                X[-1] = np.append(np.append(X[-1], raycasts[subtrack*subtrack_length+1 : min((subtrack+1)*subtrack_length, len(raycasts)-2), i*self._count_bins : (i+1)*self._count_bins], axis = 1), raycasts[subtrack*subtrack_length+1 : min((subtrack+1)*subtrack_length, len(raycasts)-2), self._count_agents*self._count_bins+i*self._count_rays : self._count_agents*self._count_bins+(i+1)*self._count_rays], axis = 1)
        
                #reshape for network input
                X[-1] = np.reshape(X[-1], (X[-1].shape[0], 1, X[-1].shape[1]))

        self._last_train_X = []
        for i in range(0, self._count_agents):
            self._last_train_X.append(X[i][0].reshape(1, 1, X[i].shape[-1]))

        for i in range(0, len(X)):
            print("Training on Subtrack " + str(i))
            self._model.fit(X[i], y[i], epochs=epochs, batch_size=batch_size, verbose=2)

    def testNetwork(self, timesteps = 10, save_tracks = None, start = "last_train"):
        cur_X = None
        cur_y = None
        if start == "random":
            #TODO
            pass
        elif start == "last_train":
            cur_X = self._last_train_X
    
        for i in range(0, timesteps):
            for j in range(0, self._count_agents):
                print(softmax(self._model.predict(cur_X[j])))
                #TODO




def main():
    #Set Variables
    COUNT_BINS_AGENTS = 21
    COUNT_RAYS_WALLS = 15
    RADIUS_FIELD_OF_VIEW_WALLS = 180
    RADIUS_FIELD_OF_VIEW_AGENTS = 300
    MAX_VIEW_RANGE = 600
    COUNT_FISHES = 3
    CLUSTER_COUNTS = (15, 20, 17)

    model = Sequential()
    model.add(LSTM(64, input_shape=(1, COUNT_BINS_AGENTS+COUNT_RAYS_WALLS+sum(list(CLUSTER_COUNTS))), dropout = 0.1))
    model.add(Dense(16))
    model.add(Dense(sum(list(CLUSTER_COUNTS))))
    model.compile(loss='mean_squared_error', optimizer='adam')

    sim = Simulation(COUNT_BINS_AGENTS, COUNT_RAYS_WALLS, RADIUS_FIELD_OF_VIEW_WALLS, RADIUS_FIELD_OF_VIEW_AGENTS, MAX_VIEW_RANGE, COUNT_FISHES, CLUSTER_COUNTS)
    sim.setModel(model)
    sim.trainNetwork("data/locomotion_data_bin.csv", "data/raycast_data.csv", 6000, 10, 1)
    sim.testNetwork()
    # #Set Variables
    # COUNT_BINS_AGENTS = 21
    # COUNT_RAYS_WALLS = 15
    # RADIUS_FIELD_OF_VIEW_WALLS = 180
    # RADIUS_FIELD_OF_VIEW_AGENTS = 300
    # MAX_VIEW_RANGE = 600
    # COUNT_FISHES = 3
    # N_CLUSTERS = 20

    # #get raycast data (input)
    # df_ray = pd.read_csv("data/raycast_data.csv", sep = ";")
    # arr_ray = df_ray.to_numpy()

    # #get locomotion data (input/output)
    # df_loc = pd.read_csv("data/locomotion_data.csv", sep = ";")
    # arr_loc = df_loc.to_numpy()

    # #create lists for locomotion clusters
    # clusters_mov, clusters_pos, clusters_ori = readClusters("data/clusters.txt")

    # #create input/output data to feed to the network
    # X = None
    # y = None
    # for i in range(0, COUNT_FISHES):
    #     if i == 0:
    #         X = np.append(np.append(arr_loc[:-1, i*3:(i+1)*3], arr_ray[1:-1, i*COUNT_BINS_AGENTS:(i+1)*COUNT_BINS_AGENTS], axis = 1), arr_ray[1:-1, COUNT_FISHES*COUNT_BINS_AGENTS+i*COUNT_RAYS_WALLS:COUNT_FISHES*COUNT_BINS_AGENTS+(i+1)*COUNT_RAYS_WALLS], axis = 1)
    #         y = arr_loc[1:, i*3:(i+1)*3]
    #     else:
    #         X = np.append(X, np.append(np.append(arr_loc[:-1, i*3:(i+1)*3], arr_ray[1:-1, i*COUNT_BINS_AGENTS:(i+1)*COUNT_BINS_AGENTS], axis = 1), arr_ray[1:-1, COUNT_FISHES*COUNT_BINS_AGENTS+i*COUNT_RAYS_WALLS:COUNT_FISHES*COUNT_BINS_AGENTS+(i+1)*COUNT_RAYS_WALLS], axis = 1), axis = 0)
    #         y = np.append(y, arr_loc[1:, i*3:(i+1)*3], axis = 0)


    # #split data into train and test
    # X_train = X[:int(len(X)*0.8)]
    # y_train = y[:int(len(X)*0.8)]

    # X_test = X[int(len(X)*0.8):]
    # y_test = y[int(len(X)*0.8):]

    # #reshape it for network input
    # X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    # X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # #create model and fit it on training data
    # model = Sequential()
    # model.add(LSTM(64, input_shape=(1, X_train.shape[2]), dropout = 0.1, return_sequences = True))
    # model.add(LSTM(32, input_shape=(1, X_train.shape[2]), dropout = 0.1))
    # model.add(Dense(16))
    # model.add(Dense(3))
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=2)
    # model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=2)

    # #predict test data and evaluate results
    # predict_test = model.predict(X_test)
    # eval_df = pd.DataFrame(data = predict_test, columns = ["pred_linear_movement", "pred_angle_radians"])
    # eval_df.insert(0, "linear_movement", y_test[:, 0], False)
    # eval_df.insert(0, "angle_radians", y_test[:, 1], False)
    # eval_df.insert(0, "VERROR_movement", abs(eval_df["linear_movement"] - eval_df["pred_linear_movement"]), False)
    # eval_df.insert(0, "temp1", abs(eval_df["angle_radians"] - (eval_df["pred_angle_radians"] + math.pi*2)), False)
    # eval_df.insert(0, "temp2", abs(eval_df["angle_radians"] - eval_df["pred_angle_radians"]), False)
    # eval_df.insert(0, "VERROR_angle", eval_df[["temp1", "temp2"]].min(axis = 1), False)
    # eval_df.drop(["temp1", "temp2"], axis = 1)
    # eval_df.insert(0, "ERROR_movement", eval_df["VERROR_movement"]/eval_df["linear_movement"], False)
    # eval_df.insert(0, "ERROR_angle", eval_df["VERROR_angle"]/eval_df["angle_radians"], False)

    

    # print("Mean VError of linear movement: " + str(eval_df["VERROR_movement"].mean()))
    # print("Median VError of linear movement: " + str(eval_df["VERROR_movement"].median()))
    # print("Mean VError of angle (radians): " + str(eval_df["VERROR_angle"].mean()))
    # print("Median VError of angle (radians): " + str(eval_df["VERROR_angle"].median()))
    # print("Mean Error of linear movement: " + str(eval_df["ERROR_movement"].mean()))
    # print("Median Error of linear movement: " + str(eval_df["ERROR_movement"].median()))
    # print("Mean Error of angle (radians): " + str(eval_df["ERROR_angle"].mean()))
    # print("Median Error of angle (radians): " + str(eval_df["ERROR_angle"].median()))

if __name__ == "__main__":
    main()