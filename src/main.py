from functions import *
from locomotion import *
from raycasts import *
from analysis import *
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Dropout
from itertools import chain
import reader
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import random
import shap

# Use python 3.6.10

class Simulation:
    def __init__(self, count_bins_agents, count_rays_walls, radius_fov_walls, radius_fov_agents, max_view_range, count_fishes, cluster_path, verbose = 1):
        self._count_bins = count_bins_agents
        self._count_rays = count_rays_walls
        self._fov_walls = radius_fov_walls
        self._fov_agents = radius_fov_agents
        self._view = max_view_range
        self._count_agents = count_fishes
        if cluster_path == None:
            self._cluster = False
        else:
            self._cluster = True
            self._clusters_path = cluster_path
            self._clusters_mov, self._clusters_pos, self._clusters_ori = readClusters(cluster_path)
            self._clusters_counts = len(self._clusters_mov), len(self._clusters_pos), len(self._clusters_ori)
        self._wall_lines = defineLines(getRedPoints(path = "data/final_redpoint_wall.jpg"))
        self._tracks = []
        self.verbose = verbose

    def setModel(self, model):
        self._model = model

    def getModel(self):
        return self._model

    def explainParameters(self):
        #TODO
        # https://github.com/slundberg/shap/blob/master/notebooks/deep_explainer/Keras%20LSTM%20for%20IMDB%20Sentiment%20Classification.ipynb
        # https://stackoverflow.com/questions/45361559/feature-importance-chart-in-neural-network-using-keras-in-python/61861991#61861991
        explainer = shap.DeepExplainer(self._model, self._tracks[0])
        shap_values = explainer.shap_values(self._tracks[0])

        shap.force_plot(explainer.expected_value, shap_values[0,:], self._tracks[0][0,:], matplotlib = True)

        shap.summary_plot(shap_values, self._tracks[0], plot_type = "bar")

    def trainNetworkOnce(self, locomotion_paths, raycast_paths, batch_size, sequence_length, epochs):
        x_train_data_all, y_train_data_all = None, None
        x_val_data_all, y_val_data_all = None, None

        for i in range(0, len(locomotion_paths)):
            #get locomotion
            df = pd.read_csv(locomotion_paths[i], sep = ";")
            locomotion = df.to_numpy()

            #get raycasts (we dont need the first raycast cause it does not have a last locomotion)
            df = pd.read_csv(raycast_paths[i], sep = ";")
            raycasts = df.to_numpy()[1:]

            loc_size = 3

            for j in range(0, len(self._count_agents)):
                loc = locomotion[:, j*loc_size : (j+1)*loc_size]
                wall_rays = raycasts[:, j*self._count_bins : (j+1)*self._count_bins]
                agent_rays = raycasts[:, self._count_agents*self._count_bins+j*self._count_rays : self._count_agents*self._count_bins+(j+1)*self._count_rays]

                trajectory = np.append(np.append(loc, wall_rays, axis = 1), agent_rays, axis = 1)

                TRAIN_SPLIT = int(0.8*len(trajectory))

                #standardize datset
                subtrack_mean = trajectory[:TRAIN_SPLIT].mean(axis = 0)
                subtrack_std = trajectory[:TRAIN_SPLIT].std(axis = 0)
                trajectory = (trajectory - subtrack_mean) / subtrack_std

                #create sequences
                x_train, y_train = multivariate_data(trajectory, trajectory[:, 0:3], 0, TRAIN_SPLIT, sequence_length, 1, 1, single_step = True)
                x_val, y_val = multivariate_data(trajectory, trajectory[:, 0:3], TRAIN_SPLIT, None, sequence_length, 1, 1, single_step = True)

                if i == 0 and j == 0:
                    x_train_data_all, y_train_data_all = x_train, y_train
                    x_val_data_all, y_val_data_all = x_val, y_val
                else:
                    print("should be (n, seq_length, 39):", x_train.shape)
                    x_train_data_all = np.append(x_train_data_all, x_train, axis = 0), np.append(y_train_data_all, y_train, axis = 0)
                    x_val_data_all = np.append(x_val_data_all, x_val, axis = 0), np.append(y_val_data_all, y_val, axis = 0)

        train_data = tf.data.Dataset.from_tensor_slices((x_train_data_all, y_train_data_all))
        train_data = train_data.cache().shuffle(10000).batch(batch_size).repeat()

        val_data = tf.data.Dataset.from_tensor_slices((x_val_data_all, y_val_data_all))
        val_data = val_data.batch(batch_size).repeat()

        if self.verbose >= 2:
            history = self._model.fit(train_data, epochs = epochs, steps_per_epoch = len(train_data) // batch_size, validation_data = val_data, validation_steps = 50, verbose = 1)
            plot_train_history(history, "Training and validation loss")
        else:
            history = self._model.fit(train_data, epochs = epochs, steps_per_epoch = len(train_data) // batch_size, validation_data = val_data, validation_steps = 50, verbose = 0)
            plot_train_history(history, "Training and validation loss")
            

    def trainNetwork(self, locomotion_path, raycasts_path, subtrack_length, batch_size, sequence_length, epochs, saveForExplainable = False):
        """
        trains the Network on the given dataset, by dividing it into subtracks, each subtrack has a length of subtrack_length,
        except the last one for each fish (when there are not enough datapoints to get to subtrack_length)
        """
        X = []
        y = []
        self._start_simulation = []
        #get last locomotion for input and next for output
        df = pd.read_csv(locomotion_path, sep = ";")
        if self._cluster:
            loc_size = sum(list(self._clusters_counts))
        else:
            loc_size = 3
        locomotion = df.to_numpy()

        df = pd.read_csv(raycasts_path, sep = ";")
        raycasts = df.to_numpy()

        if self.verbose >= 1:
            print("Started training on " + locomotion_path[-9:-4])

        #create subtracks
        for i in range(0, self._count_agents):
            for subtrack in range(0, int(len(locomotion)/subtrack_length)+1):
                #get locomotion
                X.append(locomotion[subtrack*subtrack_length : min((subtrack+1)*subtrack_length-1, len(locomotion)-2), i*loc_size : (i+1)*loc_size])
                y.append(locomotion[subtrack*subtrack_length+1 : min((subtrack+1)*subtrack_length, len(locomotion)-1), i*loc_size : (i+1)*loc_size])

                #get raycasts
                wall_rays = raycasts[subtrack*subtrack_length+1 : min((subtrack+1)*subtrack_length, len(raycasts)-2), i*self._count_bins : (i+1)*self._count_bins]
                agent_rays = raycasts[subtrack*subtrack_length+1 : min((subtrack+1)*subtrack_length, len(raycasts)-2), self._count_agents*self._count_bins+i*self._count_rays : self._count_agents*self._count_bins+(i+1)*self._count_rays]
                X[-1] = np.append(np.append(X[-1], wall_rays, axis = 1), agent_rays, axis = 1)

        for i in range(0, len(X)):
            TRAIN_SPLIT = int(0.8*len(X[i]))

            #standardize datset
            subtrack_mean = X[i][:TRAIN_SPLIT].mean(axis = 0)
            subtrack_std = X[i][:TRAIN_SPLIT].std(axis = 0)
            X[i] = (X[i] - subtrack_mean) / subtrack_std

            #create sequences
            x_train, y_train = multivariate_data(X[i], X[i][:, 0:3], 0, TRAIN_SPLIT, sequence_length, 1, 1, single_step = True)
            x_val, y_val = multivariate_data(X[i], X[i][:, 0:3], TRAIN_SPLIT, None, sequence_length, 1, 1, single_step = True)
            
            self._start_simulation.append(x_train[0:1])

            train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            train_data = train_data.cache().shuffle(10000).batch(batch_size).repeat()

            val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            val_data = val_data.batch(batch_size).repeat()

            if self.verbose >= 2:
                print("Training on subtrack", str(i))
                self._model.fit(train_data, epochs = epochs, steps_per_epoch = len(x_train) // batch_size, validation_data = val_data, validation_steps = 50, verbose = 1)
            else:
                self._model.fit(train_data, epochs = epochs, steps_per_epoch = len(x_train) // batch_size, validation_data = val_data, validation_steps = 50, verbose = 0)

    def testNetwork(self, timesteps = 10, save_tracks = None, start = "random", seed = None):
        """
        creates a Simulation for the network and created locomotions for timesteps times
        the starting position of each fish is random (if start = "random")
        """
        out_of_tank = 0
        random.seed(a = seed)
        first_pos = None
        tracks_header = np.array([list(chain.from_iterable(("Fish_" + str(i) + "_linear_movement", "Fish_" + str(i) + "_angle_new_pos", "Fish_" + str(i) + "_angle_change_orientation") for i in range(0, self._count_agents)))])
        tracks = np.empty((timesteps, tracks_header.shape[1]))
        cur_X = [[] for i in range(0, self._count_agents)]
        cur_pos = []
        locomotion = [[] for i in range(0, self._count_agents)]
        raycast_object = Raycast(self._wall_lines, self._count_bins, self._count_rays, self._fov_agents, self._fov_walls, self._view, self._count_agents)
        if start == "random":
            start_indices = random.sample(range(len(self._start_simulation)), 3)
            for i in range(0, self._count_agents):
                #(250 700) for x (125, 550) for y are good boundaries for coordiantes within the tank
                x_center, y_center, length, angle_rad = random.uniform(250, 700), random.uniform(125, 550), random.uniform(10,30), math.radians(random.uniform(0, 359))
                #cur_pos right now is x_head, y_head, length angle from look_vector to pos_x_axis
                cur_pos.append([x_center, y_center, length, angle_rad])
                cur_X[i] = self._start_simulation[start_indices[i]]

        first_pos = [[cur_pos[i][j] for j in range(0, len(cur_pos[i]))] for i in range(0, len(cur_pos))]

        for i in range(0, timesteps):
            if i!=0 and i%1000 == 0 and self.verbose >= 1:
                print("||| Timestep " + str(i) + " finished. |||")
            new_row = None

            #get next step
            for j in range(0, self._count_agents):
                pred = self._model.predict(cur_X[j])
                pred_mov, pred_pos, pred_ori = None, None, None

                if self._cluster:
                    #collect prediction for each bin and create percentages out of them
                    pred_mov_bins = softmax(pred[:, : self._clusters_counts[0]])
                    pred_pos_bins = softmax(pred[:, self._clusters_counts[0] : self._clusters_counts[0]+self._clusters_counts[1]])
                    pred_ori_bins = softmax(pred[:, self._clusters_counts[0]+self._clusters_counts[1] :])

                    #select one randomly (with its given percentage)
                    pred_mov_bins_index = selectPercentage(pred_mov_bins[0], seed = seed)
                    pred_pos_bins_index = selectPercentage(pred_pos_bins[0], seed = seed)
                    pred_ori_bins_index = selectPercentage(pred_ori_bins[0], seed = seed)

                    #translate index of bin into actual locomotion values
                    pred_mov = self._clusters_mov[pred_mov_bins_index]
                    pred_pos = self._clusters_pos[pred_pos_bins_index]
                    pred_ori = self._clusters_ori[pred_ori_bins_index]
                else:
                    pred_mov, pred_pos, pred_ori = float(pred[:, 0]), float(pred[:, 1]), float(pred[:, 2])
                

                #chage movement to be always positive (in locomotion)
                angle_pos = (cur_pos[j][3] + pred_pos) - 2*math.pi if (cur_pos[j][3] + pred_pos) > 2*math.pi else (cur_pos[j][3] + pred_pos)
                movement_vector = (abs(pred_mov)*math.cos(angle_pos), abs(pred_mov)*math.sin(angle_pos))
                angle_ori = (cur_pos[j][3] + pred_ori) - 2*math.pi if (cur_pos[j][3] + pred_ori) > 2*math.pi else (cur_pos[j][3] + pred_ori)

                temp_x = cur_pos[j][0]
                temp_y = cur_pos[j][1]
                temp_ori = cur_pos[j][3]

                #add cur_pos vector to movement_vector
                cur_pos[j][0] += movement_vector[0]
                cur_pos[j][1] += movement_vector[1]
                cur_pos[j][3] = angle_ori

                #if fish would be outside of the tank after moving, make him move towards the center instead
                if not self.isFishInsideTank(cur_pos[j][0], cur_pos[j][1]):
                    out_of_tank += 1
                    pred_mov, pred_pos, pred_ori, loco_bin = self.moveToCenter(cur_pos[j])

                    angle_pos = (temp_ori + pred_pos) - 2*math.pi if (temp_ori + pred_pos) > 2*math.pi else (temp_ori + pred_pos)
                    movement_vector = (abs(pred_mov)*math.cos(angle_pos), abs(pred_mov)*math.sin(angle_pos))
                    angle_ori = (temp_ori + pred_ori) - 2*math.pi if (temp_ori + pred_ori) > 2*math.pi else (temp_ori + pred_ori)

                    cur_pos[j][0] = temp_x + movement_vector[0]
                    cur_pos[j][1] = temp_y + movement_vector[1]
                    cur_pos[j][3] = angle_ori

                    if self._cluster:
                        locomotion[j] = loco_bin
                    else:
                        locomotion[j] = np.array([[pred_mov, pred_pos, pred_ori]])
                else:
                    #save old locomotion for next iterations network input
                    if self._cluster:
                        locomotion[j] = np.append(np.append(pred_mov_bins, pred_pos_bins, axis = 1), pred_ori_bins, axis = 1)
                    else:
                        locomotion[j] = np.array([[pred_mov, pred_pos, pred_ori]])

                #save locomotion for output
                if j == 0:
                    new_row = np.array([[pred_mov, pred_pos, pred_ori]])
                else:
                    new_row = np.append(new_row, np.array([[pred_mov, pred_pos, pred_ori]]), axis = 1)

            tracks[i] = new_row

            #get Raycasts
            input_raycasts = [None for j in range(0, self._count_agents)]
            for j in range(0, self._count_agents):
                input_raycasts[j] = [cur_pos[j][0] + cur_pos[j][2]*math.cos(cur_pos[j][3]), cur_pos[j][1] + cur_pos[j][2]*math.sin(cur_pos[j][3]), cur_pos[j][0], cur_pos[j][1]]
            input_raycasts = np.array(input_raycasts).reshape(1, self._count_agents*4)

            temp_X = [None for j in range(0, self._count_agents)]
            raycasts = raycast_object.getRays(input_raycasts)
            for j in range(0, self._count_agents):
                temp_X[j] = np.append(np.append(locomotion[j], raycasts[:, j*self._count_rays : (j+1)*self._count_rays], axis = 1), raycasts[:, self._count_agents*self._count_rays+j*self._count_bins : self._count_agents*self._count_rays+(j+1)*self._count_bins])
                temp_X[j] = temp_X[j].reshape(1, 1, temp_X[j].shape[0])
                temp_X[j] = temp_X[j].astype(np.float)
                #remove oldest observation
                cur_X[j] = np.delete(cur_X[j], 0, axis = 1)
                #append latest observation
                cur_X[j] = np.append(cur_X[j], temp_X[j], axis = 1)

        if self.verbose >= 1:
            print("fish tried to move " + str(out_of_tank) + " times out tank")

        df = pd.DataFrame(data = tracks, columns = tracks_header[0])

        if save_tracks != None:
            df.to_csv(save_tracks + "locomotion_simulation.csv", sep = ";")
            with open(save_tracks + "startposition_simulation.txt", "w+") as f:
                for elem in first_pos:
                    f.write("%s\n" % elem)
        else:
            return df, first_pos

    def isFishInsideTank(self, center_x, center_y):
        """
        checks if fish center point is outside of the fish tank
        using this method: https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/#:~:text=1)%20Draw%20a%20horizontal%20line,true%2C%20then%20point%20lies%20outside.
        """
        collissions = 0
        for i in range(0, len(self._wall_lines)):
            #get intersection of horizontal line from x to the right and get the intersection with each wall
            intersect = get_intersect((center_x, center_y), (center_x+2000, center_y), (self._wall_lines[i][0], self._wall_lines[i][1]), (self._wall_lines[i][2], self._wall_lines[i][3]))

            #check if intersection is between the x of that wall line (it was actually between these 2 points that mark that wall line) and it was to the right of our center point
            if intersect[0] >= min(self._wall_lines[i][0], self._wall_lines[i][2]) and intersect[0] <= max(self._wall_lines[i][0], self._wall_lines[i][2]) and intersect[1] >= min(self._wall_lines[i][1], self._wall_lines[i][3]) and intersect[1] >= min(self._wall_lines[i][1], self._wall_lines[i][3]) and intersect[0] >= center_x:
                collissions += 1

        if collissions % 2 == 1:
            return True
        else:
            return False

    def moveToCenter(self, cur_pos):
        """
        gives locomotion for moving to center, given a current position of a fish (center_x, center_y, length, orientation in radians)
        for use if a fish decides to go into the real world
        """
        centerPoint = 500, 350
        look_vector = cur_pos[0] + cur_pos[2]*math.cos(cur_pos[3]) - cur_pos[0], cur_pos[1] + cur_pos[2]*math.sin(cur_pos[3]) - cur_pos[1]
        vectorToCenter = centerPoint[0] - cur_pos[0], centerPoint[1] - cur_pos[1]

        mov = 3
        pos = getAngle(look_vector, vectorToCenter, mode = "radians")
        ori = cur_pos[3]

        #convert it to bin representation
        loco = np.array([[mov, pos, ori]])
        loco_bin = convertLocmotionToBin(loco, "data/clusters.txt")

        return mov, pos, ori, loco_bin

def main():
    #importance of variables for analysis later: https://stackoverflow.com/questions/45361559/feature-importance-chart-in-neural-network-using-keras-in-python/61861991#61861991
    #Set Variables
    COUNT_BINS_AGENTS = 21
    COUNT_RAYS_WALLS = 15
    RADIUS_FIELD_OF_VIEW_WALLS = 180
    RADIUS_FIELD_OF_VIEW_AGENTS = 300
    MAX_VIEW_RANGE = 709
    COUNT_FISHES = 3
    CLUSTER_COUNTS = (18, 17, 26)

    SEQUENCE_LENGTH = 70
    BATCH_SIZE = 20
    SUBTRACK_LENGTH = 6100
    EPOCHS = 1

    locomotion_paths = ["data/locomotion_data_same1.csv", "data/locomotion_data_same3.csv", "data/locomotion_data_same4.csv", "data/locomotion_data_same5.csv"]
    raycast_paths = ["data/raycast_data_same1.csv", "data/raycast_data_same3.csv", "data/raycast_data_same4.csv", "data/raycast_data_same5.csv"]

    #40
    #20

    model = Sequential()
    model.add(LSTM(40, input_shape = (SEQUENCE_LENGTH, COUNT_BINS_AGENTS+COUNT_RAYS_WALLS+3)))
    model.add(Dropout(0.3))
    model.add(Dense(20))
    model.add(Dropout(0.3))
    model.add(Dense(3))
    model.compile(optimizer = RMSprop(), loss = "mse")

    # model = load_model("models/model_LSTM64_DROPOUT02_DENSE3_70_20_6100_1")

    sim = Simulation(COUNT_BINS_AGENTS, COUNT_RAYS_WALLS, RADIUS_FIELD_OF_VIEW_WALLS, RADIUS_FIELD_OF_VIEW_AGENTS, MAX_VIEW_RANGE, COUNT_FISHES, None, verbose = 2)
    sim.setModel(model)
    sim.trainNetworkOnce(locomotion_paths[0:2], raycast_paths[0:2], BATCH_SIZE, SEQUENCE_LENGTH, EPOCHS)
    # sim.trainNetwork("data/locomotion_data_same1.csv", "data/raycast_data_same1.csv", SUBTRACK_LENGTH, BATCH_SIZE, SEQUENCE_LENGTH, EPOCHS)
    # sim.trainNetwork("data/locomotion_data_same3.csv", "data/raycast_data_same3.csv", SUBTRACK_LENGTH, BATCH_SIZE, SEQUENCE_LENGTH, EPOCHS)
    # sim.trainNetwork("data/locomotion_data_same4.csv", "data/raycast_data_same4.csv", SUBTRACK_LENGTH, BATCH_SIZE, SEQUENCE_LENGTH, EPOCHS)
    # sim.trainNetwork("data/locomotion_data_same5.csv", "data/raycast_data_same5.csv", SUBTRACK_LENGTH, BATCH_SIZE, SEQUENCE_LENGTH, EPOCHS)
    model = sim.getModel()

    model.save("models/model_LSTM_ALL")

    sim.testNetwork(timesteps = 2000, save_tracks = "data/")

if __name__ == "__main__":
    main()