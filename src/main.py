from functions import *
from locomotion import *
from raycasts import *
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
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
    def __init__(self, count_bins_agents, count_rays_walls, radius_fov_walls, radius_fov_agents, max_view_range, count_fishes, cluster_path):
        self._count_bins = count_bins_agents
        self._count_rays = count_rays_walls
        self._fov_walls = radius_fov_walls
        self._fov_agents = radius_fov_agents
        self._view = max_view_range
        self._count_agents = count_fishes
        self._clusters_path = cluster_path
        self._clusters_mov, self._clusters_pos, self._clusters_ori = readClusters(cluster_path)
        self._clusters_counts = len(self._clusters_mov), len(self._clusters_pos), len(self._clusters_ori)
        self._wall_lines = defineLines(getRedPoints(path = "data/final_redpoint_wall.jpg"))
        self._tracks = []

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

    def trainNetwork(self, locomotion_path, raycasts_path, subtrack_length, batch_size, epochs, saveForExplainable = False):
        """
        trains the Network on the given dataset, by dividing it into subtracks, each subtrack has a length of subtrack_length,
        except the last one for each fish (when there are not enough datapoints to get to subtrack_length)
        """
        X = []
        y = []
        #get last locomotion for input and next for output
        df = pd.read_csv(locomotion_path, sep = ";")
        loc_size = sum(list(self._clusters_counts))
        locomotion = df.to_numpy()

        df = pd.read_csv(raycasts_path, sep = ";")
        raycasts = df.to_numpy()
        for i in range(0, self._count_agents):
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
            self._last_train_X.append(X[i][0, :, :].reshape(1, 1, X[i].shape[-1]))

        for i in range(0, len(X)):
            print("Training on Subtrack " + str(i))
            if saveForExplainable:
                self._tracks.append(X[i])
            self._model.fit(X[i], y[i], epochs=epochs, batch_size=batch_size, verbose=1)

    def testNetwork(self, timesteps = 10, save_tracks = None, start = "random", seed = None):
        """
        creates a Simulation for the network and created locomotions for timesteps times
        the starting position of each fish is random (if start = "random")
        their last locomotion is the first movement they had in the subtrack training session (such that fish 0 has first locomotion of subtrack 0, fish 1 has subtrack 1 ..)
        """
        random.seed(a = seed)
        first_pos = None
        tracks_header = np.array([list(chain.from_iterable(("Fish_" + str(i) + "_linear_movement", "Fish_" + str(i) + "_angle_new_pos", "Fish_" + str(i) + "_angle_change_orientation") for i in range(0, self._count_agents)))])
        tracks = np.empty((timesteps, tracks_header.shape[1]))
        cur_X = [[] for i in range(0, self._count_agents)]
        cur_pos = []
        locomotion = [[] for i in range(0, self._count_agents)]
        raycast_object = Raycast(self._wall_lines, self._count_bins, self._count_rays, self._fov_agents, self._fov_walls, self._view, self._count_agents)
        if start == "random":
            for i in range(0, self._count_agents):
                #(250 700) for x (125, 550) for y are good boundaries for coordiantes within the tank
                x_center, y_center, length, angle_rad = random.uniform(250, 700), random.uniform(125, 550), random.uniform(10,30), math.radians(random.uniform(0, 359))
                #cur_pos right now is x_head, y_head, length angle from look_vector to pos_x_axis
                cur_pos.append([x_center, y_center, length, angle_rad])
                #take locomotion of first self._count_agents subtracks first locomotion
                locomotion[i] = self._last_train_X[i][:, :, 0:sum(list(self._clusters_counts))].reshape(1, sum(list(self._clusters_counts)))
        elif start == "last_train":
            # cur_pos = #todo (no priority though)
            # cur_X = self._last_train_X
            print("not supported yet")
            return

        first_pos = [[cur_pos[i][j] for j in range(0, len(cur_pos[i]))] for i in range(0, len(cur_pos))]

        for i in range(0, timesteps):
            if i!=0 and i%1000 == 0:
                print("||| Timestep " + str(i) + " finished. |||")
            new_row = None
            
            #get Raycasts
            input_raycasts = [None for j in range(0, self._count_agents)]
            for j in range(0, self._count_agents):
                input_raycasts[j] = [cur_pos[j][0] + cur_pos[j][2]*math.cos(cur_pos[j][3]), cur_pos[j][1] + cur_pos[j][2]*math.sin(cur_pos[j][3]), cur_pos[j][0], cur_pos[j][1]]
            input_raycasts = np.array(input_raycasts).reshape(1, self._count_agents*4)

            raycasts = raycast_object.getRays(input_raycasts)
            for j in range(0, self._count_agents):
                cur_X[j] = np.append(np.append(locomotion[j], raycasts[:, j*self._count_rays : (j+1)*self._count_rays], axis = 1), raycasts[:, self._count_agents*self._count_rays+j*self._count_bins : self._count_agents*self._count_rays+(j+1)*self._count_bins])
                cur_X[j] = cur_X[j].reshape(1, 1, cur_X[j].shape[-1])

            #get next step
            for j in range(0, self._count_agents):
                pred = self._model.predict(cur_X[j])
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

                #chage movement to be always positive (in locomotion)
                angle_pos = (cur_pos[j][3] + pred_pos) - 2*math.pi if (cur_pos[j][3] + pred_pos) > 2*math.pi else (cur_pos[j][3] + pred_pos)
                movement_vector = (abs(pred_mov)*math.cos(angle_pos), abs(pred_mov)*math.sin(angle_pos))

                angle_ori = (cur_pos[j][3] + pred_ori) - 2*math.pi if (cur_pos[j][3] + pred_ori) > 2*math.pi else (cur_pos[j][3] + pred_ori)

                #add cur_pos vector to movement_vector
                cur_pos[j][0] += movement_vector[0]
                cur_pos[j][1] += movement_vector[1]
                cur_pos[j][3] = angle_ori

                #save old locomotion for next iterations network input
                locomotion[j] = np.append(np.append(pred_mov_bins, pred_pos_bins, axis = 1), pred_ori_bins, axis = 1)

                #save locomotion for output
                if j == 0:
                    new_row = np.array([[pred_mov, pred_pos, pred_ori]])
                else:
                    new_row = np.append(new_row, np.array([[pred_mov, pred_pos, pred_ori]]), axis = 1)
            
            tracks[i] = new_row

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
            intersect = get_intersect((center_x, center_y), (center_x+1, center_y), (self._wall_lines[i][0], self._wall_lines[i][1]), (self._wall_lines[i][2], self._wall_lines[i][3]))

            #check if intersection is between the x of that wall line (it was actually between these 2 points that mark that wall line) and it was to the right of our center point
            if intersect[0] >= min(self._wall_lines[i][0], self._wall_lines[i][2]) and intersect[0] <= max(self._wall_lines[i][0], self._wall_lines[i][2]) and intersect[0] >= center_x:
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
        centerPoint = 225, 225
        look_vector = cur_pos[0] + cur_pos[2]*math.cos(cur_pos[3]) - cur_pos[0], cur_pos[1] + cur_pos[2]*math.sin(cur_pos[3]) - cur_pos[1]
        vectorToCenter = centerPoint[0] - cur_pos[0], centerPoint[1] - cur_pos[1]

        mov = 20
        pos = getAngle(look_vector, vectorToCenter, mode = "radians")
        ori = pos

        #convert it to bin representation
        loco = np.array([mov, pos, ori])
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

    model = Sequential()
    model.add(LSTM(256, input_shape=(1, COUNT_BINS_AGENTS+COUNT_RAYS_WALLS+sum(list(CLUSTER_COUNTS))), dropout = 0.1))
    model.add(Dense(128))
    model.add(Dense(sum(list(CLUSTER_COUNTS))))
    model.compile(loss='mean_squared_error', optimizer='adam')

    sim = Simulation(COUNT_BINS_AGENTS, COUNT_RAYS_WALLS, RADIUS_FIELD_OF_VIEW_WALLS, RADIUS_FIELD_OF_VIEW_AGENTS, MAX_VIEW_RANGE, COUNT_FISHES, "data/clusters.txt")
    sim.setModel(model)
    sim.trainNetwork("data/locomotion_data_bin_diff1.csv", "data/raycast_data_diff1.csv", 6000, 10, 10)
    sim.trainNetwork("data/locomotion_data_bin_diff2.csv", "data/raycast_data_diff2.csv", 6000, 10, 10)
    sim.trainNetwork("data/locomotion_data_bin_diff3.csv", "data/raycast_data_diff3.csv", 6000, 10, 10)
    sim.trainNetwork("data/locomotion_data_bin_diff4.csv", "data/raycast_data_diff4.csv", 6000, 10, 10)
    model = sim.getModel()



    model.save("models/model_diff_1_to_4/")
    sim.testNetwork(timesteps = 18000, save_tracks = "data/")

if __name__ == "__main__":
    main()