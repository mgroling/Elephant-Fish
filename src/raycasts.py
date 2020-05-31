import numpy as np
import pandas as pd
import imageio
import math
from functions import getRedPoints, defineLines, getDistance, get_intersect, getAngle

class Raycast:
    def __init__(self, wall_lines, count_bins_agents, count_rays_walls, radius_field_of_view_agents, radius_field_of_view_walls, max_view_range, count_fishes):
        """
        Initialize Raycast Object.
        Wall lines is expected to be a list of lists which contain lines, defined by 4 coordinates (first_point_x, first_point_y, second_point_x, second_point_y).
        Radius variables should be given in (0,360) and other variables should be positive integers.
        Angles relative to a fish are seen that from the direction the fish is looking at, 30° to the right are 30° relative to the fish and 30° to the left are 330° relative to the fish.
        """
        self._wall_lines = wall_lines
        self._agent_rays = count_bins_agents
        self._wall_rays = count_rays_walls
        self._radius_walls = radius_field_of_view_walls
        self._radius_agents = radius_field_of_view_agents
        self._max_view_range = max_view_range

        #Define bins for agent rays
        self._bins_header = np.array([[("fish_" + str(j) + "_bin_" + str((360-(radius_field_of_view_agents/2) + i*(radius_field_of_view_agents/count_bins_agents))%360) + "_" + str((360-(radius_field_of_view_agents/2) + (i+1)*(radius_field_of_view_agents/count_bins_agents))%360)) for i in range(0, count_bins_agents)] for j in range(0, count_fishes)]).flatten()
        self._bins = [(360-(radius_field_of_view_agents/2) + i*(radius_field_of_view_agents/count_bins_agents)) for i in range(0, count_bins_agents+1)]

        #Define wall rays
        #might not be optimal (for 360 radius, 180 is double and for close to 360 the two rays on the back dont have the same angle between them as others)
        self._wall_rays_header = np.array([["fish_" + str(j) + "_wall_ray_" + str((360-radius_field_of_view_walls/2 + i*(radius_field_of_view_walls/(count_rays_walls-1)))%360) for i in range(0, count_rays_walls)] for j in range(0, count_fishes)]).flatten()
        self._wall = [(360-radius_field_of_view_walls/2 + i*(radius_field_of_view_walls/(count_rays_walls-1)))%360 for i in range(0, count_rays_walls)]

    def getRays(self, np_array, path_to_save_to):
        """
        This function expects to be given a numpy array of the shape (rows, count_fishes*4) and saves a csv file at a given path (path has to end on .csv).
        The information about each given fish (or object in general) should be first_position_x, first_position_y, second_position_x, second_position_y.
        It is assumed that the fish is looking into the direction of first_positon_x - second_position_x for x and first_positon_y - second_position_y for y.
        """
        self._getFish(np_array)
        output_np_array = np.array([np.append(self._bins_header, self._wall_rays_header)])
        for i in range(0, len(np_array)):
            if i%1000 == 0:
                print("||| Frame " + str(i) + " finished. |||")
            new_row = [[] for k in range(0, len(self._bins_header))]
            distance_row = []
            for j in range(0, len(self._fishes)):
                #vector of the direction of the fish in question
                first_component, second_component = (self._fishes[j][i][0] - self._fishes[j][i][2], self._fishes[j][i][1] - self._fishes[j][i][3])
                look_vector = np.array([first_component, second_component])
                start_pos = np.array([self._fishes[j][i][0], self._fishes[j][i][1]])

                #get angles and distance for this timestep
                fishRays = self._getFishRays(j, i, look_vector)

                #put it into bins
                temp_fishAngles = fishRays[0]
                temp_fishAngles = [elem + 360 if elem < self._bins[0] else elem for elem in temp_fishAngles]
                bin_ids = np.digitize(temp_fishAngles, self._bins)
                for k in range(0, len(bin_ids)):
                    if bin_ids[k] != len(self._bins) and fishRays[1][k] < self._max_view_range:
                        new_row[bin_ids[k] + j*(len(self._bins)-1)-1].append(1 - fishRays[1][k]/self._max_view_range)

                #wall rays
                distance_row = distance_row + self._getWallRays(start_pos, look_vector)

            new_row = [[max(new_row[i], default = 0) for i in range(0, len(self._bins_header))] + distance_row]
            output_np_array = np.append(output_np_array, new_row, axis = 0)

        df = pd.DataFrame(data = output_np_array[1:], columns = output_np_array[0])
        df.to_csv(path_to_save_to, index = None, sep = ";")

    def _getFishRays(self, fish_id, row, look_vector):
        return_list = [[],[]]
        for i in range(0, len(self._fishes)):
            if i != fish_id:
                #get a vector to each other fish, from the current fish in question
                vector_to_fish = np.array([self._fishes[i][row][2] - self._fishes[fish_id][row][2], self._fishes[i][row][2] - self._fishes[fish_id][row][2]])

                distance = getDistance(self._fishes[i][row][2], self._fishes[i][row][3], self._fishes[fish_id][row][2], self._fishes[fish_id][row][3])
                angle = getAngle(look_vector, vector_to_fish)

                return_list[0].append(angle)
                return_list[1].append(distance)
        return return_list

    def _getWallRays(self, start_pos, look_vector):

        pos_x_axis = np.array([1, 0])
        
        angle_pos_x_axis_look_vector = getAngle(look_vector, pos_x_axis)
        distances = [0 for i in range(0, len(self._wall))]

        for i in range(0, len(self._wall)):
            #first we create a vector that has a certain degree to our look vector
            angle_relative_to_look_vector = (angle_pos_x_axis_look_vector + (360 - self._wall[i])) % 360
            new_ray = np.array([math.cos(math.radians(angle_relative_to_look_vector)), math.sin(math.radians(angle_relative_to_look_vector))])

            #and now we check where it collides with which wall and compute the distance to that wall
            for j in range(0, len(self._wall_lines)):
                intersection = get_intersect((self._wall_lines[j][0], self._wall_lines[j][1]), (self._wall_lines[j][2], self._wall_lines[j][3]), (start_pos[0], start_pos[1]), (start_pos[0] + new_ray[0], start_pos[1] + new_ray[1]))

                #check if it is between the two points of the line and if it is in max_view_range
                if intersection[0] >= min(self._wall_lines[j][0], self._wall_lines[j][2]) and intersection[0] <= intersection[0] <= max(self._wall_lines[j][0], self._wall_lines[j][2]) and getDistance(start_pos[0], start_pos[1], intersection[0], intersection[1]) < self._max_view_range:
                    distances[i] = (1 - getDistance(start_pos[0], start_pos[1], intersection[0], intersection[1]) / self._max_view_range)
                    break

        return distances

    def _getFish(self, np_array):
        #Create a seperate numpy array for each fish
        self._fishes = []
        for i in range(0, int(np_array.shape[1]/4)):
            self._fishes.append(np_array[:, i*4:(i+1)*4].astype(float))