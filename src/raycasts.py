import numpy as np
import pandas as pd
import imageio
import math

class Raycast:
    def __init__(self, wall_lines, count_bins_agents, count_rays_walls, radius_field_of_view, max_view_range, count_fishes):
        self._wall_lines = wall_lines
        self._agent_rays = count_bins_agents
        self._wall_rays = count_rays_walls
        self._radius = radius_field_of_view
        self._max_view_range = max_view_range

        self._bins_header = np.array([[("fish_" + str(j) + "_bin_" + str((360-(radius_field_of_view/2) + i*(radius_field_of_view/count_bins_agents))%360) + "_" + str((360-(radius_field_of_view/2) + (i+1)*(radius_field_of_view/count_bins_agents))%360)) for i in range(0, count_bins_agents)] for j in range(0, count_fishes)]).flatten()
        self._bins = [(360-(radius_field_of_view/2) + i*(radius_field_of_view/count_bins_agents)) for i in range(0, count_bins_agents+1)]

        #might not be optimal (for 360 radius, 180 is double and for close to 360 the two rays on the back dont have the same angle between them as others)
        self._wall_rays_header = np.array([["fish_" + str(j) + "_wall_ray_" + str((360-radius_field_of_view/2 + i*(radius_field_of_view/(count_rays_walls-1)))%360) for i in range(0, count_rays_walls)] for j in range(0, count_fishes)]).flatten()
        self._wall = [(360-radius_field_of_view/2 + i*(radius_field_of_view/(count_rays_walls-1)))%360 for i in range(0, count_rays_walls)]

    def getRays(self, df, path_to_save_to, agents, cols_per_agent):
        self._getFish(df, agents, cols_per_agent)
        output_np_array = np.array([np.append(self._bins_header, self._wall_rays_header)])
        print(output_np_array)
        for i in range(0, 1):
            new_row = [[] for k in range(0, len(self._bins_header))]
            distance_row = []
            for j in range(0, len(self._fishes)):
                first_component, second_component = (self._fishes[j][i][0] - self._fishes[j][i][3], self._fishes[j][i][1] - self._fishes[j][i][4])
                look_vector = np.array([first_component, second_component])
                start_pos = np.array([self._fishes[j][i][0], self._fishes[j][i][1]])

                #Initialize an orthogonal vector, that points to the right of your first vector.
                orth_look_vector = np.array([second_component, -first_component])

                #get angles and distance for this timestep
                fishRays = self._getFishRays(j, i, look_vector, orth_look_vector)

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

    def _getFishRays(self, fish_id, row, look_vector, orth_look_vector):
        return_list = [[],[]]
        for i in range(0, len(self._fishes)):
            if i != fish_id:
                vector_to_fish = np.array([self._fishes[i][row][3] - self._fishes[fish_id][row][3], self._fishes[i][row][4] - self._fishes[fish_id][row][4]])

                temp = np.dot(look_vector, vector_to_fish)/np.linalg.norm(look_vector)/np.linalg.norm(vector_to_fish)
                angle = np.degrees(np.arccos(np.clip(temp, -1, 1)))

                temp_orth = np.dot(orth_look_vector, vector_to_fish)/np.linalg.norm(orth_look_vector)/np.linalg.norm(vector_to_fish)
                angle_orth = np.degrees(np.arccos(np.clip(temp_orth, -1, 1)))

                distance = getDistance(self._fishes[i][row][3], self._fishes[i][row][4], self._fishes[fish_id][row][3], self._fishes[fish_id][row][4])

                #We do this orthogonal look vector because if we do not, then we only get values between 0 and 180 and so any angle on the left will be treated the same as any angle on the right.
                if angle_orth > 90:
                    angle = 360 - angle

                return_list[0].append(angle)
                return_list[1].append(distance)
        return return_list

    def _getWallRays(self, start_pos, look_vector):

        pos_x_axis = np.array([1, 0])
        pos_y_axis = np.array([0, 1])

        temp = np.dot(pos_x_axis, look_vector)/np.linalg.norm(look_vector)/np.linalg.norm(pos_x_axis)
        angle_pos_x_axis_look_vector = np.degrees(np.arccos(np.clip(temp, -1, 1)))

        temp = np.dot(pos_y_axis, look_vector)/np.linalg.norm(look_vector)/np.linalg.norm(pos_y_axis)
        angle_pos_y_axis_look_vector = np.degrees(np.arccos(np.clip(temp, -1, 1)))

        if angle_pos_y_axis_look_vector > 90:
            angle_pos_x_axis_look_vector = 360 - angle_pos_x_axis_look_vector

        distances = [0 for i in range(0, len(self._wall))]

        for i in range(0, len(self._wall)):
            #first we create a vector that has a certain degree to our look vector
            angle_relative_to_look_vector = (angle_pos_x_axis_look_vector + (360 - self._wall[i])) % 360

            new_ray = np.array([math.cos(math.radians(angle_relative_to_look_vector)), math.sin(math.radians(angle_relative_to_look_vector))])

            #and now we check where it collides with which wall and compute the distance to that wall
            for j in range(0, len(self._wall_lines)):
                intersection = get_intersect((self._wall_lines[j][0], self._wall_lines[j][1]), (self._wall_lines[j][2], self._wall_lines[j][3]), (start_pos[0], start_pos[1]), (start_pos[0] + new_ray[0], start_pos[1] + new_ray[1]))

                #check if it is between the two points of the line
                if intersection[0] >= min(self._wall_lines[j][0], self._wall_lines[j][2]) and intersection[0] <= intersection[0] <= max(self._wall_lines[j][0], self._wall_lines[j][2]):
                    distances[i] = (1 - getDistance(start_pos[0], start_pos[1], intersection[0], intersection[1]) / self._max_view_range)
                    break

        return distances



    def _getFish(self, np_array, agents, cols_per_agent):
        #Create a seperate numpy array for each fish
        self._fishes = []
        for i in range(0, agents):
            self._fishes.append(np_array[:, i*cols_per_agent:(i+1)*cols_per_agent].astype(float))

def getRedPoints(cluster_distance = 50, path = "I:/Code/SWP/Raycasts/data/redpoints_walls.jpg", red_min_value = 200):
    """
    Given a Path, this function will return a list of points in the form of tuples (x, y).
    The points are read from the picture in a way such that points that they must exceed the red_min_value and only one will be considered in the range of cluster_distance.
    """
    im = imageio.imread(path)
    point_cluster_center = []
    add_new = True
    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            add_new = True
            if im[i, j, 0] > red_min_value:
                for k in range(0, len(point_cluster_center)):
                    if getDistance(point_cluster_center[k][0], point_cluster_center[k][1], i, j) < cluster_distance:
                        add_new = False
                if add_new:
                    point_cluster_center.append((i,j))
    return point_cluster_center

def defineLines(points):
    """
    Given a list of points, this function will return a list of lines in the form of tuples (x1, y1, x2, y2).
    Points given have to be in a circle-like structure.
    """
    lines_list = []
    #First of we choose a point to look at, then we search for the nearest nearest other point (from our pot) to that one and remove the chosen point from our pot.
    while len(points) > 1:
        current_point = points[0]
        points.pop(0)
        cur_min_index = 0
        cur_min_dist = getDistance(current_point[0], current_point[1], points[0][0], points[0][1])
        for  i in range(1, len(points)):
            temp = getDistance(current_point[0], current_point[1], points[i][0], points[i][1])
            if temp < cur_min_dist:
                cur_min_dist = temp
                cur_min_index = i
        lines_list.append((current_point[0], current_point[1], points[cur_min_index][0], points[cur_min_index][1]))

    #For our last line to be computed correctly, we take the 2 points that were only used one for now and define a line between them.
    temp = []
    lines_list_single_points = [(elem[0], elem[1]) for elem in lines_list] + [(elem[2], elem[3]) for elem in lines_list]
    count_points = {x:lines_list_single_points.count(x) for x in lines_list_single_points}
    for elem in count_points.items():
        if elem[1] == 1:
            temp.append(elem[0])
    lines_list.append((temp[0][0], temp[0][1], temp[1][0], temp[1][1]))
    return lines_list

def getDistance(x1, y1, x2, y2):
    """
    Computes distance between 2 given points.
    """
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def get_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    Code taken from: https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)

#main

#wall lines were not using the same coordinates as deeplabcut, so we had to invert one (580 is length, 582 is height)
our_wall_lines = [(580-elem[0], 582-elem[1], 580-elem[2], 582-elem[3]) for elem in defineLines(getRedPoints())]

ray = Raycast(our_wall_lines, 6, 3, 120, 1000, 3)

np_array = pd.read_csv("I:/Code/SWP/Raycasts/data/3fishDLC_resnet152_track_fishesMay7shuffle1_100000.csv").to_numpy()[2:, 1:]

ray.getRays(np_array, "I:/Code/SWP/Raycasts/data/savee.csv", 3, 12)