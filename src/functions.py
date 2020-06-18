import numpy as np
import pandas as pd
import math
import imageio

def getRedPoints(cluster_distance = 25, path = "I:/Code/SWP/Raycasts/data/redpoints_walls.jpg", red_min_value = 200):
    """
    Given a Path, this function will return a list of points in the form of tuples (x, y).
    The points are read from the picture in a way such that points that exceed the red_min_value will be taken and only one will be considered in the range of cluster_distance.
    """
    im = imageio.imread(path)
    point_cluster_center = []
    add_new = True
    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            add_new = True
            if im[i, j, 0] > red_min_value:
                for k in range(0, len(point_cluster_center)):
                    if getDistance(point_cluster_center[k][0], point_cluster_center[k][1], j, i) < cluster_distance:
                        add_new = False
                if add_new:
                    point_cluster_center.append((j,i))
    return point_cluster_center

def defineLines(points):
    """
    Given a list of points (in a circle-like strucuture), this function will return a list of lines in the form of tuples (x1, y1, x2, y2).
    Points given have to be sorted by x or y (ascending or descending) in order for this to work correctly, if given unsorted this might return wrong values.
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

    #For our last line to be computed correctly, we take the 2 points that were only used once for now and define a line between them.
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

def getAngle(vector1, vector2, mode = "degrees"):
    """
    Given 2 vectors, in the form of tuples (x1, y1) this will return an angle in degrees, if not specfified further.
    If mode is anything else than "degrees", it will return angle in radians
    30° on the right are actually 30° and 30° on the left are 330° (relative to vector1).
    """
    #Initialize an orthogonal vector, that points to the right of your first vector.
    orth_vector1 = (vector1[1], -vector1[0])

    #Calculate angle between vector1 and vector2 (however this will only yield angles between 0° and 180° (the shorter one))
    temp = np.dot(vector1, vector2)/np.linalg.norm(vector1)/np.linalg.norm(vector2)
    angle = np.degrees(np.arccos(np.clip(temp, -1, 1)))

    #Calculate angle between orth_vector1 and vector2 (so we can get a degree between 0° and 360°)
    temp_orth = np.dot(orth_vector1, vector2)/np.linalg.norm(orth_vector1)/np.linalg.norm(vector2)
    angle_orth = np.degrees(np.arccos(np.clip(temp_orth, -1, 1)))

    #It is on the left side of our vector
    if angle_orth > 90:
        angle = 360 - angle

    return angle if mode == "degrees" else math.radians(angle)


def get_distances(tracks):
    """
    Computes distances for [x1 y1 x2 y2 ...] and [x1_next y1_next x2_next y2_next...]
    input: values in the format of extract_coordinates()
    output: [[dis1 dis2 ....]]
    careful: output array is one row shorter as input!
    """
    n_rows, n_cols = tracks.shape
    assert n_cols % 2 == 0
    assert n_cols > 1
    # Get distances of all points between 2 frames
    tracks1 = tracks[0:-1,]
    tracks2 = tracks[1:,]
    mov = tracks1 - tracks2                                 # subract x_curr x_next
    mov = mov**2                                            # power
    dist = np.atleast_2d(np.sum(mov[:,[0,1]], axis = 1))    # add x and y to eachother
    for i in range(1,int(n_cols/2)):                        # do to the rest of the cols
        dist = np.vstack((dist, np.sum(mov[:,2*i:2*i + 2], axis = 1) ))
    dist = np.sqrt(dist.T)                                  # take square root to gain distances

    return dist


def get_indices(i):
    """
    returns right indices for fishpositions
    """
    return (2*i, 2*i + 1)

def readClusters(path):
    """
    reads saved clusters from the getClusters function and returns a list for each locomotion type, they are ordered like in the file
    """
    with open(path, "r") as f:
        content = f.readlines()
    
    content = [x.strip() for x in content]
    count_clusters = tuple(map(int, content[1][1:-1].split(', ')))

    mov_clusters_ = content[2:2+count_clusters[0]]
    pos_clusters_ = content[2+count_clusters[0]:2+count_clusters[0]+count_clusters[1]]
    ori_clusters_ = content[2+count_clusters[0]+count_clusters[1]:2+count_clusters[0]+count_clusters[1]+count_clusters[2]]

    mov_clusters = [float(elem) for elem in mov_clusters_]
    pos_clusters = [float(elem) for elem in pos_clusters_]
    ori_clusters = [float(elem) for elem in ori_clusters_]

    return mov_clusters, pos_clusters, ori_clusters

def distancesToClusters(points, clusters):
    """
    computes distances from all points to all clusters
    not sure if this works for non 1d data
    """
    distances = None
    for j in range(0, len(clusters)):
        temp = np.abs(points - float(clusters[j])).reshape(-1, 1)
        if j == 0:
            distances = temp
        else: 
            distances = np.append(distances, temp, axis = 1)
    
    return distances

def softmax(np_array):
    """
    Compute softmax values row-wise (probabilites)
    """
    temp = np.exp(np_array - np.max(np_array, axis = 1).reshape(-1, 1))
    return np.divide(temp, np.sum(temp, axis = 1).reshape(-1, 1))

def selectPercentage(array):
    """
    Given an array of percentages that add up to 1, this selects one of the numbers with the certain percentage given
    returns the index of the selected percentage
    """
    value = 0
    rand = np.random.rand()
    for i in range(0, len(array)):
        value += array[i]
        if value > rand:
            return i