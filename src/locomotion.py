import numpy as np
import pandas as pd
import math
from functions import getAngle, getDistance, readClusters, distancesToClusters, softmax, get_indices, convPolarToCart
from itertools import chain
from reader import *
from sklearn.cluster import KMeans

def getLocomotion(np_array, path_to_save_to = None, mode="radians"):
    """
    This function expects to be given a numpy array of the shape (rows, count_fishes*4) and saves a csv file at a given path (path has to end on .csv).
    The information about each given fish (or object in general) should be first_position_x, first_position_y, second_position_x, second_position_y.
    It is assumed that the fish is looking into the direction of first_positon_x - second_position_x for x and first_positon_y - second_position_y for y.
    """
    header = np.array([list(chain.from_iterable(("Fish_" + str(i) + "_linear_movement", "Fish_" + str(i) + "_angle_new_pos", "Fish_" + str(i) + "_angle_change_orientation") for i in range(0, int(np_array.shape[1]/4))))])
    output = np.empty((np_array.shape[0]-1, header.shape[1]))

    for i in range(0, np_array.shape[0]-1):
        if i!=0 and i%1000 == 0:
            print("||| Frame " + str(i) + " finished. |||")
        new_row = [0 for k in range(0, int(3*np_array.shape[1]/4))]
        for j in range(0, int(np_array.shape[1]/4)):
            head_x = np_array[i, j*4]
            head_y = np_array[i, j*4+1]
            center_x = np_array[i, j*4+2]
            center_y = np_array[i, j*4+3]

            head_x_next = np_array[i+1, j*4]
            head_y_next = np_array[i+1, j*4+1]
            center_x_next = np_array[i+1, j*4+2]
            center_y_next = np_array[i+1, j*4+3]

            #look vector
            look_vector = (head_x - center_x, head_y - center_y)
            #new look vector
            look_vector_next = (head_x_next - center_x_next, head_y_next - center_y_next)
            #vector to new position
            vector_next = (center_x_next - center_x, center_y_next - center_y)

            new_row[j*3+1] = getAngle(look_vector, vector_next, mode = mode)
            new_row[j*3+2] = getAngle(look_vector, look_vector_next, mode = mode)
            temp = getDistance(center_x, center_y, center_x_next, center_y_next)
            #its forward movement if it's new position is not at the back of the fish and otherwise it is backward movement
            new_row[j*3] = -temp if new_row[j*3+1] > math.pi/2 and new_row[j*3+1] < 3/2*math.pi else temp
        output[i] = new_row

    if path_to_save_to == None:
        return output
    else:
        df = pd.DataFrame(data = output, columns = header[0])
        df.to_csv(path_to_save_to, index = None, sep = ";")

def convertLocmotionToBin(loco, clusters_path, path_to_save = None, probabilities = True):
    #get cluster centers
    clusters_mov, clusters_pos, clusters_ori = readClusters(clusters_path)

    result = None
    #convert locomotion into bin representation for each fish
    for i in range(0, int(loco.shape[1]/3)):
        if probabilities:
            #compute distances to cluster centers and invert them (1/x)
            dist_mov = 1 / distancesToClusters(loco[:, i*3], clusters_mov)
            dist_pos = 1 / distancesToClusters(loco[:, i*3+1], clusters_pos)
            dist_ori = 1 / distancesToClusters(loco[:, i*3+2], clusters_ori)

            #get probabilites row-wise with softmax function and append header
            prob_mov = np.append(np.array([["Fish_" + str(i) + "_prob_mov_bin_" + str(j) for j in range(0, len(clusters_mov))]]), softmax(dist_mov), axis = 0)
            prob_pos = np.append(np.array([["Fish_" + str(i) + "_prob_pos_bin_" + str(j) for j in range(0, len(clusters_pos))]]), softmax(dist_pos), axis = 0)
            prob_ori = np.append(np.array([["Fish_" + str(i) + "_prob_ori_bin_" + str(j) for j in range(0, len(clusters_ori))]]), softmax(dist_ori), axis = 0)

            temp = np.append(np.append(prob_mov, prob_pos, axis = 1), prob_ori, axis = 1)
            if i == 0:
                result = temp
            else:
                result = np.append(result, temp , axis = 1)
        else:
            #todo
            pass

    if path_to_save == None:
        return result[1:]
    else:
        df = pd.DataFrame(data = result[1:], columns = result[0])
        df.to_csv(path_to_save, sep = ";")


def row_l2c( coords, locs ):
    """
    Returns 1d ndarray with new coordinates based on previos coordinades and given locomotions,
    Output: [center1_x, center1_y, orientation1, ...]
    """
    nfish = len(coords) // 3

    # coords indices
    xs = [3 * x for x in range(nfish)]
    ys = [3 * x + 1 for x in range(nfish)]
    os = [3 * x + 2 for x in range(nfish)]
    # locs indices
    lin = [3 * x for x in range(nfish)]
    ang = [3 * x + 1 for x in range(nfish)]
    ori = [3 * x + 2 for x in range(nfish)]
    # computation
    new_angles = ( coords[os] + locs[ang] ) % ( np.pi * 2 )
    xvals = np.cos( new_angles ) * np.abs( locs[lin] )
    yvals = np.sin( new_angles ) * np.abs( locs[lin] )
    out = np.empty( coords.shape )
    out[xs] = coords[xs] + xvals
    out[ys] = coords[ys] + yvals
    out[os] = ( coords[os] + locs[ori] ) % ( np.pi * 2 )
    return out


def convLocToCart( loc, startpoints ):
    """
    Converts locomotion np array to coordinates,
    assumens first fish "looks" upwards
    loc:
        2d array, per row 3 entries per fish, [linear movement, angulare movement, turn movement]:
        [
            [fish1_lin, fish1_ang, fish1_trn, fish2_lin, fish2_ang, fish2_trn, ...]
            [fish1_lin, fish1_ang, fish1_trn, fish2_lin, fish2_ang, fish2_trn, ...]
            ...
        ]
    startpoints:
        two nodes per fish exactly:
        [head1_x, head1_y, center1_x, center1_y, head2_x, head2_y, center2_x, center2_y,...]
    Output:

        [
            [center1_x, center1_y, orientation1, ...]
            [center1_x, center1_y, orientation1, ...]
            ...
        ]
    """
    row, col = loc.shape
    assert row > 1
    assert col % 3 == 0
    nfish = col // 3
    assert len(startpoints) / nfish == 4

    # save [center1_x, center2_y, orientation1, center2_x, ...] for every fish
    out = np.empty([row + 1,nfish * 3])

    # 1. Distances Center - Head, out setup
    disCH = []
    for f in range(nfish):
        disCH.append( getDistance( startpoints[4 * f], startpoints[4 * f + 1], startpoints[4 * f + 2], startpoints[4 * f + 3] ) )
        out[0,3 * f] = startpoints[4 * f + 2]
        out[0,3 * f + 1] = startpoints[4 * f + 3]
        # Angle between Fish Orientation and the unit vector
        out[0,3 * f + 2] = getAngle( (1,0,), (startpoints[4 * f] - startpoints[4 * f + 2], startpoints[4 * f + 1] - startpoints[4 * f + 3],), "radians" )

    for i in range(0, row):
        out[i + 1] = row_l2c( out[i], loc[i] )

    return convPolarToCart( out, disCH )

def updateLocomotions():
    """
    Update all locomotion files
    """
    getLocomotion( extract_coordinates( "data/sleap_1_diff1.h5", [b'head',b'center'], fish_to_extract=[0,1,2]), "data/locomotion_data_diff1.csv" )
    getLocomotion( extract_coordinates( "data/sleap_1_diff2.h5", [b'head',b'center'], fish_to_extract=[0,1,2]), "data/locomotion_data_diff2.csv" )
    getLocomotion( extract_coordinates( "data/sleap_1_diff3.h5", [b'head',b'center'], fish_to_extract=[0,1,2])[0:17000], "data/locomotion_data_diff3.csv" )
    getLocomotion( extract_coordinates( "data/sleap_1_diff4.h5", [b'head',b'center'], fish_to_extract=[0,1,2])[120:], "data/locomotion_data_diff4.csv" )
    getLocomotion( extract_coordinates( "data/sleap_1_same1.h5", [b'head',b'center'], fish_to_extract=[0,1,2]), "data/locomotion_data_same1.csv" )
    getLocomotion( extract_coordinates( "data/sleap_1_same3.h5", [b'head',b'center'], fish_to_extract=[0,1,2])[130:], "data/locomotion_data_same3.csv" )
    getLocomotion( extract_coordinates( "data/sleap_1_same4.h5", [b'head',b'center'], fish_to_extract=[0,1,2]), "data/locomotion_data_same4.csv" )
    getLocomotion( extract_coordinates( "data/sleap_1_same5.h5", [b'head',b'center'], fish_to_extract=[0,1,2]), "data/locomotion_data_same5.csv" )

def main():

    updateLocomotions()

    # get locomotion
    # df = pd.read_csv("data/locomotion_data_diff2.csv", sep = ";")
    # loc = df.to_numpy()

    # convLocToCart( loc, [282.05801392, 85.2730484, 278.16235352, 112.26922607, 396.72821045, 223.87356567, 388.54510498, 198.40411377, 345.84439087, 438.7845459, 325.3197937, 426.67755127] )

    # convertLocmotionToBin(loco, "data/clusters.txt", "data/locomotion_data_bin_diff4.csv")

if __name__ == "__main__":
    main()