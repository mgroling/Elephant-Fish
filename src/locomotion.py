import numpy as np
import pandas as pd
import math
from functions import getAngle, getDistance, readClusters, distancesToClusters, softmax, get_indices
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
            vector_next = (center_x - center_x_next, center_y - center_y_next)

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


def row_l2c( locs, coords ):
    """
    Returns 1d ndarray with new coordinates based on previos coordinades and given locs
    """
    nfish = len(coords) // 2

    out = np.array( coords )

    # for f in range(nfish):
    #     ix, iy = get_indices( f )
    #     x = coords[ix]
    #     y = coords[iy]
    #     lin = locs[2 * f]
    #     ang = locs[2 * f + 1]

    #     nang = (ang + trns[f]) % 2*np.pi

    #     # Polar to kartesian
    #     vx = math.cos( nang ) * lin
    #     vy = math.sin( nang ) * lin

    xs = [4 * x for x in range(nfish)]
    ys = [4 * x + 1 for x in range(nfish)]
    lins = np.abs( locs[[3 * x for x in range(nfish)]] )
    angs = locs[[3 * x + 1 for x in range(nfish)]]
    trns = locs[[3 * x + 2 for x in range(nfish)]]
    nangs = (angs + trns) % ( 2 * np.pi )
    print( "lins ", lins )
    print( "angs ", angs )
    print( "trns ", trns )
    print( "nangs ", nangs )
    xvals = np.cos( nangs ) * lins
    yvals = np.sin( nangs ) * lins

    print(xvals)
    print(yvals)

    print(out)
    t = out.copy()

    out[xs] = out[xs] + yvals
    out[ys] = out[ys] + xvals

    print(out)

    print( getDistance(t[0], t[1], out[0], out[1]) )



def convertLocomotionToCoordinates( loc, startpoints ):
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
    """
    row, col = loc.shape
    assert col % 3 == 0
    nfish = col // 3
    assert len(startpoints) / nfish == 4

    out = np.empty([col + 1,nfish * 4])
    out[0] = startpoints

    # # Separate turn and other velocities, as they are shifted by one
    # inds = [3 * x + 2 for x in range(nfish)]
    # locT = loc[:,inds]
    # locT = np.insert( locT, 0, np.array( [0.0] * nfish), 0 )
    # inds = [x for x in range(col) if (x % 3) == 1 or (x % 3) == 0]
    # loc = loc[:,inds]

    row_l2c( loc[0] , out[0] )

    for i in range(1, row + 1):
        pass




def main():
    file = "data/sleap_1_diff2.h5"

    temp = extract_coordinates(file, [b'head'], fish_to_extract=[0,1,2])

    print(temp[0])
    print(temp[1])
    print( getDistance(temp[0,0], temp[0,1], temp[1,0], temp[1,1]) )

    # get locomotion
    df = pd.read_csv("data/locomotion_data_diff2.csv", sep = ";")
    loc = df.to_numpy()
    print("loc 0: ", loc[0])

    convertLocomotionToCoordinates( loc, [282.05801392, 85.2730484, 396.72821045, 223.87356567, 345.84439087, 438.7845459] )


    # convertLocmotionToBin(loco, "data/clusters.txt", "data/locomotion_data_bin_diff4.csv")

if __name__ == "__main__":
    main()