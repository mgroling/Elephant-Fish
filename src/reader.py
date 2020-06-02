import h5py
import pandas as pd
import numpy as np
import sys
import scipy
import locomotion

def read_slp(file, loud = False):
    """
    reads a sleap file and converts it into appropiate format
    """

    # Open file
    with h5py.File(file, 'r') as f:
        node_names = f['node_names'] # [names of nodes]
        track_names = f['track_names'] # [names of tracked instances]
        track_occupancy = f['track_occupancy'] # [frames, tracked_Instance]
        tracks = f['tracks'] # [fish, x/y, tracking data, frames ]

        # If you want to know how the data looks like
        if loud:
            print(type(f))
            print(f.keys())
            print("\nNODE NAMES")
            print(node_names.shape) # 3. SPALTE in trakcs
            print(node_names[:])
            print(type(node_names))
            print("\nTRACK NAMES")
            print(track_names.shape) # 1. SPALTE in tracks, am Ende nur 3 - gibt dir den getrackten Fisch
            print(track_names[0:9])
            print("\nTRACK OCCUPANCY")
            print(track_occupancy.shape) # Boolean in welchem Frame ein Track aktiv ist
            print(track_occupancy[0,0])
            print(track_occupancy[11000,0])
            print("\nTRACKS")
            print(tracks.shape) # der heiße scheiß, zweite SPALTE SIND X UND Y, ICH WEIß LEIDER NICHT WIE RUM (aber ist ja eigentlich auch erstmal egal), 4. Spalte obviously die Frames
            print(tracks[0,:,:,0]) # Fisch 0
            print(tracks[0,:,:,30]) # Fisch 0 nach 30 frames
            print(tracks[1,:,:,0]) # Fisch 1
            print(tracks[2,:,:,0]) # Fisch 2

            # CONFIDENCE WIRD NICHT MITGEGEBEN. ICH WERD NOCHMAL SCHAUEN WIE MAN DAS EVTL- berücksichtigen kann.

        return (node_names[:], track_names[:], track_occupancy[:], tracks[:])


def extract_coordinates(file, nodes_to_extract, fish_to_extract = [0,1,2]):
    """
    Extracts coordinates for given node names in file and returns pandas dataframe
    nodes_to_extract: String array containing at least one of: [b'head', b'center', b'l_fin_basis', b'r_fin_basis', b'l_fin_end', b'r_fin_end', b'l_body', b'r_body', b'tail_basis', b'tail_end']
    fish_to_extract: array containing the fishes you want to extract
    Output: nparray of form: [frames, [node0_fish0_x, node0_fish0_y, node1_fish0_x, node1_fish0_y, ..., node0_fish1_x, node0_fish1_y, ...]
    """

    # Get data from file
    node_names, track_names, track_occupancy, tracks = read_slp(file)

    # Print some information about read data
    n_frames = len(tracks[0,0,0,:])
    n_fishes = len(track_names[:])

    print("File: ", file)
    print("Nodes: ", len(node_names[:]))
    print("Frames in video: ", )
    print("Fish in dataset: ", n_fishes)
    print("Fish - Tracked frames: ")
    for x in range(n_fishes):
        print("{} - {} frames".format(x, len(np.where(track_occupancy[:,x] == 1)[0] )))

    if max(fish_to_extract) >= n_fishes:
        print("Error: invalid fishes in argument fish_to_extract")
        sys.exit

    # Get indices of wanted nodes
    node_indices = list(np.where( np.in1d(node_names, nodes_to_extract) )[0])
    if len(node_indices) == 0:
        print("Error: invalid node_names")
    elif len(node_indices) != len(nodes_to_extract):
        print("Error: mapping node_names to nodes in file failed")
    else:
        print("Node indices are ", node_indices)

    # Put frames second, convert to 2d array
    tr = tracks.reshape(-1, tracks.shape[-1]).transpose()

    # Swap x and y values into right positions
    # Stackoverflow magic https://stackoverflow.com/a/20265477
    permutation = [0,2,4,6,8,10,12,14,16,18,1,3,5,7,9,11,13,15,17,19,20,22,24,26,28,30,32,34,36,38,21,23,25,27,29,31,33,35,37,39,40,42,44,46,48,50,52,54,56,58,41,43,45,47,49,51,53,55,57,59]
    idx = np.empty_like(permutation)
    idx[permutation] = np.arange(len(permutation))
    rtracks = tr[:, idx]

    # Return appropiate data: only wanted nodes, and wanted fishes
    e_indices = []
    for i in fish_to_extract:
        # calculate correct positions for indices
        x_pos = list(map(lambda x: 2 * x + (20 * i), node_indices))
        y_pos = list(map(lambda x: 2 * x + 1 + (20 * i), node_indices))
        # merge both into one list
        appendix = [pos for tup in zip(x_pos,y_pos) for pos in tup]

        e_indices = e_indices + appendix

    return interpolate_missing_values(rtracks[:, e_indices])


def interpolate_missing_values(data):
    """
    input: values in the format of extract_coordinates()
    output: values in same format, without nan rows
    careful: this directly modifies your data
    """
    print("Interpolate Missing values:")
    n_row, n_col = data.shape

    # Iterate through every row for each column
    for col in range(n_col):
        print("column {}".format(col))

        curr_row = 0
        last_not_nan_row = -1
        while curr_row < n_row :

            if not np.isnan(data[curr_row, col]):
                # Marc (harr harr) as non nan value
                last_not_nan_row = curr_row
            elif last_not_nan_row == -1:
                # Edge case, you start with nan value
                # This case, take first defined value und place it inside
                while np.isnan(data[curr_row, col]):
                    curr_row += 1
                last_not_nan_row = 0
                while last_not_nan_row < curr_row:
                    data[last_not_nan_row, col] = data[curr_row, col]
                    last_not_nan_row += 1
                # @TODO (if time): derive movement vector and add that to missing values (difficulty: if there is a directly missing nan afterwards)
            else:
                # move to next non nan row
                rows_moved = 0
                while curr_row < n_row and np.isnan(data[curr_row, col]):
                    curr_row += 1
                    rows_moved += 1

                last_real_value = data[last_not_nan_row, col]

                if curr_row >= n_row:
                    # Edge Case, you end with nan value:
                    # Just place last defined value in
                    while last_not_nan_row < curr_row:
                        data[last_not_nan_row, col] = last_real_value
                        last_not_nan_row += 1
                else:
                    # compute distance between both defined rows
                    distance =  last_real_value - data[curr_row, col]
                    # compute step sizes in between
                    step = distance / (rows_moved + 1)
                    # Fill values in between
                    last_not_nan_row += 1
                    step_count = 1
                    while last_not_nan_row < curr_row:
                        data[last_not_nan_row, col] = last_real_value - step * step_count
                        assert step_count <= rows_moved
                        step_count += 1
                        last_not_nan_row += 1

            curr_row += 1
    return data



def interpolate_outliers(data, max_tolerated_movement=12):
    """
    input: values in the format of extract_coordinates()
    output: values in same format, without outlier values
    careful: this directly modifies your data
    """
    n_rows, n_cols = data.shape
    assert n_cols % 2 == 0
    assert n_cols > 1
    # Get distances of all points between 2 frames
    lastrow = data[data.shape[0] - 1]       # shift all data by one to the front, double the last row
    data2 = np.vstack( (np.delete(data, 0, 0), lastrow) )
    mov = data - data2                      # subract x_curr x_next
    mov = mov**2                            # power
    dist = np.sum(mov[:,[0,1]], axis = 1)   # add x and y to eachother
    for i in range(1,int(n_cols/2)):        # do to the rest of the cols
        dist = np.vstack((dist, np.sum(mov[:,[2*i,2*i + 1]], axis = 1) ))
    dist = np.sqrt(dist.T)                  # take square root to gain distances

    dist = dist[0:(dist.shape[0] - 1),]    # get rid of last column (it is 0)
    print("avg:", np.mean(dist, axis=0))
    print("max:", np.amax(dist, axis=0))
    print("min:", np.amin(dist, axis=0))
    print(np.where(dist > max_tolerated_movement))


if __name__ == "__main__":
    file = "data/sleap_1_Diffgroup1-1.h5"

    output = extract_coordinates(file, [b'head'], fish_to_extract=[0])
    # print("First 20 rows")
    # print(output[0:20,:])
    # print("nan values")
    # print(np.where(np.isnan(output)))
    # print("With removed values:")
    # output[12,0] = float('nan')
    # output[13,0] = float('nan')
    # output[14,0] = float('nan')
    # output[16,0] = float('nan')
    # output[3,1] = float('nan')
    # output[0,3] = float('nan')
    # output[0,2] = float('nan')
    # output[1,2] = float('nan')
    # output[2,2] = float('nan')
    # output[19,1] = float('nan')
    # output[19,3] = float('nan')
    # output[18,3] = float('nan')
    # output[17,3] = float('nan')
    # print(output[0:20,:])
    # interpolate_missing_values(output)

    woutlier = interpolate_outliers(output)
    #print(output.mean(axis = 0))
    #print(woutlier)
