import h5py
import pandas as pd
import numpy as np
import sys
import scipy
import functions

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


def extract_rows(file, nodes_to_extract, fish_to_extract = [0,1,2], verbose = False):
    """
    Extracts specific rows for given sleap file and returns numpy array
    nodes_to_extract: String array containing at least one of: [b'head', b'center', b'l_fin_basis', b'r_fin_basis', b'l_fin_end', b'r_fin_end', b'l_body', b'r_body', b'tail_basis', b'tail_end']
    fish_to_extract: array containing the fishes you want to extract
    Output: nparray of form: [frames, [node0_fish0_x, node0_fish0_y, node1_fish0_x, node1_fish0_y, ..., node0_fish1_x, node0_fish1_y, ...]
    """

    # Get data from file
    node_names, track_names, track_occupancy, tracks = read_slp(file)

    # Print some information about read data
    n_frames = len(tracks[0,0,0,:])
    n_fishes = len(track_names[:])

    if verbose:
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
        if verbose:
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

    return rtracks[:, e_indices]


def interpolate_missing_values(data, verbose = False):
    """
    input: values in the format of extract_rows()
    output: values in same format, without nan rows
    careful: this directly modifies your data
    """
    if verbose:
        print("Interpolated missing values")
    n_row, n_col = data.shape

    # Iterate through every row for each column
    for col in range(n_col):

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


def replace_point_by_middle(data, col, i1, i2, i3):
    """
    Replaces i2 by the point in the middle of i1 and i2, if i2 does not exist,
    it takes the vector from i-1 to i and adds it on i for i2
    """
    n_rows, n_cols = data.shape
    if i3 < n_rows:
        # Take middlepoint from value before and behind outlier as new value
        midway = (data[i1, col] - data[i3, col])/2
        newval = data[i1, col] - midway
    else:
        # Take difference from both values before and replace outlier with it
        lastdiff = data[i1 - 1, col] - data[i1, col]
        newval = data[i1, col] - lastdiff

    data[i2, col] = newval


def correct_outlier(data, col1, col2, i1, max_tolerated_movement):
    """
    Recursive function to adjust distance between i1 and i1+1 so that it is not marked as outliers.
    Recursively removes outliers created this way
    I am pretty sure this is a more special case of correct_wrongly_interpolated_outliers. But refactoring is not the job now
    """
    n_rows, n_cols = data.shape
    assert i1 + 1 < n_rows

    replace_point_by_middle(data, col1, i1, i1 + 1, i1 + 2)
    replace_point_by_middle(data, col2, i1, i1 + 1, i1 + 2)

    dis_1 = functions.getDistance(data[i1,col1], data[i1,col2], data[i1 + 1,col1], data[i1 + 1,col2])
    if i1 + 2 < n_rows:
        dis_2 = functions.getDistance(data[i1 + 1,col1], data[i1 + 1,col2], data[i1 + 2,col1], data[i1 + 2,col2])
    if abs(dis_1) > max_tolerated_movement:
        if i1 == 0:
            # Edge case when you are at beginning, replace with value above
            data[i1, col1] = data[i1 + 1, col1]
            data[i1, col2] = data[i1 + 1, col2]
        else:
            # Recursively adjust values
            correct_outlier(data, col1, col2, i1-1, max_tolerated_movement)
    if i1 + 2 < n_rows and abs(dis_2) > max_tolerated_movement:
        if i1 + 3 == n_rows:
            # Edge case when you are at the start, replace with value below
            data[i1 + 2, col1] = data[i1 + 1, col1]
            data[i1 + 2, col2] = data[i1 + 1, col2]
        else:
            # Recursively adjust values
            correct_outlier(data, col1, col2, i1+1, max_tolerated_movement)


def fill_steps_in(data, col, i1, i2, n_steps):
    """
    Fills all values in col of data between i1 and i2 evenly spaced between value of i1 and i2
    """
    first = data[i1,col]
    step = (first - data[i2, col])/n_steps
    for x in range(1,n_steps):
        data[i1 + x, col] = first - step * x


def fill_vector_in(data, col, i1, i2):
    """
    Fills all values in col of data between i1 and i2 evenly spaced with same size as the value i1 - i2
    """
    first = data[i1, col]
    dis = first - data[i1 - 1, col]
    n_steps = i2 - i1
    for x in range(1, n_steps):
        data[i1 + x, col] = first + dis * x


def correct_wrongly_interpolated_outliers(data, col1, col2, out_group, max_tolerated_movement):
    """
    Interpolates the values newly for which due to prediction mistakes data was interpolated wrongly
    """
    n_rows, n_cols = data.shape
    assert n_cols > 1
    assert len(out_group) > 2
    i_first = out_group[0]
    if out_group[-1] + 1 < n_rows:
        # Interpolate values newly between first and last non outlier value
        i_last = out_group[-1] + 1
        n_vals = len(out_group)

        fill_steps_in(data, col1, i_first, i_last, n_vals)  # x
        fill_steps_in(data, col2, i_first, i_last, n_vals)  # y
    else:
        # Edge Case, you reached the end of the file (I wonder if this code will ever be executed)
        print("OH MY GOD, THIS REALLY HAS BEEN EXECUTED")
        # Take vector from last defined value
        i_last = out_group[-1]

        fill_vector_in(data, col1, i_first, i_last)
        fill_vector_in(data, col2, i_first, i_last)

    # Check if distance for first value has gotten better, (others are checked later anyway)
    dis = functions.getDistance(data[i_first,col1], data[i_first,col2], data[i_first + 1,col1], data[i_first + 1,col2])
    if abs(dis) > max_tolerated_movement:
        # Enter fixing loop
        correct_outlier(data, col1, col2, i_first, max_tolerated_movement)


def interpolate_outliers_rec(data, max_tolerated_movement=20, verbose = False):
    """
    input: values in the format of extract_coordinates()
    output: values in same format, without outlier values
    careful: this directly modifies your data
    do not set max_tolerated_movement less then 16 (15.06) - it will not work :)
    """
    if verbose:
        print("Interpolate Outliers:")          # Announce this function loudly and passionately
    n_rows, n_cols = data.shape

    # Get distances of all points between 2 frames
    dist = functions.get_distances(data)

    if verbose:
        print("Before:")
        print("avg:", np.mean(dist, axis=0))
        print("max:", np.amax(dist, axis=0))
        print("min:", np.amin(dist, axis=0))

    assert n_rows >= 2
    for col in range(int(n_cols/2)):
        # Get outliers of specific column
        i_out =list( np.where(dist[:,col] > max_tolerated_movement)[0])
        # column indices in 'data'
        col1, col2 = 2*col, 2*col + 1

        for outlier in i_out:
            # recheck if outlier candidate is still valid, often when fixing an outlier you fix it for the next distance aswell
            dis = functions.getDistance(data[outlier,col1], data[outlier,col2], data[outlier + 1,col1], data[outlier + 1,col2])
            if abs(dis) > max_tolerated_movement:
                if outlier + 1 in i_out and outlier + 2 in i_out:
                    # This case is if the interpolation lead to outliers, since the positions outside of the nan values are too far away.
                    # Get all frames which are behind each other+
                    out_group = [outlier, outlier + 1, outlier + 2,]
                    outlier += 2
                    while outlier + 1 in i_out:
                        out_group.append(outlier + 1)
                        outlier += 1
                    correct_wrongly_interpolated_outliers(data, col1, col2, out_group, max_tolerated_movement)
                else:
                    # correct outlier by replacing it with the value in the middle from points before and after it
                    correct_outlier(data, col1, col2, outlier, max_tolerated_movement)

    # To check, we recalculate distances and look if there is any outliers still left
    if verbose:
        dist = functions.get_distances(data)
        print("After:")
        print("avg:", np.mean(dist, axis=0))
        print("max:", np.amax(dist, axis=0))
        print("min:", np.amin(dist, axis=0))
        print("Outliers left: ", np.where(dist[:,] > max_tolerated_movement))


def error_start(index):
    """
    error message and print. Quick and dirty
    """
    print("Max_tolerated movement violation at start of video:")
    print("Cut video before Frame ", index)
    sys.exit()


def error_end(index):
    """
    error message print. Quick and dirty
    """
    print("Max_tolerated movement violation at end of video:")
    print("Cut video after Frame ", index)
    sys.exit()


def avg_dis(data, col1, col2, i_start, i_end, n_points):
    """
    Computes the average distance between i_start and i_end
    for given amount of n_points
    """
    assert n_points > 1
    return functions.getDistance(data[i_start,col1], data[i_start, col2], data[i_end,col1], data[i_end,col2]) / (n_points - 1)


def interpolate_outliers(data, max_tolerated_movement=20, verbose = False):
    """
    input: values in the format of extract_coordinates()
    output: values in same format, without outlier values
    careful: this directly modifies your data
    do not set max_tolerated_movement less then 16 (15.06) - it will not work :)
    """
    if verbose:
        print("Interpolate Outliers:")          # Announce this function loudly and passionately
    n_rows, n_cols = data.shape

    # Get distances of all points between 2 frames
    dist = functions.get_distances(data)

    if verbose:
        print("Before:")
        print("avg:", np.mean(dist, axis=0))
        print("max:", np.amax(dist, axis=0))
        print("min:", np.amin(dist, axis=0))

    assert n_rows >= 2
    for col in range(int(n_cols/2)):
        # Get outliers of specific column
        i_out =list( np.where(dist[:,col] > max_tolerated_movement)[0])
        # column indices in 'data'
        col1, col2 = 2*col, 2*col + 1

        for outlier in i_out:
            # recheck if outlier candidate is still valid, often when fixing an outlier you fix it for the next distance aswell
            curr_dis = functions.getDistance(data[outlier,col1], data[outlier,col2], data[outlier + 1,col1], data[outlier + 1,col2])
            if abs(curr_dis) > max_tolerated_movement:
                i_curr = outlier + 1
                i_start = outlier
                # Find next point which is no outlier
                while( dist[i_curr,col] > max_tolerated_movement ):
                    i_curr += 1
                    if i_curr >= n_rows:
                        error_end(outlier)
                i_end = i_curr
                # Compute distances and check if we would interpolate,
                # would the dis be > then max_tolerated_movement
                # if yes, take further points until it works
                switch = True
                while( avg_dis(data, col1, col2, i_start, i_end, (i_end - i_start + 1)) > max_tolerated_movement ):
                    if switch:
                        i_end += 1
                        if( i_end >= n_rows ):
                            error_end(outlier)
                    else:
                        i_start -= 1
                        if( i_start < 0 ):
                            error_start(i_curr) #i_curr because it has the latest og outlier

                fill_steps_in(data, col1, i_start, i_end, (i_end - i_start))
                fill_steps_in(data, col2, i_start, i_end, (i_end - i_start))


    # To check, we recalculate distances and look if there is any outliers still left
    if verbose:
        dist = functions.get_distances(data)
        print("After:")
        print("avg:", np.mean(dist, axis=0))
        print("max:", np.amax(dist, axis=0))
        print("min:", np.amin(dist, axis=0))
        print("Outliers left: ", np.where(dist[:,] > max_tolerated_movement))


def extract_coordinates(file, nodes_to_extract, fish_to_extract = [0,1,2], interpolate_nans = True, interpolate_outlier = True, i_o_rec = False, verbose = False):
    """
    Extracts specific rows for given sleap file and returns numpy array, cleaning up data if not specified otherwise
    interpolate missing values will always be run if interpolate outliers is activated
    nodes_to_extract: String array containing at least one of: [b'head', b'center', b'l_fin_basis', b'r_fin_basis', b'l_fin_end', b'r_fin_end', b'l_body', b'r_body', b'tail_basis', b'tail_end']
    fish_to_extract: array containing the fishes you want to extract
    Output: nparray of form: [frames, [node0_fish0_x, node0_fish0_y, node1_fish0_x, node1_fish0_y, ..., node0_fish1_x, node0_fish1_y, ...]
    i_o_rec: interpolation of outliers recursive, may run in recursion limit and takes
             a bit longer, but sometimes is more accurate and handles edge cases
    """
    ret = extract_rows(file, nodes_to_extract, fish_to_extract, verbose=verbose)

    if interpolate_nans or interpolate_outlier:
        interpolate_missing_values(ret, verbose=verbose)

    if interpolate_outlier:
        if i_o_rec:
            interpolate_outliers_rec(ret, verbose=verbose)
        else:
            interpolate_outliers(ret, verbose=verbose)

    return ret



if __name__ == "__main__":
    file = "data/sleap_1_diff4.h5"

    output = extract_coordinates(file, [b'head', b'center', b'l_fin_basis', b'r_fin_basis', b'l_fin_end', b'r_fin_end', b'l_body', b'r_body', b'tail_basis', b'tail_end'], fish_to_extract=[0], verbose=True)
    #output2 = extract_coordinates(file2, [b'head'], fish_to_extract=[0])
    print(output.shape)
    #print(output2[9145:9147,])
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

    # woutlier = interpolate_outliers(output)
    #print(output.mean(axis = 0))
    #print(woutlier)
