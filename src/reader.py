import h5py
import pandas as pd
import numpy as np
import sys

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
    Output: nparray of form: [fishes, x/y coordinate, node_names, frames]
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

    # Return appropiate data: only wanted nodes, and wanted fishes
    return tracks[:, :, node_indices, :][fish_to_extract,:,:,:]



if __name__ == "__main__":
    # just a test
    # a, b, c, d = read_slp("data/MARC_USE_THIS_DATA.h5", True)
    # print(a)
    # print(b)
    file = "data/MARC_USE_THIS_DATA.h5"

    output = extract_coordinates(file, [b'head', b'center'], fish_to_extract=[0,1,2])
    print(output)
    print(output[0,:,:,0]) #fish 0 at frame 0
    print(output.shape)

    # only head node x and y from fish 0 data:
    # headonly = extract_coordinates(file, [b'head'], [0])