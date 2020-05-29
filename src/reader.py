import h5py
import pandas as pd
import numpy as np
import sys

def read_slp(filepath, loud = False):
    """
    reads a sleap file and converts it into appropiate format
    """
    # funktioniert meines erachtens nicht, vielleicht hab ich es auch nicht richtig gemacht
    # raw_data = pd.read_hdf(filepath)
    # print(type(raw_data))


    with h5py.File(filepath, 'r') as f:
        node_names = f['node_names']
        track_names = f['track_names'] # am ende nur noch 3
        track_occupancy = f['track_occupancy']
        tracks = f['tracks']

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
            print(track_occupancy.shape) # VERMUTUNG: Boolean in welchem Frame ein Track aktiv ist
            print(track_occupancy[0,0])
            print(track_occupancy[11000,0]) # <- Theorie bestätigt (visueller abgleich im Programm)
            print("\nTRACKS")
            print(tracks.shape) # der heiße scheiß, zweite SPALTE SIND X UND Y, ICH WEIß LEIDER NICHT WIE RUM (aber ist ja eigentlich auch erstmal egal), 4. Spalte obviously die Frames
            print(tracks[0,:,:,0]) # Fisch 0
            print(tracks[1,:,:,0]) # Fisch 1
            print(tracks[40,:,:,0]) # Fisch 40 nan weil nicht im Bild bei frame 0, bzw track_occupancy[0,40] wird 0 sein

            # CONFIDENCE WIRD NICHT MITGEGEBEN. ICH WERD NOCHMAL SCHAUEN WIE MAN DAS EVTL- berücksichtigen kann.

            # wahrscheinlich sinnvoll hier schon iwie datenverarbeitung zu machen (integrity checks, ob die daten im richtigen Rahmen sind etc.)
    # print(node_names[:]) # this does not work since out of the if the dataframes do not seem to be defined anymore.
    # return node_names


def extract_coordinates(file, nodes_to_extract, fish_to_extract):
    """
    Extracts coordinates for given node names in file and returns pandas dataframe
    nodes_to_extract: String array containing at least one of: [b'head', b'center', b'l_fin_basis', b'r_fin_basis', b'l_fin_end', b'r_fin_end', b'l_body', b'r_body', b'tail_basis', b'tail_end']
    fish_to_extract: 0,1 or 2
    """

    if fish_to_extract < 0 or fish_to_extract > 3:
        print("Error: invalid fish")
        sys.exit()

    # Open file
    with h5py.File(file, 'r') as f:
        node_names = f['node_names']
        track_names = f['track_names']
        track_occupancy = f['track_occupancy']
        tracks = f['tracks'] #[fish, x/y, boolean_track, ]

        # Get indices of wanted nodes
        node_indices = np.where( np.in1d(node_names, nodes_to_extract) )[0]
        if len(node_indices) == 0:
            print("Error: invalid node_names")
        else:
            print("Node indices are ", node_indices)





if __name__ == "__main__":
    # just a test
    # read_slp("data/MARC_USE_THIS_DATA.h5", True)
    extract_coordinates("data/MARC_USE_THIS_DATA.h5", [b'head', b'center'], 0)