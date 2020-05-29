import h5py
import pandas as pd

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

    return (node_names, track_names, track_occupancy, tracks)


def extract_coordinates(file, nodes_to_extract, fish_to_extract):
    """
    Extracts coordinates for given node names in file and returns pandas dataframe
    nodes_to_extract: String array containing at least one of: [b'head', b'center', b'l_fin_basis', b'r_fin_basis', b'l_fin_end', b'r_fin_end', b'l_body', b'r_body', b'tail_basis', b'tail_end']
    """
    # Get h5 object
    node_names, track_names, track_occupancy, tracks = read_slp(file)




if __name__ == "__main__":
    # just a test
    read_slp("data/MARC_USE_THIS_DATA.h5", True)