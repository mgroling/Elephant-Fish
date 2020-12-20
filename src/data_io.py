# File to load data

import numpy as np
import pandas as pd

def loadTrackData( path ):
    """
    Loads trackData from path
    """
    return np.load( path )


def lazytrackData( n ):
    """
    Loads predefined datasets
    """
    track1 = "data/trackData/track1.npy"
    track2 = "data/trackData/track2.npy"
    track3 = "data/trackData/track3.npy"
    track4 = "data/trackData/track4.npy"
    track5 = "data/trackData/track5.npy"
    track6 = "data/trackData/track6.npy"
    track7 = "data/trackData/track7.npy"
    track8 = "data/trackData/track8.npy"
    tracks = [ track1, track2, track3, track4, track5, track6, track7, track8 ]
    return np.load( tracks[n] )


def loadRaycastData( path ):
    """
    Loads raycast data from path
    """
    raycasts = pd.read_csv( path, sep = ";" ).to_numpy()
    print( raycasts )
    print( raycasts.shape )
