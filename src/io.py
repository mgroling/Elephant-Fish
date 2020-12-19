# File to load track coordinates

import numpy as np

def loadTrack( filename ):
    """
    Loads trackdata from filename
    """
    return np.load( filename )