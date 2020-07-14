# Python file for the nmodel
import numpy as np
import pandas as pd
from functions import getDistances, getAngles
from reader import extract_coordinates

def getnView( tracksFish, tracksOther, nfish=3 ):
    """
    Input: tracks from protagonist fish, only [head,center] expected
    Output gives distances to all nodes from other fishes in tracks
    nfish total amount of fish (fish in tracksOther + 1)
    """
    rows, cols = tracksFish.shape
    rows2, cols2 = tracksOther.shape
    assert rows == rows2
    assert rows > 1
    assert cols == 4
    nnodes = cols2 // (nfish - 1) // 2

    out = np.empty( ( rows, nnodes * (nfish-1) * 2 ) )
    center = tracksFish[:,[2,3]]
    # 1. Head - Center
    vec_ch = tracksFish[:,[0,1]] - center
    # 2. compute values for every fish and node
    for f in range( nfish - 1 ):
        for n in range( nnodes ):
            ixy = [nnodes * f + 2 * n, nnodes * f + 2 * n + 1]
            # node - center
            vec_cn = tracksOther[:,ixy] - center
            out[:,nnodes * 2 * f + 2 * n] = getDistances( tracksOther[:,ixy], center )
            out[:,nnodes * 2 * f + 2 * n + 1] = getAngles( vec_ch, vec_cn )

    return out


def stealWallRays( path, COUNT_RAYS_WALLS=15 ):
    """
    Steals wall rays from raycast data
    """
    raycasts = pd.read_csv( path, sep = ";" ).to_numpy()
    return raycasts[:,-15:]


def main():
    tracks = extract_coordinates( "data/sleap_1_diff1.h5", [b'head', b'center', b'tail_basis'] )
    print( tracks[0] )
    print( getnView( tracks[:,0:4], tracks[:,6:] )[0] )
    print( stealWallRays( "data/raycast_data_diff1.csv" ).shape )

if __name__ == "__main__":
    main()