# Python file for the nmodel
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from functions import getDistances, getAngles, getDistance, getAngle, defineLines, getRedPoints
from reader import extract_coordinates
from locomotion import getnLoc, row_l2c, row_l2c_additional_nodes, row_l2c_additional_nodes
from evaluation import plot_train_history
from raycasts import Raycast

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
            ixy = [2 * nnodes * f + 2 * n, 2 * nnodes * f + 2 * n + 1]
            # node - center
            vec_cn = tracksOther[:,ixy] - center
            out[:,nnodes * 2 * f + 2 * n] = getDistances( tracksOther[:,ixy], center )
            out[:,nnodes * 2 * f + 2 * n + 1] = getAngles( vec_ch, vec_cn )

    return out

def getnViewSingle( posFish, posOther, nnodes, nfish=3 ):
    """
    Same as above but for one row
    """
    center = np.array( posFish[2:4] )
    head = np.array( posFish[0:2] )
    vec_ch = head - center
    out = np.empty( ( nnodes * (nfish - 1) * 2 ) )
    for f in range( nfish - 1 ):
        for n in range( nnodes ):
            ixy = [2 * nnodes * f + 2 * n, 2 * nnodes * f + 2 * n + 1]
            # node - center
            vec_cn = posOther[ixy] - center
            out[nnodes * 2 * f + 2 * n] = getDistance( (posOther[ixy])[0], (posOther[ixy])[1], center[0], center[1] )
            out[nnodes * 2 * f + 2 * n + 1] = getAngle( vec_ch, vec_cn, "radians" )
    return out


def stealWallRays( path, COUNT_RAYS_WALLS=15, nfish=3 ):
    """
    Steals wall rays from raycast data
    """
    raycasts = pd.read_csv( path, sep = ";" ).to_numpy()
    return raycasts[:,-15 * nfish:]


def createModel( name, U_LSTM, U_DENSE, U_OUT, input_shape, dropout=None ):
    """
    Creates and returns an RNN model
    droupout example: [0.1, 0.1]
    """
    nmodel = tf.keras.models.Sequential( name=name )
    print( "input shape: {}".format( input_shape ) )
    nmodel.add( tf.keras.layers.LSTM( U_LSTM , input_shape=input_shape ) )
    if dropout is not None:
        nmodel.add( tf.keras.layers.Dropout( dropout[0] ) )
    nmodel.add( tf.keras.layers.Dense( U_DENSE ) )
    if dropout is not None:
        nmodel.add( tf.keras.layers.Dropout( dropout[1] ) )
    nmodel.add( tf.keras.layers.Dense( U_OUT ) )
    nmodel.compile( optimizer=tf.keras.optimizers.RMSprop(), loss='mean_squared_error' )
    nmodel.summary()
    return nmodel


def multivariate_data( dataset, target, start_index, end_index, history_size, target_size, step, single_step=False ):
    """
    Taken from https://www.tensorflow.org/tutorials/structured_data/time_series
    """
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len( dataset ) - target_size

    for i in range( start_index, end_index ):
        indices = range( i - history_size, i, step )
        data.append( dataset[indices] )

        if single_step:
            labels.append( target[i + target_size] )
        else:
            labels.append( target[i:i + target_size] )

    return np.array( data ), np.array( labels )


def loadData( pathsTracksets, pathsRaycasts, nodes, nfish, N_WRAYS, N_VIEWS, D_LOC, D_DATA, D_OUT, HIST_SIZE, TARGET_SIZE, mean, SPLIT=0.9, getmean=False, pathToSave=None ):
    """
    pathsTrackset and pathsRaycast need to be in same order
    """
    assert len( pathsTracksets ) == len( pathsRaycasts )
    nnodes = len( nodes )

    if mean is not None:
        arr = np.load( mean, allow_pickle=True )
        meanv = arr[0]
        std = arr[1]
        meanTGT = arr[2]
        stdTGT = arr[3]
        print( "Using mean and std:")
        print( meanv )
        print( std )
        print( "Using mean and std for target:" )
        print( meanTGT )
        print( stdTGT )

    # Load tracks and raycasts in
    x_data_train = []
    y_data_train = []
    x_data_val = []
    y_data_val = []
    totaldataset = []
    targetdataset = []
    for i in range( len( pathsTracksets ) ):
        if pathsTracksets[i] == "data/sleap_1_same3.h5":
            tracks = extract_coordinates( pathsTracksets[i], nodes, [x for x in range(nfish)] )[130:]
        else:
            tracks = extract_coordinates( pathsTracksets[i], nodes, [x for x in range(nfish)] )
        wRays = stealWallRays( pathsRaycasts[i], COUNT_RAYS_WALLS=N_WRAYS, nfish=nfish )
        nLoc = getnLoc( tracks, nnodes=nnodes, nfish=nfish )
        splitindex = int( tracks.shape[0] * SPLIT )
        for f in range( nfish ):
            fdataset = np.empty( ( tracks.shape[0], D_DATA ) )
            # View
            track_indices_ch = [f * nnodes * 2, f * nnodes * 2 + 1, f * nnodes * 2 + 2, f * nnodes * 2 + 3]
            track_iSubtract = [x + nnodes * 2 * f for x in range( nnodes * 2 )]
            track_indices_otherFish = [x for x in range( nnodes * nfish * 2 ) if x not in track_iSubtract]
            fnView = getnView( tracks[:,track_indices_ch], tracks[:,track_indices_otherFish], nfish = nfish )
            # RayCasts
            wRays_indices_fish = [f * N_WRAYS + x for x in range( N_WRAYS )]
            fwrays = wRays[:,wRays_indices_fish]
            # Locomotion
            nLoc_indices = [f * ( nnodes * 2 + 1 ) + x for x in range( nnodes * 2 + 1 )]
            fnLoc = nLoc[:,nLoc_indices]
            # Merge to dataset
            fdataset[:,:N_VIEWS] = fnView
            fdataset[:,N_VIEWS:-D_LOC] = fwrays
            fdataset[1:,-D_LOC:] = fnLoc
            fdataset[0,-D_LOC:] = fnLoc[0]
            # Target
            ftarget = np.empty( ( tracks.shape[0], D_LOC ) )
            ftarget[:-1] = fnLoc
            ftarget[-1] = fnLoc[-1]

            if mean is not None:
                fdataset = np.nan_to_num( ( fdataset - meanv ) / std )
                ftarget = np.nan_to_num( ( ftarget - meanTGT ) / stdTGT )

            x_train, y_train = multivariate_data( fdataset, ftarget, 0, splitindex, HIST_SIZE, TARGET_SIZE, 1, single_step=True )
            x_data_train.append( x_train )
            y_data_train.append( y_train )

            x_val, y_val = multivariate_data( fdataset, ftarget, splitindex, None, HIST_SIZE, TARGET_SIZE, 1, single_step=True )
            x_data_val.append( x_val )
            y_data_val.append( y_val )

            if getmean:
                totaldataset.append( np.array( fdataset[:splitindex] ) ) # for mean and std calculation
                targetdataset.append( np.array( ftarget[:splitindex] ) )

    if getmean:
        totaldataset = np.concatenate( totaldataset, axis=0 )
        meanv = totaldataset.mean( axis=0 )
        std = totaldataset.std( axis=0 )

        print( "mean   :" )
        print( meanv )
        print( "std    :" )
        print( std )

        targetdataset = np.concatenate( targetdataset, axis=0 )
        meanTGT = targetdataset.mean( axis=0 )
        stdTGT = targetdataset.std( axis=0 )

        print( "meanTGT:" )
        print( meanTGT )
        print( "stTGT  :" )
        print( stdTGT )

        np.save( pathToSave, np.array( [meanv, std, meanTGT, stdTGT] ) )

    x_data_train = np.concatenate( x_data_train, axis=0 )
    y_data_train = np.concatenate( y_data_train, axis=0 )
    x_data_val = np.concatenate( x_data_val, axis=0 )
    y_data_val = np.concatenate( y_data_val, axis=0 )

    return x_data_train, y_data_train, x_data_val, y_data_val


def simulate( model, nnodes, nfish, startinput, startpos, startloc, timesteps, N_VIEWS, D_LOC, N_WRAYS, FOV_WALLS, MAX_VIEW_RANGE, mean ):
    """
    returns positions of nodes, then polar position of center, then nLoc of predictions
    """
    assert len( startpos ) == nnodes * 2 * nfish
    assert startinput.shape[-1] == D_LOC + N_VIEWS + N_WRAYS
    assert nnodes >= 2
    nLoc = np.empty( ( timesteps, (nnodes * 2 + 1) * nfish ) )
    pos = np.empty( ( timesteps + 1, nnodes * 2 * nfish ) ) # saves x, y val for every node
    posCenterPolar = np.empty( (timesteps + 1, 3 * nfish ) ) # saves c_x, c_y, orienation
    # Initialize first row
    modelinput = startinput
    pos[0] = startpos
    for f in range(nfish):
        posCenterPolar[0, 3 * f] = startpos[nnodes * 2 * f + 2]
        posCenterPolar[0, 3 * f + 1] = startpos[nnodes * 2 * f + 3]
        # Angle between Fish Orientation and the unit vector
        # Head - Center
        x = ( startpos[nnodes * 2 * f] - startpos[nnodes * 2 * f + 2] )
        y = ( startpos[nnodes * 2 * f + 1] - startpos[nnodes * 2 * f + 3] )
        posCenterPolar[0, 3 * f + 2] = getAngle( ( 1, 0 ), ( x, y ), "radians" )

    # MARCS RAYCAST OBJECT
    walllines = defineLines( getRedPoints(path = "data/final_redpoint_wall.jpg") )
    unnecessary = 10
    raycast_object = Raycast( walllines, unnecessary, N_WRAYS, unnecessary, FOV_WALLS, MAX_VIEW_RANGE, nfish )

    # main loop
    for t in range( timesteps ):
        if t % 500 == 0:
            print( "Frame {:6}".format( t ) )
        for f in range( nfish ):
            # 1. Compute input for fish
            # 2. Compute prediction
            # 3. Compute new positions

            # 1. Input for fish
            inp = np.empty( (N_VIEWS + N_WRAYS + D_LOC) )

            # nView
            pos_indices_ch = [f * nnodes * 2, f * nnodes * 2 + 1, f * nnodes * 2 + 2, f * nnodes * 2 + 3]
            pos_iSubtract = [x + nnodes * 2 * f for x in range( nnodes * 2 )]
            pos_indices_otherFish = [x for x in range( nnodes * nfish * 2 ) if x not in pos_iSubtract]
            inp[:N_VIEWS] = getnViewSingle( pos[t,pos_indices_ch], pos[t,pos_indices_otherFish], nnodes=nnodes, nfish = nfish )

            # wRays
            pos_ind_center = [f * nnodes * 2 + 2, f * nnodes * 2 + 3]
            pos_ind_head = [f * nnodes * 2, f * nnodes * 2 + 1]
            fcenter = pos[t,pos_ind_center]
            fhead = pos[t,pos_ind_head]
            vec_ch = fhead - fcenter
            inp[N_VIEWS:-D_LOC] = raycast_object._getWallRays( fcenter, vec_ch )

            # nLoc
            loc_ind = [f * D_LOC + x for x in range(D_LOC)]
            if t == 0:
                inp[-D_LOC:] = np.array( startloc )[loc_ind]
            else:
                inp[-D_LOC:] = nLoc[t - 1, loc_ind]

            # 2. Prediction
            # Shift all observations
            modelinput[:-1] = modelinput[1:]
            # Insert newest
            modelinput[-1] = inp
            prediction = model.predict( np.array( [modelinput] ) )
            # prediction = model[t,loc_ind] # to test correcness of simulation insert a loc file as model
            nLoc[t, loc_ind] = prediction

            # 3. New positions

            # 3.1 new center position
            loc_ind_linAngTurn = [f * D_LOC, f * D_LOC + 1, f * D_LOC + 2]
            pos_ind_xyOri = [f * 3, f * 3 + 1, f * 3 + 2]
            posCenterPolar[t + 1, pos_ind_xyOri] = row_l2c( posCenterPolar[t, pos_ind_xyOri], nLoc[t, loc_ind_linAngTurn] )

            # 3.2 all the other positions
            pred_ind_otherNodes = [ 3 + x for x in range( (nnodes - 1 ) * 2 ) ]
            pos[t + 1, pos_ind_center] = posCenterPolar[t + 1,[f * 3, f * 3 + 1]]
            output = row_l2c_additional_nodes( posCenterPolar[t + 1,pos_ind_xyOri], prediction[0][pred_ind_otherNodes] )

            pos[t + 1, pos_ind_head] = output[0:2]
            pos_ind_otherNodes = [nnodes * 2 * f + 4 + x for x in range( ( nnodes - 2 ) * 2 )]
            pos[t + 1, pos_ind_otherNodes] = output[2:]

    return pos, posCenterPolar, nLoc


def getDatasets( x_train, y_train, x_val, y_val, BATCH_SIZE, BUFFER_SIZE ):
    """
    Train the network
    """
    # Put data into datasets
    train_data = tf.data.Dataset.from_tensor_slices( ( x_train, y_train ) )
    train_data = train_data.cache().shuffle( BUFFER_SIZE ).batch( BATCH_SIZE ).repeat()

    val_data = tf.data.Dataset.from_tensor_slices( ( x_val, y_val ) )
    val_data = val_data.batch( BATCH_SIZE ).repeat()

    return train_data, val_data


def saveModel( path, model ):
    """
    saves model
    """
    # handle dir
    if not os.path.isdir( path ):
        # create dir
        try:
            os.mkdir( path )
        except OSError:
            print("Dir Creation failed")
    if path[-1] != "/":
        path = path + "/"

    return model.save( path )


def loadStartData( path, pathRaycast, nodes, nfish, HIST_SIZE,TARGET_SIZE, N_WRAYS, D_DATA, N_VIEWS, D_LOC, mean, notrandom=None ):
    """
    Loads random startpoint from path diffset
    """

    if mean is not None:
        arr = np.load( mean, allow_pickle=True )
        meanv = arr[0]
        std = arr[1]
        meanTGT = arr[2]
        stdTGT = arr[3]
        print( "Using mean and std:")
        print( meanv )
        print( std )
        print( "Using mean and std for target:" )
        print( meanTGT )
        print( stdTGT )

    nnodes = len( nodes )
    x_data_train = []
    tracks = extract_coordinates( path, nodes, [x for x in range(nfish)] )
    nLoc = getnLoc( tracks, nnodes=nnodes, nfish=nfish )
    wRays = stealWallRays( pathRaycast, COUNT_RAYS_WALLS=N_WRAYS, nfish=nfish )

    for f in range( nfish ):
        fdataset = np.empty( ( tracks.shape[0], D_DATA ) )
        # View
        track_indices_ch = [f * nnodes * 2, f * nnodes * 2 + 1, f * nnodes * 2 + 2, f * nnodes * 2 + 3]
        track_iSubtract = [x + nnodes * 2 * f for x in range( nnodes * 2 )]
        track_indices_otherFish = [x for x in range( nnodes * nfish * 2 ) if x not in track_iSubtract]
        fnView = getnView( tracks[:,track_indices_ch], tracks[:,track_indices_otherFish], nfish = nfish )
        # RayCasts
        wRays_indices_fish = [f * N_WRAYS + x for x in range( N_WRAYS )]
        fwrays = wRays[:,wRays_indices_fish]
        # Locomotion
        nLoc_indices = [f * ( nnodes * 2 + 1 ) + x for x in range( nnodes * 2 + 1 )]
        fnLoc = nLoc[:,nLoc_indices]
        # Merge to dataset
        fdataset[:,:N_VIEWS] = fnView
        fdataset[:,N_VIEWS:-D_LOC] = fwrays
        fdataset[1:,-D_LOC:] = fnLoc
        fdataset[0,-D_LOC:] = fnLoc[0]
        # Target
        ftarget = np.empty( ( tracks.shape[0], D_LOC ) )
        ftarget[:-1] = fnLoc
        ftarget[-1] = fnLoc[-1]

        if mean is not None:
            fdataset = np.nan_to_num( ( fdataset - meanv ) / std )
            ftarget = np.nan_to_num( ( ftarget - meanTGT ) / stdTGT )

        x_train, y_train = multivariate_data( fdataset, ftarget, 0, None, HIST_SIZE, TARGET_SIZE, 1, single_step=True )
        x_data_train.append( x_train )

    x_data_train = np.concatenate( x_data_train, axis=0 )
    if notrandom is None:
        thatoneisit = np.random.randint( 0,x_data_train.shape[-1] )
        return x_data_train[thatoneisit], tracks[thatoneisit * HIST_SIZE], nLoc[thatoneisit * HIST_SIZE]

    return x_data_train[notrandom], tracks[notrandom * HIST_SIZE], nLoc[notrandom * HIST_SIZE]


def main():
    """
    """
    # Parameters
    SPLIT = 0.9
    BATCH_SIZE = 10
    BUFFER_SIZE= 10000
    EPOCHS = 20
    HIST_SIZE = 70 # frames to be looked on or SEQ_LEN
    TARGET_SIZE = 0
    N_NODES = 4
    N_FISH = 3
    N_WRAYS = 15
    N_VIEWS = (N_FISH - 1 ) * N_NODES * 2
    D_LOC = N_NODES * 2 + 1
    D_DATA = N_VIEWS + N_WRAYS + D_LOC
    D_OUT = D_LOC
    U_LSTM = 128
    U_DENSE = 64
    U_OUT = D_LOC
    FOV_WALLS = 180
    MAX_VIEW_RANGE = 709
    STARTSEQ = 0
    SIM_STEPS = 3000
    MEAN = "data/mean_same1234_node4.npy"
    NAME = "4model_v8"
    NAME = NAME + "_" + str( U_LSTM ) + "_" + str( U_DENSE ) + "_" + str( U_OUT ) + "_" + str( BATCH_SIZE ) + "_" + str( HIST_SIZE )
    LOAD = "4model_v6_40_20_9_10_70"

    tf.random.set_seed(13)

    same1 = "data/sleap_1_same1.h5"
    same3 = "data/sleap_1_same3.h5"
    same4 = "data/sleap_1_same4.h5"
    same5 = "data/sleap_1_same5.h5"
    same1rays = "data/raycast_data_same1.csv"
    same3rays = "data/raycast_data_same3.csv"
    same4rays = "data/raycast_data_same4.csv"
    same5rays = "data/raycast_data_same5.csv"

    diff1 = "data/sleap_1_diff1.h5"
    diff1rays = "data/raycast_data_diff1.csv"

    if False:
        pathsTracksets = [same1,same3,same4,same5]
        pathsRaycasts = [same1rays,same3rays,same4rays,same5rays]

        x_train, y_train, x_val, y_val = loadData( pathsTracksets, pathsRaycasts, nodes=[b'head', b'center', b'tail_basis', b'tail_end'], nfish=3, N_WRAYS=N_WRAYS, N_VIEWS=N_VIEWS, D_LOC=D_LOC, D_DATA=D_DATA, D_OUT=D_OUT, SPLIT=SPLIT, HIST_SIZE=HIST_SIZE, TARGET_SIZE=TARGET_SIZE, mean=MEAN )
        print( "x_train: {}".format( x_train.shape ) )
        print( "y_train: {}".format( y_train.shape ) )
        print( "x_val  : {}".format( x_val.shape ) )
        print( "y_val  : {}".format( y_val.shape ) )

        traindata, valdata = getDatasets( x_train, y_train, x_val, y_val, BATCH_SIZE=BATCH_SIZE, BUFFER_SIZE=BUFFER_SIZE )

        nmodel = createModel( NAME, U_LSTM, U_DENSE, U_OUT, x_train.shape[-2:], dropout=[0.3,0.3] )

        EVAL_INTERVAL = ( x_train.shape[0] ) // BATCH_SIZE
        VAL_INTERVAL = ( x_val.shape[0] ) // BATCH_SIZE

        history = nmodel.fit( traindata, epochs=EPOCHS, steps_per_epoch=EVAL_INTERVAL, validation_data=valdata, validation_steps=VAL_INTERVAL )

        saveModel( NAME, nmodel )
        plot_train_history( history, NAME )
    else:
        # nmodel = tf.keras.models.load_model( LOAD )
        # tracks = extract_coordinates( same1, [b'head', b'center', b'tail_basis', b'tail_end'] )
        # nLoc = getnLoc( tracks, 4, 3 )
        # print( getnLoc( tracks, nnodes=N_NODES, nfish=N_FISH )[100] )
        pathsTracksets = [same1]
        pathsRaycasts = [same1rays]

        x_train, y_train, x_val, y_val = loadData( pathsTracksets, pathsRaycasts, nodes=[b'head', b'center', b'tail_basis', b'tail_end'], nfish=3, N_WRAYS=N_WRAYS, N_VIEWS=N_VIEWS, D_LOC=D_LOC, D_DATA=D_DATA, D_OUT=D_OUT, SPLIT=SPLIT, HIST_SIZE=HIST_SIZE, TARGET_SIZE=TARGET_SIZE, mean=MEAN )
        print( "x_train: {}".format( x_train.shape ) )
        print( "y_train: {}".format( y_train.shape ) )
        print( "x_val  : {}".format( x_val.shape ) )
        print( "y_val  : {}".format( y_val.shape ) )

        traindata, valdata = getDatasets( x_train, y_train, x_val, y_val, BATCH_SIZE=BATCH_SIZE, BUFFER_SIZE=BUFFER_SIZE )
        # for x, y in traindata.take(1):
        #     predictun = nmodel.predict( x )
        #     print( "target" )
        #     print( y[0] )

        # print( "prediction" )
        # print( predictun[0] )

        startinput, startpos, startloc = loadStartData( diff1, diff1rays, [b'head', b'center', b'tail_basis', b'tail_end'], 3, notrandom=STARTSEQ, HIST_SIZE=HIST_SIZE, TARGET_SIZE=TARGET_SIZE, N_WRAYS=N_WRAYS, D_DATA=D_DATA, N_VIEWS=N_VIEWS, D_LOC=D_LOC, MEAN=MEAN )

        pos, posC, nLocs = simulate( model=nmodel, nnodes=N_NODES, nfish=N_FISH, startinput=startinput, startpos=startpos, startloc=startloc, timesteps=SIM_STEPS, N_VIEWS=N_VIEWS, N_WRAYS=N_WRAYS, D_LOC=D_LOC, FOV_WALLS=FOV_WALLS, MAX_VIEW_RANGE=MAX_VIEW_RANGE, mean=MEAN )

        df = pd.DataFrame( data = pos )
        df.to_csv( LOAD + "tracks.csv", sep = ";" )



if __name__ == "__main__":
    main()
