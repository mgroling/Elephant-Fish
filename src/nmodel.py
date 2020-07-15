# Python file for the nmodel
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from functions import getDistances, getAngles
from reader import extract_coordinates
from locomotion import getnLoc
from evaluation import plot_train_history

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


def stealWallRays( path, COUNT_RAYS_WALLS=15, nfish=3 ):
    """
    Steals wall rays from raycast data
    """
    raycasts = pd.read_csv( path, sep = ";" ).to_numpy()
    return raycasts[:,-15 * nfish:]


def createNetwork( BATCH_SIZE=10, SEQ_LEN=75, COUNT_RAYS_WALLS=15, N_FISH=3, N_NODES=4 ):
    """
    Creates and returns an RNN model
    """
    N_VIEWS =  (N_FISH - 1 ) * N_NODES * 2
    N_LOC = N_NODES * 2 + 1
    dimdata = N_VIEWS + COUNT_RAYS_WALLS + N_LOC
    n_outputlayer = N_NODES * 20 + 20
    # (batch,)
    ishape = ( BATCH_SIZE, SEQ_LEN, dimdata )

    model = keras.Sequential()
    model.add( layers.InputLayer( batch_input_shape=ishape ) )
    model.add( layers.LSTM(dimdata) )
    # model.add( layers.Dense( 100 ) )
    model.add( layers.Dense( n_outputlayer ) )
    model.summary()
    model.name = "4model_norm_v0"
    return model


def trainNetwork( network, nLoc, nView, wallRays, BATCH_SIZE=10, SEQ_LEN=75, COUNT_RAYS_WALLS=15, N_FISH=3, N_NODES=4 ):
    pass


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


def getData( TRAINSPLIT ):
    """
    Full Manual Frankencode now
    """
    tracks = extract_coordinates( "data/sleap_1_diff1.h5", [b'head', b'center', b'tail_basis', b'tail_end'] )
    nLoc = getnLoc( tracks, nnodes=4, nfish=3 )
    nView = getnView( tracks[:,0:4], tracks[:,8:], nfish=3 )
    wallrays = stealWallRays( "data/raycast_data_diff1.csv", COUNT_RAYS_WALLS=15, nfish=3 )[:,0:15]
    rowsLoc, colsLoc = nLoc.shape
    f1Loc = colsLoc // 3
    data = np.empty( ( rowsLoc + 1, f1Loc + nView.shape[-1] + wallrays.shape[-1] ) )
    data[:,0:nView.shape[-1]] = nView
    data[:,nView.shape[-1]:-f1Loc] = wallrays
    # duplicate loc at beginning ( i know i know ... )
    data[0:,-f1Loc:] = nLoc[0,:f1Loc]
    data[1:,-f1Loc:] = nLoc[:,:f1Loc]
    data = data[:-1]
    target = nLoc[:,:f1Loc]

    # Standardize data
    data_mean = data[:TRAINSPLIT].mean( axis=0 )
    data_std = data[:TRAINSPLIT].std( axis=0 )
    print( data[0] )
    data = ( data - data_mean ) / data_std
    target = ( target - data_mean[-f1Loc:] ) / data_std[-f1Loc:]
    return np.nan_to_num( data ) , np.nan_to_num( target ), data_mean[-f1Loc:], data_std[-f1Loc:]


def main():
    TRAINSPLIT = 15000
    BATCH_SIZE = 10
    BUFFER_SIZE= 10000
    HIST_SIZE = 70 # frames to be looked on or SEQ_LEN
    TARGET_SIZE = 0
    N_NODES = 4
    N_FISH = 3
    N_WRAYS = 15
    N_VIEWS =  (N_FISH - 1 ) * N_NODES * 2
    D_LOC = N_NODES * 2 + 1
    D_DATA = N_VIEWS + N_WRAYS + D_LOC
    D_OUT = D_LOC
    EPOCHS = 200

    data, target, data_mean, data_std = getData( TRAINSPLIT )
    assert D_OUT == target.shape[-1]
    # print( np.amax( data, axis=0 ) )
    # print( np.amin( data, axis=0 ) )
    # print( np.mean( data, axis=0 ) )

    x_train, y_train = multivariate_data( data, target, 0, TRAINSPLIT, HIST_SIZE, TARGET_SIZE, 1, single_step=True )
    x_val, y_val = multivariate_data( data, target, TRAINSPLIT, None, HIST_SIZE, TARGET_SIZE, 1, single_step=True )
    print( "Single entry shape: {}".format( x_train[0].shape ) )

    train_data = tf.data.Dataset.from_tensor_slices( ( x_train, y_train ) )
    train_data = train_data.cache().shuffle( BUFFER_SIZE ).batch( BATCH_SIZE ).repeat()

    val_data = tf.data.Dataset.from_tensor_slices( ( x_val, y_val ) )
    val_data = val_data.batch( BATCH_SIZE ).repeat()

    nmodel = tf.keras.models.Sequential()
    print( "prediction: ({}, {})".format( HIST_SIZE, D_DATA ) )
    print( "input shape: {}".format( x_train.shape[-2:] ) )
    nmodel.add( tf.keras.layers.LSTM( D_DATA * 4, input_shape=x_train.shape[-2:] ) )
    nmodel.add( tf.keras.layers.Dense( 100 ) )
    nmodel.add( tf.keras.layers.Dense( D_OUT ) )

    nmodel.compile( optimizer=tf.keras.optimizers.RMSprop(), loss='mean_squared_error' )

    EVALUATION_INTERVAL = len( x_train ) // BATCH_SIZE
    VAL_INTERVAL = len( x_val ) // BATCH_SIZE
    print( "Eval Interval: ", EVALUATION_INTERVAL )
    print( "Val interval:", VAL_INTERVAL )
    history = nmodel.fit( train_data, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_data, validation_steps=VAL_INTERVAL )

    # model = keras.models.load_model('path/to/location')

    for x, y in val_data.take(1):
        p = nmodel.predict(x)
        print( p )
        print( p.shape )
        print( "target" )
        print( y )
        print( y.shape )
        print( "unnormalized" )
        un = p * data_std + data_mean
        uy = y * data_std + data_mean
        print( un )
        print( un.p )
        print( "target" )
        print( uy )
        print( uy.shape )

    nmodel.save( "models/" )
    plot_train_history( history, "4model_normalized_v0" )

if __name__ == "__main__":
    main()