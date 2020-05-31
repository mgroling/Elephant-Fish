from raycasts import *
import reader
import tensorflow as tf
import pandas as pd
import numpy as np
import os

# Use python 3.6.10

def main():
    #for marc
    os.chdir("SWP/Elephant-Fish")

    # Set variables
    COUNT_BINS_AGENTS = 21
    WALL_RAYS_WALLS = 15
    RADIUS_FIELD_OF_VIEW_WALLS = 180
    RADIUS_FIELD_OF_VIEW_AGENTS = 300
    MAX_VIEW_RANGE = 600
    COUNT_FISHES = 3

    # Read data
    track_data = reader.extract_coordinates("data/MARC_USE_THIS_DATA.h5", [b'head', b'center'], fish_to_extract = [0,1,2])

    #get raycast data (input)
    df = pd.read_csv("/data/raycast_data.csv")
    input_raw = df.to_numpy()

    #get locomotion data (input/output)

    # Tensorflow magic

    # Wir brauchen noch eine Funktion die zur runtime raycasts bestimmt später für die analyse

if __name__ == "__main__":
    main()