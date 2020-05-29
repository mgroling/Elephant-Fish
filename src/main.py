from raycasts import *
import reader
import tensorflow as tf
import pandas as pd
import numpy as np


# Use python 3.6.10

def main():
    # Set variables
    COUNT_BINS_AGENTS = 6
    WALL_RAYS_WALLS = 3
    RADIUS_FIELD_OF_VIEW = 120
    MAX_VIEW_RANGE = 1000
    COUNT_FISHES = 3

    # Read data
    track_data = reader.extract_coordinates("data/MARC_USE_THIS_DATA.h5", [b'head', b'center'], fish_to_extract = [0,1,2])

    # Extract raycasts
    our_wall_lines = [(580-elem[0], 582-elem[1], 580-elem[2], 582-elem[3]) for elem in defineLines(getRedPoints())]
    ray = Raycast(our_wall_lines, COUNT_BINS_AGENTS, WALL_RAYS_WALLS, RADIUS_FIELD_OF_VIEW, MAX_VIEW_RANGE, COUNT_FISHES)

    # Tensorflow magic

    # Wir brauchen noch eine Funktion die zur runtime raycasts bestimmt später für die analyse

if __name__ == "__main__":
    main()