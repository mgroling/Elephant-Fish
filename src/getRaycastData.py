from raycasts import *
from reader import *
import pandas as pd
import numpy as np
import os

def main():
    #for marc
    os.chdir("SWP/Elephant-Fish")

    #Set variables
    COUNT_BINS_AGENTS = 21
    WALL_RAYS_WALLS = 15
    RADIUS_FIELD_OF_VIEW_WALLS = 180
    RADIUS_FIELD_OF_VIEW_AGENTS = 300
    MAX_VIEW_RANGE = 600
    COUNT_FISHES = 3

    #Extract Raycasts
    our_wall_lines = defineLines(getRedPoints(path = "data/final_redpoint_wall.jpg"))
    ray = Raycast(our_wall_lines, COUNT_BINS_AGENTS, WALL_RAYS_WALLS, RADIUS_FIELD_OF_VIEW_AGENTS, RADIUS_FIELD_OF_VIEW_WALLS, MAX_VIEW_RANGE, COUNT_FISHES)

    file = "data/sleap_1_Diffgroup1-1.h5"

    temp = extract_coordinates(file, [b'head', b'center'], fish_to_extract=[0,1,2])

    #remove rows with Nans in it
    temp = temp[~np.isnan(temp).any(axis=1)]

    print("shape:",temp.shape)

    #get rays and save them
    ray.getRays(temp , "data/raycast_data.csv")

if __name__ == "__main__":
    main()