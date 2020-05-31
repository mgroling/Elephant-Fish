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

    file = "data/MARC_USE_THIS_DATA.h5"

    output = extract_coordinates(file, [b'head', b'center'], fish_to_extract=[0,1,2])

    #reshape output into shape in a way that it can be fed to getRays
    #new shape is (count_frames, (fish1_pos1_x, fish1_pos1_y, fish1_pos2_x, fish1_pos2_y, fish2_pos1_x, fish2_pos1_y, fish2_pos2_x, fish2_pos2_y, fish3_pos1_x, fish3_pos1_y, fish3_pos2_x, fish3_pos2_y))
    temp = output.reshape(-1, output.shape[-1])
    temp = np.transpose(temp)

    permutation = [0,2,1,3,4,6,5,7,8,10,9,11]
    new = np.empty_like(permutation)
    new[permutation] = np.arange(len(permutation))
    temp = temp[:, new]

    #remove rows with Nans in it
    temp = temp[~np.isnan(temp).any(axis=1)]

    print("shape:",temp.shape)

    #get rays and save them
    ray.getRays(temp , "data/raycast_data.csv")

if __name__ == "__main__":
    main()