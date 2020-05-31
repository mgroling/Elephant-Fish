import pandas as pd
import numpy as np
import os
from reader import *
from locomotion import getLocomotion

def main():
    #for marc
    os.chdir("SWP/Elephant-Fish")
 
    file = "data/sleap_1_Diffgroup1-1.h5"

    output = extract_coordinates(file, [b'head', b'center'], fish_to_extract=[0,1,2])

    #reshape output into shape in a way that it can be fed to getLocomotion
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

    #get locomotion and save it
    getLocomotion(temp , "data/locomotion_data.csv")

if __name__ == "__main__":
    main()