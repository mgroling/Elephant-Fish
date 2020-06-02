import pandas as pd
import numpy as np
import os
from reader import *
from locomotion import getLocomotion

def main():

    file = "data/sleap_1_Diffgroup1-1.h5"

    temp = extract_coordinates(file, [b'head', b'center'], fish_to_extract=[0,1,2])

    #remove rows with Nans in it
    temp = temp[~np.isnan(temp).any(axis=1)]

    print("shape:",temp.shape)

    #get locomotion and save it
    getLocomotion(temp , "data/locomotion_data.csv")

if __name__ == "__main__":
    main()