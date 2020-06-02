import numpy as np
import pandas as pd
import math
from functions import getAngle, getDistance
from itertools import chain

def getLocomotion(np_array, path_to_save_to):
    """
    This function expects to be given a numpy array of the shape (rows, count_fishes*4) and saves a csv file at a given path (path has to end on .csv).
    The information about each given fish (or object in general) should be first_position_x, first_position_y, second_position_x, second_position_y.
    It is assumed that the fish is looking into the direction of first_positon_x - second_position_x for x and first_positon_y - second_position_y for y.
    """
    output = np.array([list(chain.from_iterable(("Fish_" + str(i) + "_linear_movement", "Fish_" + str(i) + "_angle_radians") for i in range(0, int(np_array.shape[1]/4))))])

    for i in range(0, np_array.shape[0]-1):
        if i%1000 == 0:
            print("||| Frame " + str(i) + " finished. |||")
        new_row = [0 for k in range(0, int(2*np_array.shape[1]/4))]
        for j in range(0, int(np_array.shape[1]/4)):
            head_x = np_array[i, j*4]
            head_y = np_array[i, j*4+1]
            center_x = np_array[i, j*4+2]
            center_y = np_array[i, j*4+3]

            head_x_next = np_array[i+1, j*4]
            head_y_next = np_array[i+1, j*4+1]
            center_x_next = np_array[i+1, j*4+2]
            center_y_next = np_array[i+1, j*4+3]

            #look vector
            vector = (head_x - center_x, head_y - center_y)
            #vector to new position
            vector_next = (head_x - head_x_next, head_y - head_y_next)

            new_row[j*2+1] = getAngle(vector, vector_next, mode = "radians")
            temp = getDistance(head_x, head_y, head_x_next, head_y_next)
            #its forward movement if it's new position is not at the back of the fish and otherwise it is backward movement
            new_row[j*2] = temp if new_row[j*2+1] > 90 and new_row[j*2+1] < 270 else -temp
        output = np.append(output, [new_row], axis = 0)

    df = pd.DataFrame(data = output[1:], columns = output[0])
    df.to_csv(path_to_save_to, index = None, sep = ";")
