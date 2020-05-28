import pandas as pd

def read_slp(filepath):
    """
    reads a sleap file and converts it into appropiate format
    """
    raw_data = pd.read_hdf(filepath)
    print(type(raw_data))

    return raw_data