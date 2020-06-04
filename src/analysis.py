# File to analyse the results
# Copied from Moritz Maxeiner Masterthesis: https://git.imp.fu-berlin.de/bioroboticslab/robofish/ai

from numpy import float32
from numpy.linalg import norm
import reader
import matplotlib.pyplot as plt
from functions import *


def normalize_series(x):
    """
    Given a series of vectors, return a series of normalized vectors.
    Null vectors are mapped to `NaN` vectors.
    """
    return (x.T / norm(x, axis=-1)).T


def calc_iid(a, b):
    """
    Given two series of poses - with X and Y coordinates of their positions as the first two elements -
    return the inter-individual distance (between the positions).
    """
    return norm(b[:, :2] - a[:, :2], axis=-1)


def calc_tlvc(a, b, tau_min, tau_max):
    """
    Given two velocity series and both minimum and maximum time lag return the
    time lagged velocity correlation from the first to the second series.
    """
    length = tau_max - tau_min
    return float32(
        [
            (a[t] @ b[t + tau_min :][:length].T).mean()
            for t in range(min(len(a), len(b) - tau_max + 1))
        ]
    )


def calc_follow(a, b):
    """
    input: a,b 2d array with x and y values
    output: follow metric as defined by Moritz Maxeiner
    """
    a_v = a[1:, :2] - a[:-1, :2]
    b_p = normalize_series(b[:-1, :2] - a[:-1, :2])
    print(b_p)
    return (a_v * b_p).sum(axis=-1)


def plot_follow(tracks, file = "data/follow.png", max_tolerated_movement=20, count_bins=21):
    """
    Create and save Follow graph, only use center nodes for it
    count_bins is the number bins
    """
    assert tracks.shape[-1] % 2 == 0
    nfish = int(tracks.shape[-1] / 2)

    follow = []
    # for every fish combination
    for i1 in range(nfish):
        for i2 in range(i1 + 1, nfish):
            f1_x, f1_y = get_indices(i1)
            f2_x, f2_y = get_indices(i2)
            follow.append(calc_follow(tracks[:, f1_x:f1_y + 1], tracks[:, f2_x:f2_y + 1]))

    follow = np.concatenate(follow, axis=0)

    step = max_tolerated_movement/count_bins
    bins = [x*step - max_tolerated_movement/2 for x in range(count_bins)] + [max_tolerated_movement/2]
    bin_labels = np.round([ x - max_tolerated_movement/2 + step/2 for x in range(count_bins - 1)], 2)
    valsToPlot = np.histogram(follow, bins=bins)[0]

    # Plot the thing
    y_pos = np.arange(len(valsToPlot))
    plt.bar(y_pos, valsToPlot)
    # correct labels
    plt.xticks(y_pos, bin_labels)
    plt.show()


def main():
    file = "data/sleap_1_Diffgroup1-1.h5"
    fish1 = reader.extract_coordinates(file, [b'center'], [0])
    fish2 = reader.extract_coordinates(file, [b'center'], [1])
    tracks = reader.extract_coordinates(file, [b'center'], [0,1,2])

    plot_follow(tracks)


if __name__ == "__main__":
    main()
