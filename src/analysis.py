# File to analyse the results
# Copied from Moritz Maxeiner Masterthesis: https://git.imp.fu-berlin.de/bioroboticslab/robofish/ai

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn
import pandas as pd
import numpy as np

import reader
from functions import *

def normalize_series(x):
    """
    Given a series of vectors, return a series of normalized vectors.
    Null vectors are mapped to `NaN` vectors.
    """
    return (x.T / np.linalg.norm(x, axis=-1)).T


def calc_iid(a, b):
    """
    Given two series of poses - with X and Y coordinates of their positions as the first two elements -
    return the inter-individual distance (between the positions).
    """
    return np.linalg.norm(b[:, :2] - a[:, :2], axis=-1)


def calc_tlvc(a, b, tau_min, tau_max):
    """
    Given two velocity series and both minimum and maximum time lag return the
    time lagged velocity correlation from the first to the second series.
    """
    length = tau_max - tau_min
    return np.float32(
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
    return (a_v * b_p).sum(axis=-1)


def plot_follow(tracks, max_tolerated_movement=20):
    """
    Create and save Follow graph, only use center nodes for it
    count_bins is the number bins
    """
    assert tracks.shape[-1] % 2 == 0
    nfish = int(tracks.shape[-1] / 2)

    follow = []
    # for every fish combination calculate the follow
    for i1 in range(nfish):
        for i2 in range(i1 + 1, nfish):
            f1_x, f1_y = get_indices(i1)
            f2_x, f2_y = get_indices(i2)
            follow.append(calc_follow(tracks[:, f1_x:f1_y + 1], tracks[:, f2_x:f2_y + 1]))
            follow.append(calc_follow(tracks[:, f2_x:f2_y + 1], tracks[:, f1_x:f1_y + 1]))

    follow = np.concatenate(follow, axis=0)

    # Create bins for bars
    bins_pos = [x + 0.5 for x in range(0, max_tolerated_movement)] + [max_tolerated_movement]
    bins_neg = [x * -1 for x in reversed(bins_pos)]
    bins = bins_neg + bins_pos
    # Create labels
    labels = list(range(-max_tolerated_movement, max_tolerated_movement + 1))
    valsToPlot = np.histogram(follow, bins=bins)[0]
    # Set up plot
    y_pos = np.arange(len(valsToPlot))

    plt.bar(y_pos, valsToPlot)
    plt.xticks(y_pos, labels)

    plt.plot()
    return plt.gcf()


def plot_follow_iid(tracks):
    """
    plots fancy graph with follow and iid, only use with center values
    copied from Moritz Maxeiner
    """

    assert tracks.shape[-1] % 2 == 0
    nfish = int(tracks.shape[-1] / 2)

    follow = []
    iid = []

    # for every fish combination calculate the follow
    for i1 in range(nfish):
        for i2 in range(i1 + 1, nfish):
            f1_x, f1_y = get_indices(i1)
            f2_x, f2_y = get_indices(i2)
            iid.append(calc_iid(tracks[:-1, f1_x:f1_y + 1], tracks[:-1, f2_x:f2_y + 1]))
            iid.append(iid[-1])

            follow.append(calc_follow(tracks[:, f1_x:f1_y + 1], tracks[:, f2_x:f2_y + 1]))
            follow.append(calc_follow(tracks[:, f2_x:f2_y + 1], tracks[:, f1_x:f1_y + 1]))

    follow_iid_data = pd.DataFrame(
        {"IID [cm]": np.concatenate(iid, axis=0), "Follow": np.concatenate(follow, axis=0)}
    )

    grid = seaborn.jointplot(
        x="IID [cm]", y="Follow", data=follow_iid_data, linewidth=0, s=1, kind="scatter"
    )
    grid.ax_joint.set_xlim(0, 142)
    grid.fig.set_figwidth(9)
    grid.fig.set_figheight(6)
    grid.fig.subplots_adjust(top=0.9)
    return grid.fig


def plot_locomotion(paths, path_to_save, round):
    """
    paths should be iterable
    path_to_save is the folder in which it will be saved
    round will round to one decimal place (round = 2 -> all values will be of the form x.yz)
    Creates Bar Plots for the Dataframes, one for linear_movement and for angle_radians
    """
    df_list = []
    for path in paths:
        df_list.append(pd.read_csv(path, sep = ";"))
    df_all = pd.concat(df_list)
    #round values
    df_all = df_all.round(round)

    #get all linear_movement columns respectivly all angle_radians columns
    mov_cols = [col for col in df_all.columns if "linear_movement" in col]
    ang_cols = [col for col in df_all.columns if "angle_radians" in col]

    #melt linear_movement columns together in one
    df_mov = df_all[mov_cols].melt(var_name = "columns", value_name = "value")
    #count the values and plot it
    ax = df_mov["value"].value_counts().sort_index().plot.bar()
    for i, t in enumerate(ax.get_xticklabels()):
        if (i % 2) != 0:
            t.set_visible(False)
    fig = ax.get_figure()
    fig.set_size_inches(25, 12.5)
    fig.savefig(path_to_save + "plot_linear_movement.png")

    df_ang = df_all[ang_cols].melt(var_name = "columns", value_name = "value")
    ax = df_ang["value"].value_counts().sort_index().plot.bar()
    fig = ax.get_figure()
    fig.set_size_inches(25, 12.5)
    fig.savefig(path_to_save + "plot_angle_radians.png")


def save_figure(fig, path = "figures/latest_plot.png", size = (25, 12.5)):
    """
    Saves the given figure in path with certain size
    """
    x, y = size
    fig.set_size_inches(x, y)
    fig.savefig(path)
    plt.close(fig)


def plot_positions(track, track2 = None):

    frames, positions = track.shape

    assert positions % 2 == 0

    i_x = list(range(0,positions,2))
    i_y = list(range(1,positions,2))

    fig = plt.figure()

    def update_points(n, track, points):
        points.set_xdata(track[n,i_x])
        points.set_ydata(track[n,i_y])

    def update_points2(n, track, track2, points, points2):
        points.set_xdata(track[n,i_x])
        points.set_ydata(track[n,i_y])
        points2.set_xdata(track2[n,i_x])
        points2.set_ydata(track2[n,i_y])

    plt.xlim(0, 960)
    plt.ylim(9,720)

    points, = plt.plot([], [], 'r.')
    if track2 is None:
        point_animation = animation.FuncAnimation(fig, update_points, track.shape[0],fargs=(track, points), interval=10)
    else:
        points2, = plt.plot([], [], 'b.')
        point_animation = animation.FuncAnimation(fig, update_points2, track.shape[0],fargs=(track, track2, points, points2), interval=10)

    plt.show()


def main():
    file = "data/sleap_1_Diffgroup1-1.h5"
    # fish1 = reader.extract_coordinates(file, [b'center'], [0])
    # fish2 = reader.extract_coordinates(file, [b'center'], [1])
    # tracks = reader.extract_coordinates(file, [b'head', b'center', b'l_fin_basis', b'r_fin_basis', b'l_fin_end', b'r_fin_end', b'l_body', b'r_body', b'tail_basis', b'tail_end'], [0,1,2])
    # tracks2 = reader.extract_coordinates(file, [b'head', b'center', b'l_fin_basis', b'r_fin_basis', b'l_fin_end', b'r_fin_end', b'l_body', b'r_body', b'tail_basis', b'tail_end'], [0,1,2], interpolate_nans=False, interpolate_outlier=False)
    tracks = reader.extract_coordinates(file, [b'center'])

    fig = plot_follow_iid(tracks)
    save_figure(fig, "figures/follow_iid.png")
    # plot_positions(tracks, tracks2)


if __name__ == "__main__":
    main()
