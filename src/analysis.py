# File to analyse the results
# Copied from Moritz Maxeiner Masterthesis: https://git.imp.fu-berlin.de/bioroboticslab/robofish/ai

import reader
import matplotlib.pyplot as plt
from functions import *
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


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


def plot_follow(tracks, file = "data/follow.png", max_tolerated_movement=20):
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
    print(y_pos)
    plt.bar(y_pos, valsToPlot)
    plt.xticks(y_pos, labels)
    plt.show()


def plot_locomotion(paths, path_to_save, round):
    """
    paths should be iterable
    path_to_save is the folder in which it will be saved
    round will round to one decimal place (round = 2 -> all values will be of the form x.yz)
    Creates Bar Plots for the Dataframes, one for linear_movement and for angle_radians
    """
    #right now it thinks that 0 and 2*pi are not the same for pos and ori, maybe find a solution for it
    df_list = []
    for path in paths:
        df_list.append(pd.read_csv(path, sep = ";"))
    df_all = pd.concat(df_list)
    #round values
    df_all = df_all.round(round)

    #get all linear_movement columns respectivly all angle columns
    mov_cols = [col for col in df_all.columns if "linear_movement" in col]
    ang_pos_cols = [col for col in df_all.columns if "angle_new_pos" in col]
    ang_ori_cols = [col for col in df_all.columns if "angle_change_orientation" in col]

    #melt linear_movement columns together in one
    df_temp = df_all[mov_cols].melt(var_name = "columns", value_name = "value")
    #count the values and plot it
    ax = df_temp["value"].value_counts().sort_index().plot.bar()
    for i, t in enumerate(ax.get_xticklabels()):
        if (i % 2) != 0:
            t.set_visible(False)
    fig = ax.get_figure()
    fig.set_size_inches(25, 12.5)
    fig.savefig(path_to_save + "plot_linear_movement.png")

    df_temp = df_all[ang_pos_cols].melt(var_name = "columns", value_name = "value")
    ax = df_temp["value"].value_counts().sort_index().plot.bar()
    for i, t in enumerate(ax.get_xticklabels()):
        if (i % 2) != 0:
            t.set_visible(True)
    fig = ax.get_figure()
    fig.set_size_inches(25, 12.5)
    fig.savefig(path_to_save + "plot_angle_new_pos.png")

    df_temp = df_all[ang_ori_cols].melt(var_name = "columns", value_name = "value")
    ax = df_temp["value"].value_counts().sort_index().plot.bar()
    fig = ax.get_figure()
    fig.set_size_inches(25, 12.5)
    fig.savefig(path_to_save + "plot_angle_change_orientation.png")

def getClusters(paths, path_to_save, round, count_clusters = (20, 20, 20)):
    """
    paths should be iterable
    path_to_save is the folder in which it will be saved
    round will round to one decimal place (round = 2 -> all values will be of the form x.yz)
    Finds cluster centers for the dataframes (for mov, pos, ori) (count_clusters should be a tuple with (count_clusters_mov, count_clusters_pos, count_clusters_ori)) and saves these clusters
    """
    #right now it thinks that 0 and 2*pi are not the same for pos and ori, maybe find a solution for it
    df_list = []
    for path in paths:
        df_list.append(pd.read_csv(path, sep = ";"))
    df_all = pd.concat(df_list)
    #round values
    df_all = df_all.round(round)

    #get all linear_movement columns respectivly all angle columns
    mov_cols = [col for col in df_all.columns if "linear_movement" in col]
    ang_pos_cols = [col for col in df_all.columns if "angle_new_pos" in col]
    ang_ori_cols = [col for col in df_all.columns if "angle_change_orientation" in col]

    df_mov = df_all[mov_cols].melt(var_name = "columns", value_name = "value")
    df_pos = df_all[ang_pos_cols].melt(var_name = "columns", value_name = "value")
    df_ori = df_all[ang_ori_cols].melt(var_name = "columns", value_name = "value")

    kmeans_mov = KMeans(n_clusters = count_clusters[0]).fit(df_mov["value"].to_numpy().reshape(-1, 1))
    kmeans_pos = KMeans(n_clusters = count_clusters[1]).fit(df_pos["value"].to_numpy().reshape(-1, 1))
    kmeans_ori = KMeans(n_clusters = count_clusters[2]).fit(df_ori["value"].to_numpy().reshape(-1, 1))

    with open(path_to_save + "clusters.txt", "w+") as f:
        f.write("count_clusters(mov, pos, ori)\n" + str(count_clusters) + "\n")
        for elem in kmeans_mov.cluster_centers_:
            f.write(str(float(elem)) +"\n")
        for elem in kmeans_pos.cluster_centers_:
            f.write(str(float(elem)) +"\n")
        for elem in kmeans_ori.cluster_centers_:
            f.write(str(float(elem)) +"\n")

def main():
    # file = "data/sleap_1_Diffgroup1-1.h5"
    # fish1 = reader.extract_coordinates(file, [b'center'], [0])
    # fish2 = reader.extract_coordinates(file, [b'center'], [1])
    # tracks = reader.extract_coordinates(file, [b'center'], [0,1,2])

    # plot_follow(tracks)

    getClusters(["data/locomotion_data.csv"], "data/", 2, (15, 20, 17))

if __name__ == "__main__":
    main()
