# File to analyse the results
# Copied from Moritz Maxeiner Masterthesis: https://git.imp.fu-berlin.de/bioroboticslab/robofish/ai

from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import collections
from kneed import KneeLocator
from yellowbrick.cluster import distortion_score
import matplotlib.pyplot as plt

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


def getClusters(paths, path_to_save, count_clusters = (20, 20, 20), verbose = False):
    """
    paths should be iterable
    path_to_save is the folder in which it will be saved
    Finds cluster centers for the dataframes (for mov, pos, ori) (count_clusters should be a tuple with (count_clusters_mov, count_clusters_pos, count_clusters_ori)) and saves these clusters
    """
    #right now it thinks that 0 and 2*pi are not the same for pos and ori, maybe find a solution for it
    df_list = []
    for path in paths:
        df_list.append(pd.read_csv(path, sep = ";"))
    df_all = pd.concat(df_list)

    #get all linear_movement columns respectivly all angle columns
    mov_cols = [col for col in df_all.columns if "linear_movement" in col]
    ang_pos_cols = [col for col in df_all.columns if "angle_new_pos" in col]
    ang_ori_cols = [col for col in df_all.columns if "angle_change_orientation" in col]

    df_mov = df_all[mov_cols].melt(var_name = "columns", value_name = "value")
    df_pos = df_all[ang_pos_cols].melt(var_name = "columns", value_name = "value")
    df_ori = df_all[ang_ori_cols].melt(var_name = "columns", value_name = "value")

    df_pos["value"] = convertRadiansRange(df_pos["value"].to_numpy())
    df_ori["value"] = convertRadiansRange(df_ori["value"].to_numpy())

    kmeans_mov = KMeans(n_clusters = count_clusters[0]).fit(df_mov["value"].to_numpy().reshape(-1, 1))
    kmeans_pos = KMeans(n_clusters = count_clusters[1]).fit(df_pos["value"].to_numpy().reshape(-1, 1))
    kmeans_ori = KMeans(n_clusters = count_clusters[2]).fit(df_ori["value"].to_numpy().reshape(-1, 1))

    if verbose == True:
        freq_mov = collections.Counter(kmeans_mov.labels_)
        centers = kmeans_mov.cluster_centers_
        x, y = zip(*freq_mov.items())
        x, y = list(x), list(y)
        for i in range(0, len(x)):
            x[i] = float(centers[x[i]])
        temp_1 = [X for _,X in sorted(zip(x,y))]
        temp_2 = [Y for Y,_ in sorted(zip(x,y))]
        plt.figure(figsize = (24, 16))
        ax = plt.bar(np.arange(count_clusters[0]), temp_1)
        plt.title("Elements per cluster for linear movement", fontsize = 30)
        plt.xlabel("clusters", fontsize = 30)
        plt.ylabel("count elements", fontsize = 30)
        plt.xticks(np.arange(count_clusters[0]), np.round(temp_2, 3))
        plt.savefig("figures/cluster_plots/mov_elems_per_cluster_" + str(count_clusters[0]))
        plt.clf()

        freq_pos = collections.Counter(kmeans_pos.labels_)
        centers = kmeans_pos.cluster_centers_
        x, y = zip(*freq_pos.items())
        x, y = list(x), list(y)
        for i in range(0, len(x)):
            x[i] = float(centers[x[i]])
        temp_1 = [X for _,X in sorted(zip(x,y))]
        temp_2 = [Y for Y,_ in sorted(zip(x,y))]
        plt.figure(figsize = (24, 16))
        ax = plt.bar(np.arange(count_clusters[0]), temp_1)
        plt.title("Elements per cluster for change in position (radians)", fontsize = 30)
        plt.xlabel("clusters", fontsize = 30)
        plt.ylabel("count elements", fontsize = 30)
        plt.xticks(np.arange(count_clusters[0]), np.round(temp_2, 3))
        plt.savefig("figures/cluster_plots/pos_elems_per_cluster_" + str(count_clusters[0]))
        plt.clf()

        freq_ori = collections.Counter(kmeans_ori.labels_)
        centers = kmeans_ori.cluster_centers_
        x, y = zip(*freq_ori.items())
        x, y = list(x), list(y)
        for i in range(0, len(x)):
            x[i] = float(centers[x[i]])
        temp_1 = [X for _,X in sorted(zip(x,y))]
        temp_2 = [Y for Y,_ in sorted(zip(x,y))]
        plt.figure(figsize = (24, 16))
        ax = plt.bar(np.arange(count_clusters[0]), temp_1)
        plt.title("Elements per cluster for change in orientation (radians)", fontsize = 30)
        plt.xlabel("clusters", fontsize = 30)
        plt.ylabel("count elements", fontsize = 30)
        plt.xticks(np.arange(count_clusters[0]), np.round(temp_2, 3))
        plt.savefig("figures/cluster_plots/ori_elems_per_cluster_" + str(count_clusters[0]))
        plt.clf()

    with open(path_to_save + "clusters.txt", "w+") as f:
        f.write("count_clusters(mov, pos, ori)\n" + str(count_clusters) + "\n")
        for elem in kmeans_mov.cluster_centers_:
            f.write(str(float(elem)) +"\n")
        for elem in kmeans_pos.cluster_centers_:
            f.write(str(float(elem)) +"\n")
        for elem in kmeans_ori.cluster_centers_:
            f.write(str(float(elem)) +"\n")

def createAverageDistanceForClusters(paths, count_cluster_min, count_cluster_max, count_cluster_step):
    #right now it thinks that 0 and 2*pi are not the same for pos and ori, maybe find a solution for it
    df_list = []
    for path in paths:
        df_list.append(pd.read_csv(path, sep = ";"))
    df_all = pd.concat(df_list)

    #get all linear_movement columns respectivly all angle columns
    mov_cols = [col for col in df_all.columns if "linear_movement" in col]
    ang_pos_cols = [col for col in df_all.columns if "angle_new_pos" in col]
    ang_ori_cols = [col for col in df_all.columns if "angle_change_orientation" in col]

    df_mov = df_all[mov_cols].melt(var_name = "columns", value_name = "value")
    df_pos = df_all[ang_pos_cols].melt(var_name = "columns", value_name = "value")
    df_ori = df_all[ang_ori_cols].melt(var_name = "columns", value_name = "value")

    df_pos["value"] = convertRadiansRange(df_pos["value"].to_numpy())
    df_ori["value"] = convertRadiansRange(df_ori["value"].to_numpy())

    y_mov = []
    y_pos = []
    y_ori = []
    x = []

    for i in range(count_cluster_min, count_cluster_max+1, count_cluster_step):
        print("||| Now computing for n_clusters = " + str(i) + " |||")
        kmeans_mov = KMeans(n_clusters = i)
        kmeans_pos = KMeans(n_clusters = i)
        kmeans_ori = KMeans(n_clusters = i)

        kmeans_mov.fit_transform(df_mov["value"].to_numpy().reshape(-1, 1))
        kmeans_pos.fit_transform(df_pos["value"].to_numpy().reshape(-1, 1))
        kmeans_ori.fit_transform(df_ori["value"].to_numpy().reshape(-1, 1))

        dist_mov = distortion_score(df_mov["value"].to_numpy().reshape(-1, 1), kmeans_mov.labels_)
        dist_pos = distortion_score(df_pos["value"].to_numpy().reshape(-1, 1), kmeans_pos.labels_)
        dist_ori = distortion_score(df_ori["value"].to_numpy().reshape(-1, 1), kmeans_ori.labels_)

        y_mov.append(dist_mov)
        y_pos.append(dist_pos)
        y_ori.append(dist_ori)
        x.append(i)

    return x, y_mov, y_pos, y_ori

def kneeLocatorPlotter(x, y, title, kindOfLoc):
    # https://www.kaggle.com/kevinarvai/knee-elbow-point-detection

    kneedle = KneeLocator(x, y, S=1.0, curve = "convex", direction = "decreasing")

    plt.figure(figsize = (24, 16))
    plt.plot(x, y)
    plt.title(title + ", optimal k = " + str(kneedle.knee), fontsize = 30)
    plt.ylabel("distortion_score", fontsize = 30)
    plt.xlabel("count clusters", fontsize = 30)
    plt.savefig("figures/cluster_plots/knee_plot_" +  kindOfLoc)

    return kneedle.knee

if __name__ == "__main__":
    paths = ["data/locomotion_data_diff1.csv", "data/locomotion_data_diff2.csv", "data/locomotion_data_diff3.csv", "data/locomotion_data_diff4.csv", "data/locomotion_data_same1.csv", "data/locomotion_data_same3.csv", "data/locomotion_data_same4.csv", "data/locomotion_data_same5.csv"]
    #getClusters(paths, "data/", (30,30,30), verbose = True)
    x, y_mov, y_pos, y_ori = createAverageDistanceForClusters(["data/locomotion_data_diff1.csv"], 1, 25, 1)

    print(kneeLocatorPlotter(x, y_mov, "Count clusters vs. distortion score for linear movement", "mov"))
    print(kneeLocatorPlotter(x, y_pos, "Count clusters vs. distortion score for change in position (radians)", "pos"))
    print(kneeLocatorPlotter(x, y_ori, "Count clusters vs. distortion score for change in orientation (radians)", "ori"))
