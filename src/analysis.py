# File to analyse the results
# Copied from Moritz Maxeiner Masterthesis: https://git.imp.fu-berlin.de/bioroboticslab/robofish/ai

from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import collections
from kneed import KneeLocator
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
    x_cols = [col for col in df_all.columns if "next_x" in col]
    y_cols = [col for col in df_all.columns if "next_y" in col]
    ori_cols = [col for col in df_all.columns if "angle_change_orientation" in col]

    df_mov = df_all[x_cols].melt(var_name = "columns", value_name = "value")
    df_pos = df_all[y_cols].melt(var_name = "columns", value_name = "value")
    df_ori = df_all[ori_cols].melt(var_name = "columns", value_name = "value")

    # df_pos["value"] = convertRadiansRange(df_pos["value"].to_numpy())
    # df_ori["value"] = convertRadiansRange(df_ori["value"].to_numpy())

    kmeans_mov = KMeans(n_clusters = count_clusters[0]).fit(df_mov["value"].to_numpy().reshape(-1, 1))
    kmeans_pos = KMeans(n_clusters = count_clusters[1]).fit(df_pos["value"].to_numpy().reshape(-1, 1))
    kmeans_ori = KMeans(n_clusters = count_clusters[2]).fit(df_ori["value"].to_numpy().reshape(-1, 1))

    if verbose:
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
        ax = plt.bar(np.arange(count_clusters[1]), temp_1)
        plt.title("Elements per cluster for angular change (radians)", fontsize = 30)
        plt.xlabel("clusters", fontsize = 30)
        plt.ylabel("count elements", fontsize = 30)
        plt.xticks(np.arange(count_clusters[1]), np.round(temp_2, 3))
        plt.savefig("figures/cluster_plots/ang_elems_per_cluster_" + str(count_clusters[1]))
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
        ax = plt.bar(np.arange(count_clusters[2]), temp_1)
        plt.title("Elements per cluster for change in orientation (radians)", fontsize = 30)
        plt.xlabel("clusters", fontsize = 30)
        plt.ylabel("count elements", fontsize = 30)
        plt.xticks(np.arange(count_clusters[2]), np.round(temp_2, 3))
        plt.savefig("figures/cluster_plots/ori_elems_per_cluster_" + str(count_clusters[2]))
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
        y_mov_temp = []
        y_pos_temp = []
        y_ori_temp = []
        for j in range(0, 3):
            kmeans_mov = KMeans(n_clusters = i).fit(df_mov["value"].to_numpy().reshape(-1, 1))
            kmeans_pos = KMeans(n_clusters = i).fit(df_pos["value"].to_numpy().reshape(-1, 1))
            kmeans_ori = KMeans(n_clusters = i).fit(df_ori["value"].to_numpy().reshape(-1, 1))

            dist_mov = kmeans_mov.inertia_ / len(df_mov.index)
            dist_pos = kmeans_pos.inertia_ / len(df_pos.index)
            dist_ori = kmeans_ori.inertia_ / len(df_ori.index)

            y_mov_temp.append(dist_mov)
            y_pos_temp.append(dist_pos)
            y_ori_temp.append(dist_ori)

        y_mov.append(np.mean(np.array(y_mov_temp)))
        y_pos.append(np.mean(np.array(y_pos_temp)))
        y_ori.append(np.mean(np.array(y_ori_temp)))
        x.append(i)

    return x, y_mov, y_pos, y_ori

def getLinearCenters(paths, step_x, step_y, step_ori):
    """
    creates a center for the range of min value, max_value, step for x, y and ori and saves it to data/clusters.txt
    """
    df_list = []
    for path in paths:
        df_list.append(pd.read_csv(path, sep = ";"))
    df_all = pd.concat(df_list)

    #get all linear_movement columns respectivly all angle columns
    x_cols = [col for col in df_all.columns if "next_x" in col]
    y_cols = [col for col in df_all.columns if "next_y" in col]
    ori_cols = [col for col in df_all.columns if "angle_change_orientation" in col]

    df_x = df_all[x_cols].melt(var_name = "columns", value_name = "value")
    df_y = df_all[y_cols].melt(var_name = "columns", value_name = "value")
    df_ori = df_all[ori_cols].melt(var_name = "columns", value_name = "value")

    min_x, max_x = df_x["value"].min(), df_x["value"].max()
    min_y, max_y = df_y["value"].min(), df_y["value"].max()
    min_ori, max_ori = df_ori["value"].min(), df_ori["value"].max()

    centers_x, centers_y, centers_ori = np.arange(min_x, max_x, step_x), np.arange(min_y, max_y, step_y), np.arange(min_ori, max_ori, step_ori)

    with open("data/clusters.txt", "w+") as f:
        f.write("count_clusters(x, y, ori)\n" + str((len(centers_x), len(centers_y), len(centers_ori))) + "\n")
        for elem in centers_x:
            f.write(str(float(elem)) +"\n")
        for elem in centers_y:
            f.write(str(float(elem)) +"\n")
        for elem in centers_ori:
            f.write(str(float(elem)) +"\n")

def kneeLocatorPlotter(x, y, title, kindOfLoc):
    # https://www.kaggle.com/kevinarvai/knee-elbow-point-detection

    kneedle = KneeLocator(x, y, S=1.0, curve = "convex", direction = "decreasing")

    plt.figure(figsize = (24, 16))
    plt.plot(x, y)
    plt.xticks(x)
    plt.title(title + ", optimal k = " + str(kneedle.knee), fontsize = 30)
    plt.ylabel("average squared distance to nearest cluster", fontsize = 30)
    plt.xlabel("count clusters", fontsize = 30)
    plt.savefig("figures/cluster_plots/knee_plot_" +  kindOfLoc)

    # plt.style.use("ggplot")
    # kneedle.plot_knee()

    return kneedle.knee

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()

if __name__ == "__main__":
    paths = ["data/locomotion_data_diff1.csv", "data/locomotion_data_diff2.csv", "data/locomotion_data_diff3.csv", "data/locomotion_data_diff4.csv", "data/locomotion_data_same1.csv", "data/locomotion_data_same3.csv", "data/locomotion_data_same4.csv", "data/locomotion_data_same5.csv"]
    getClusters(paths, "data/", (25, 25, 25), verbose = False)
    # getLinearCenters(paths, 0.5, 0.5, 0.25)
    # x, y_mov, y_pos, y_ori = createAverageDistanceForClusters(["data/locomotion_data_diff1.csv"], 8, 50, 1)

    # print(kneeLocatorPlotter(x, y_mov, "Count clusters vs. avg. distance to each cluster for linear movement", "mov"))
    # print(kneeLocatorPlotter(x, y_pos, "Count clusters vs. avg. distance to each cluster for change in position (radians)", "pos"))
    # print(kneeLocatorPlotter(x, y_ori, "Count clusters vs. avg. distance to each cluster for change in orientation (radians)", "ori"))
