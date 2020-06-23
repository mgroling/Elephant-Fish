# File to analyse the results
# Copied from Moritz Maxeiner Masterthesis: https://git.imp.fu-berlin.de/bioroboticslab/robofish/ai

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import os

import reader
from functions import *
import locomotion

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
    Expects one node per fish at max: tracks: [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              ...
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
    Expects one node per fish at max: tracks: [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              ...
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

def getClusters(paths, path_to_save, count_clusters = (20, 20, 20)):
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


def plot_positions(track, track2 = None):
    """
    Animation of all postions. Not optimized.
    """

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


def plot_tankpositions(tracks):
    """
    Heatmap of fishpositions
    By Moritz Maxeiner
    """

    assert tracks.shape[-1] % 2 == 0
    nfish = int(tracks.shape[-1] / 2)

    x_pos = []
    y_pos = []
    for i1 in range(nfish):
        f1_x, f1_y = get_indices(i1)
        x_pos.append( tracks[:, f1_x] )
        y_pos.append( tracks[:, f1_y] )

    x_pos = np.concatenate(x_pos, axis=0)
    y_pos = np.concatenate(y_pos, axis=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(top=0.91)
    ax.set_xlim(0, 960)
    ax.set_ylim(0, 720)
    seaborn.kdeplot(x_pos, y_pos, n_levels=25, shade=True, ax=ax)
    return fig


def plot_tlvc_iid(tracks, time_step = (1000/30), tau_seconds=(0.3, 1.3)):
    """
    TLVC_IDD by Moritz Maxeiner
    Expects one node per fish at max: tracks: [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              ...
    """
    tau_min_seconds, tau_max_seconds = tau_seconds

    assert tracks.shape[-1] % 2 == 0
    nfish = int(tracks.shape[-1] / 2)

    tlvc = []
    iid = []

    tau_min_frames = int(tau_min_seconds * 1000.0 / time_step)
    tau_max_frames = int(tau_max_seconds * 1000.0 / time_step)

    # for every fish combination calculate the follow
    for i1 in range(nfish):
        for i2 in range(i1 + 1, nfish):
            f1_x, f1_y = get_indices(i1)
            f2_x, f2_y = get_indices(i2)
            iid.append(calc_iid(tracks[1 : -tau_max_frames + 1, f1_x:f1_y + 1], tracks[1 : -tau_max_frames + 1, f2_x:f2_y + 1]))
            iid.append(iid[-1])

            a_v = tracks[1:, f1_x:f1_y + 1] - tracks[:-1, f1_x:f1_y + 1]
            b_v = tracks[1:, f2_x:f2_y + 1] - tracks[:-1, f2_x:f2_y + 1]
            tlvc.append(calc_tlvc(a_v, b_v, tau_min_frames, tau_max_frames))
            tlvc.append(calc_tlvc(b_v, a_v, tau_min_frames, tau_max_frames))

    tlvc_iid_data = pd.DataFrame(
        {"IID [cm]": np.concatenate(iid, axis=0), "TLVC": np.concatenate(tlvc, axis=0)}
    )

    grid = seaborn.jointplot(
        x="IID [cm]", y="TLVC", data=tlvc_iid_data, linewidth=0, s=1, kind="scatter"
    )
    grid.ax_joint.set_xlim(0, 142)
    grid.fig.set_figwidth(9)
    grid.fig.set_figheight(6)
    grid.fig.subplots_adjust(top=0.9)
    return grid.fig


def plot_velocities(tracks):
    """
    Plots the velocities
    Expects two nodes per fish exactly: tracks: [head1_x, head1_y, center1_x, center1_y, head2_x, head2_y, center2_x, center2_y,...]
                                                [head1_x, head1_y, center1_x, center1_y, head2_x, head2_y, center2_x, center2_y,...]
                                                ...
    """

    assert tracks.shape[-1] % 4 == 0
    nfish = int(tracks.shape[-1] / 4)
    dis = get_distances(tracks)
    print("avg:", np.mean(dis, axis=0))
    print("max:", np.amax(dis, axis=0))
    print("min:", np.amin(dis, axis=0))

    locs = locomotion.getLocomotion(tracks, None, False)

    print("Before:")
    print("avg:", np.mean(locs, axis=0))
    print("max:", np.amax(locs, axis=0))
    print("min:", np.amin(locs, axis=0))

    # Get dem indices
    i_lin = [x * 3 for x in range(nfish)]
    i_ang = [x * 3 + 1 for x in range(nfish)]
    i_trn = [x * 3 + 2 for x in range(nfish)]
    linear_velocities = locs[:,i_lin]
    angular_velocities = locs[:,i_ang]
    turn_velocities = locs[:,i_trn]

    angular_velocities = np.concatenate(angular_velocities, axis=0)
    linear_velocities = np.concatenate(linear_velocities, axis=0)
    turn_velocities = np.concatenate(turn_velocities, axis=0)

    fig_angular, ax = plt.subplots(figsize=(18, 18))
    fig_angular.subplots_adjust(top=0.93)
    ax.set_xlim(0, np.pi * 2)
    seaborn.distplot(pd.Series(angular_velocities, name="Angular velocities"), ax=ax)

    fig_turn, ax = plt.subplots(figsize=(18, 18))
    fig_turn.subplots_adjust(top=0.93)
    ax.set_xlim(0, np.pi * 2)
    seaborn.distplot(pd.Series(turn_velocities, name="Turning velocities"), ax=ax)

    fig_linear, ax = plt.subplots(figsize=(18, 18))
    fig_linear.subplots_adjust(top=0.93)
    ax.set_xlim(-20, 20)
    seaborn.distplot(pd.Series(linear_velocities, name="Linear velocities"), ax=ax)

    return fig_linear, fig_angular, fig_turn


def plot_trajectories(tracks, world=(960,720)):
    """
    Plots tank trajectory of fishes
    Expects one node per fish at max: tracks: [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              ...
    """
    assert tracks.shape[-1] % 2 == 0
    nfish = int(tracks.shape[-1] / 2)

    data = {
        fish: pd.DataFrame(
            {
                "x": tracks[:,fish*2],
                "y": tracks[:,fish*2 + 1],
            }
        )
        for fish in range(nfish)
    }
    combined_data = pd.concat([data[fish].assign(Agent=f"Agent {fish}") for fish in data.keys()])

    fig, ax = plt.subplots(figsize=(6, 6))

    seaborn.set_style("white", {"axes.linewidth": 2, "axes.edgecolor": "black"})

    seaborn.scatterplot(x="x", y="y", hue="Agent", linewidth=0, s=16, data=combined_data, ax=ax)
    ax.set_xlim(0, world[0])
    ax.set_ylim(0, world[1])
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.yaxis.set_ticks_position("left")
    ax.yaxis.set_label_position("left")

    ax.scatter(
        [frame["x"][0] for frame in data.values()],
        [frame["y"][0] for frame in data.values()],
        marker="h",
        c="black",
        s=64,
        label="Start",
    )
    ax.scatter(
        [frame["x"][len(frame["x"]) - 1] for frame in data.values()],
        [frame["y"][len(frame["y"]) - 1] for frame in data.values()],
        marker="x",
        c="black",
        s=64,
        label="End",
    )
    ax.legend()

    return fig


def create_plots(tracks, path = "figures/latest_plots", time_step = (1000/30), tau_seconds=(0.3, 1.3) ):
    """
    For given tracks create all plots in given path
    tracks only is allowed to include one node per fish!
    Expects two nodes per fish exactly: tracks: [head1_x, head1_y, center1_x, center1_y, head2_x, head2_y, center2_x, center2_y,...]
                                                [head1_x, head1_y, center1_x, center1_y, head2_x, head2_y, center2_x, center2_y,...]
                                                ...
    """
    assert tracks.shape[-1] % 4 == 0
    nfish = int(tracks.shape[-1] / 4)

    # handle dir
    if not os.path.isdir(path):
        # create dir
        try:
            os.mkdir(path)
        except OSError:
            print("Dir Creation failed")
    if path[-1] != "/":
        path = path + "/"

    # Extract Center nodes
    i_center_values = [x for x in range(nfish * 4) if x % 4 < 2]
    tracksCenter = tracks[:,i_center_values]

    # make and save graphs
    # missing: iid
    save_figure(plot_follow(tracksCenter), path=(path + "follow.png"))
    save_figure(plot_follow_iid(tracksCenter), path=(path + "follow_iid.png"))
    save_figure(plot_tlvc_iid(tracksCenter, time_step, tau_seconds), path=(path + "tlvc_iid.png"))
    save_figure(plot_tankpositions(tracksCenter), path=(path + "tankpostions.png"), size=(24,18))
    # Velocities
    lin,ang,trn = plot_velocities(tracks)
    save_figure(lin, path=(path + "velocities_linear.png") , size=(18, 18))
    save_figure(ang, path=(path + "velocities_angular.png") , size=(18, 18))
    save_figure(trn, path=(path + "velocities_trn.png") , size=(18, 18))
    # Trajectories
    save_figure(plot_trajectories(tracksCenter), path=(path + "trajectories_all.png"), size=(24,18))
    # Print trajectories for each fish
    if nfish != 1:
        for f in range(nfish):
            fx, fy = get_indices(f)
            save_figure(plot_trajectories(tracksCenter[:,[fx,fy]]), path=(path + "trajectories_agent" + str(f) + ".png"), size=(24,18))


def save_figure(fig, path = "figures/latest_plot.png", size = (25, 12.5)):
    """
    Saves the given figure in path with certain size
    """
    x, y = size
    fig.set_size_inches(x, y)
    fig.savefig(path)
    plt.close(fig)


def main():
    file = "data/sleap_1_diff1.h5"
    tracks = reader.extract_coordinates(file, [b'head',b'center'])

    create_plots(tracks)

if __name__ == "__main__":
    main()
