# Python file to evaluate the fish network
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn
import os
import reader
import locomotion
from analysis import *
from functions import *


def plot_follow(tracks, max_tolerated_movement=20):
    """
    Create and save Follow graph, only use center nodes for it
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

    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.93)
    ax.set_xlim(-max_tolerated_movement, max_tolerated_movement)
    seaborn.distplot(pd.Series(follow, name="Follow"), ax=ax, hist_kws={"rwidth":0.9, "color":"y"})

    return fig


def plot_iid(tracks):
    """
    Create and save iid graph, only use center nodes for it
    Expects one node per fish at max: tracks: [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              ...
    """
    assert tracks.shape[-1] % 2 == 0
    nfish = int(tracks.shape[-1] / 2)

    iid = []

    # for every fish combination calculate iid
    for i1 in range(nfish):
        for i2 in range(i1 + 1, nfish):
            f1_x, f1_y = get_indices(i1)
            f2_x, f2_y = get_indices(i2)
            iid.append(calc_iid(tracks[:-1, f1_x:f1_y + 1], tracks[:-1, f2_x:f2_y + 1]))
            iid.append(iid[-1])

    iid = np.concatenate(iid, axis=0)

    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.93)
    ax.set_xlim(0, 700)
    seaborn.distplot(pd.Series(iid, name="IID [pixel]"), ax=ax, hist_kws={"rwidth":0.9, "color":"y"})

    return fig


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
        {"IID [pixel]": np.concatenate(iid, axis=0), "Follow": np.concatenate(follow, axis=0)}
    )

    grid = seaborn.jointplot(
        x="IID [pixel]", y="Follow", data=follow_iid_data, linewidth=0, s=1, kind="scatter"
    )
    grid.ax_joint.set_xlim(0, 700)
    grid.fig.set_figwidth(9)
    grid.fig.set_figheight(6)
    grid.fig.subplots_adjust(top=0.9)
    return grid.fig


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
        {"IID [pixel]": np.concatenate(iid, axis=0), "TLVC": np.concatenate(tlvc, axis=0)}
    )

    grid = seaborn.jointplot(
        x="IID [pixel]", y="TLVC", data=tlvc_iid_data, linewidth=0, s=1, kind="scatter"
    )
    grid.ax_joint.set_xlim(0, 700)
    grid.fig.set_figwidth(9)
    grid.fig.set_figheight(6)
    grid.fig.subplots_adjust(top=0.9)
    return grid.fig


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


def plot_velocities( tracks, clusterfile = None ):
    """
    Plots the velocities
    Expects two nodes per fish exactly: tracks: [head1_x, head1_y, center1_x, center1_y, head2_x, head2_y, center2_x, center2_y,...]
                                                [head1_x, head1_y, center1_x, center1_y, head2_x, head2_y, center2_x, center2_y,...]
                                                ...
    """

    assert tracks.shape[-1] % 4 == 0
    nfish = int(tracks.shape[-1] / 4)

    if clusterfile is not None:
        cLin, cAng, cOri = readClusters( clusterfile )

    locs = locomotion.getLocomotion(tracks, None)

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

    angular_velocities = convertRadiansRange(angular_velocities)
    turn_velocities = convertRadiansRange(turn_velocities)

    fig_angular, ax = plt.subplots(figsize=(18, 18))
    fig_angular.subplots_adjust(top=0.93)
    ax.set_xlim(-np.pi, np.pi)
    seaborn.distplot(pd.Series(angular_velocities, name="Angular movement"), ax=ax, hist_kws={"rwidth":0.9, "color":"y"})
    if clusterfile is not None:
        seaborn.rugplot(cAng, height=0.03, ax=ax, color="r", linewidth=3)

    fig_turn, ax = plt.subplots(figsize=(18, 18))
    fig_turn.subplots_adjust(top=0.93)
    ax.set_xlim(-np.pi, np.pi)
    seaborn.distplot(pd.Series(turn_velocities, name="Orientational movement"), ax=ax, hist_kws={"rwidth":0.9, "color":"y"})
    if clusterfile is not None:
        seaborn.rugplot(cOri, height=0.03, ax=ax, color="r", linewidth=3)

    fig_linear, ax = plt.subplots(figsize=(18, 18))
    fig_linear.subplots_adjust(top=0.93)
    ax.set_xlim(-20, 20)
    seaborn.distplot(pd.Series(linear_velocities, name="Linear movement"), ax=ax, hist_kws={"rwidth":0.9, "color":"y"})
    if clusterfile is not None:
        seaborn.rugplot(cLin, height=0.03, ax=ax, color="r", linewidth=3)

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


def create_plots(tracks, path = "figures/latest_plots", time_step = (1000/30), tau_seconds=(0.3, 1.3), clusterfile = "data/clusters.txt" ):
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
    save_figure(plot_iid(tracksCenter), path=(path + "iid.png"))
    save_figure(plot_follow(tracksCenter), path=(path + "follow.png"))
    save_figure(plot_follow_iid(tracksCenter), path=(path + "follow_iid.png"))
    save_figure(plot_tlvc_iid(tracksCenter, time_step, tau_seconds), path=(path + "tlvc_iid.png"))
    save_figure(plot_tankpositions(tracksCenter), path=(path + "tankpostions.png"), size=(24,18))
    # Velocities
    lin,ang,trn = plot_velocities(tracks, clusterfile=clusterfile)
    save_figure(lin, path=(path + "locomotion_linear.png") , size=(18, 18))
    save_figure(ang, path=(path + "locomotion_angular.png") , size=(18, 18))
    save_figure(trn, path=(path + "locomotion_trn.png") , size=(18, 18))
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


def animate_positions(track, track2 = None):
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


def main():
    file = "data/sleap_1_diff1.h5"
    tracks = reader.extract_coordinates(file, [b'head',b'center'])

    create_plots(tracks)


if __name__ == "__main__":
    main()
