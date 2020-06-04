import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        if (i % 3) != 0:
            t.set_visible(False)
    fig = ax.get_figure()
    fig.set_size_inches(25, 12.5)
    fig.savefig(path_to_save + "plot_linear_movement.png")

    df_ang = df_all[ang_cols].melt(var_name = "columns", value_name = "value")
    ax = df_ang["value"].value_counts().sort_index().plot.bar()
    for i, t in enumerate(ax.get_xticklabels()):
        if (i % 3) != 0:
            t.set_visible(True)
    fig = ax.get_figure()
    fig.set_size_inches(25, 12.5)
    fig.savefig(path_to_save + "plot_angle_radians.png")