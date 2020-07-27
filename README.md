# Elephant-Fish

![Hi, I am Peter](data/peter.jpg)

## Authors

* Marc Gröling
* Gabriel Kressin

## The goal

How do animals decide what to do and where to move as a group? We use Guppys as a model for collective behavior. Typically, individual fish have been modeled with simple rules that determine their direction of movement in the immediate future. But even our “simple” Guppys are much more complex - they exhibit consistent behavioral variation (“personalities”), they remember things learned in the past and they build up social preferences. In project “RoboFish” our goal is to learn (data-driven!) models of fish behavior through observation and manipulation (using a robotic fish). Your goal in this software project is to learn deep neural network models (or RNNs) from live fish data and see how the prediction accuracy changes when providing additional information about the environment or the target individual than just trajectories.

##### TL;DR Goal is to create a similar model to the model from [Moritz Master thesis](https://git.imp.fu-berlin.de/bioroboticslab/robofish/mm_master). And then see how the accuracy compares to a model which is trained on additional information.

## Project state

Here you find the presentations:
* [Midterm Presentation](https://docs.google.com/presentation/d/1j3MFCuJ0u3WaQ5IgOig-MALxQPnDhmf9EoApNko3mCg/edit?usp=sharing)
* [Final Presentation](https://docs.google.com/presentation/d/1CoAPEFmZlw0pXVqQY0UqzyyiwGDfyVOOpM-6B-adfpI/edit?usp=sharing).

All functions have documentation and for most there is above or in the `main()` function of that file.

We managed to do and implement following:
* Video labelling and track extraction with sleap
  + Tracking Data found in `data/sleap_1_[diff/same]x.hd5`
  + Reader implementation for data, missing data interpolation and outlier correction
  + Found in `src/reader.py`, usage: `extract_coordinates( ... )`
* Locomotion and nLoc computation for trackdata
  + Found in `src/locomotion.py`
  + Usage: `getLocomotion( ... )`
  + or   : `getnLoc( ... )`
  + Computed locomotions are in `data/locomotion_data_[diff/same]x.csv
* Raycast and wallrays computation
  + Found in `src/raycasts.py`
  + Usage: Create a raycast object and then call `ray.getRays( ... )` (also computes wallrays)
  + Computed raycasts and wallrays are in `data/raycast_data_[diff/same]x.csv
* Evaluation and Analysis of given tracks
  + Found in `src/evaluation.py`
  + Usage: `create_plots( ... )` (single trackset)
  + or   : `create_all_plots_seperate( ... )` (creates seperate plots for all live fish tracksets)
  + or   : `create_all_plots_together( ... )` (creates  plots for all live fish tracksets together)
  + or you use a specific plotting function, be carefull for the amount of nodes for the input!
  + There is plots and figures for all tracks from the fish videos in the `figures` folder!
* Visualization of tracks
  + Found in `src/visualization.py`
  + See down below in "visualization" for more information
* Baseline model
  + Found in `src/main.py`
  + See below for explanation
* nmodel
  + Found in `src/nmodel.py`
  + See below for explanation

# Dependencies

These dependencies will be needed for this project to work, make sure you have installed them correctly.
* Python 3.6.10
* sklearn
* numpy
* python
* imageio
* seaborn
* matplotlib
* scipy
* h5py
* sys
* math
* os
* itertools
* keras
* tensorflow 2.2.1
* random
* shap
* collections
* kneed

For visualization:
* opencv2 4.3.0.36 (opencv-python)

# General Information

## Bin Representation of Locomotion ( this was the goal, we actually did not get there )

For our model input and output of locomotion should have been in the form of bin representation.\
This means that a given locomotion, e.g. linear movement of 3 is matched to bins, say bin 2 and 5 where 2 has a higher percentage cause it represents the locomotion better.\
We used this form of locomotion representation because movement of fishes is not deterministic and there is not just one right output value for each input, but there is a distribution over the possible locomotion.\
However this leaves one with a decision to which bin centers should be used and how many one should use.\
For the problem of how which center points to use, we decided that it would make sense to apply kmeans to our given locomotion from the videos, giving us higher precision in sections where a lot of movement occured.\
Now for the problem of how many center points to use, we had an objective to use as few center points as possible, that still lead to an acceptable amount of loss of information. After considering plots of locomotion, we decided that at least at 8 bins should be used for linear movement and for change in orientation (this basically gives us a wasd-keyboard for change in orientation and enough bins to have certain speeds of going forward and backward) and at least 16 for the angular change (this plot was a lot more evenly distributed, than the orientation plot, which lead us to to the belief that this one needs more ways of moving)\
Now we had the minimum number of bins to use, however this would probably not be that good. We plotted the loss of information vs count of clusters and used a method to find the elbow of that curve (point that has the maximum curvature). (see figures/cluster_plots) And this gives us kind of an optimal value for how many cluster centers to use. We decided to set a max value of 50 cluster centers, the loss of information stagnated at about that point.

##### TL;DR We should have used bin representation of locomotion (using kmeans as centers) and used 18 bins for linear movement, 17 for change in orientation and 26 for angular change.

## Actual Locomotion

The locomotion for each timestep is represented with 3 variables, given center and head at timestep t-1 and t:\
first let us define the look vector: the look vector is defined as the subtraction of the head node - the center node (basically the direction in which the fish is looking)\
-linear movement, which is the distance between the center point at t-1 and t\
-angular movement, which is the angle between the look_vector at timestep t-1 and the vector from the center at t-1 to the center at t\
-orientational movement, which is the angle between the look_vector at timestep t-1 and the look_vector at timestep t\

## Raycasts

In this master thesis https://www.mi.fu-berlin.de/inf/groups/ag-ki/Theses/Completed-theses/Master_Diploma-theses/2019/Maxeiner/MA-Maxeiner.pdf it is explained how raycasts were done. It can be found under section 3.2.2 and 3.2.3


# BASELINE Model

## Input
The Baseline model gets as input a sequence of hand-crafted vectors, which consist of the following: last locomotion (the locomotion that lead the fish to this point), wall rays and agent rays. The sequence length basically means that the network gets this hand-crafted vector for the last length_sequence timesteps.

## Output

The output is the locomotion for the following step.

## Network structure

We tried different network structures, but they all looked similar:\
LSTM-layer with 40-128 nodes\
Dropout-layer with a Dropout of 0.1-0.3\
Dense-layer with 20-64 nodes\
Dropout-layer with a Dropout of 0.1-0.3\
Dense-layer with as many nodes as the output is big (3 for non-bin-approach)\

# n Node Model

n stands for the amount of nodes

## The nloc model

Datastructure to represent movement. n stands for the amount of nodes per fish
* The first three arguments provide the change from the old center point to the new one, dis, angular speed and orientation. All in relation to the previous orientation.
* From that point the base fish model will be computed on
* The rest (n - 1) * 2 arguments are the distance and orientation to the center node

This is how an nloc array looks:
```
[
    [f1_lin, f1_ang, f1_ori, f1_1_dis, d1_1_ori, f1_2_dis, f1_2_ori, ..., f2_lin, f2_ang, f2_ori, ... ]
    ...
]
```

## The nView

Datastructure to represent the view of a fish. A fish in the n Node Model views n nodes for every other fish. This is possible since we are only using 3 fishes constantly.
The nView vector saves the distances and angles from the center of the fish to the n nodes of every other fish.

This is how an nView array looks like for fish1:
```
[
    [f2_n1_dis, f2_n1_ang, f2_n2_dis, f2_n2_ang, ..., f3_n1_dis, f3_n1_ang, ...]
]
```

## Input

The model gets as input:
An Tensor with HIST_SIZE amount of datasets, each containing BATCH_SIZE amount of datapoints in the form of:
* Wall rays, which are taken from beforehand computed raycastdata with `stealWallrays( ... )` ( N_WRAYS is the amount of variables in a single row)
* nView `getnView` for the certain fish ( N_VIEWS is the amount of variables in a single row )
* nLoc from previos timestep `locomotion.getnLoc` ( D_LOC is the amount of variables in a single row)

Which leaves the input with shape (HIST_SIZE, BATCH_SIZE, N_VIEWS + N_WRAYS + D_LOC )

## Output
* The nloc for that timestep

output shape ( INPUT_BATCH_SIZE , D_LOC )

## Useful functions

There is a few functions to make your life easier:
* `loadData( ... )` to load the data in
* `getDatasets( ... )` shape data into correct tf datasets
* `createModel( ... )` creates the model
* `saveModel( ... )` saves the model
* `plot_train_history( ... )` plots the train history

## Training

You train the model with `model.fit( ... )` (tensorflow API).

## Simulation

1. Load the model in
2. Get Start locations with `loadStartData( ... )`
3. Call `simulate( ... )`, which will return the trackset, polarCoordinates of the center and the prediction results.
4. You can save the tracksets to then use evaluation/visualization on them.

Optionally you can lazily switch "True" to "False" in line 461 in `src/nmodel.py`, where you have an exemplary simulation.

# Visualization

Make sure to have opencv 4.3.0 set up correctly, also set up ffmpeg/gstreamer.

## Tracksets on Videos

```
visualization.addTracksOnVideo( inputvideo, outputvideo, tracks, nfish = 3, fps=30, dimension=(960,720), psize=1, showvid=False, skeleton=None )

inputvideo: path to inputvideo
outputviedo: path to outputvideo
tracks: trackset with tracks for every frame of inputvideo
nfish: number of fish in trackset
psize: size of the points put on fish
showvid: show video while rendering (for debugging)
skeleton: A mapping of indices to connect points with lines,
    e.g. trackset with rows: [head1_x, head1_y, center1_x, center1_y, head2_x, ...]
    to connect center and head give [(0,1)]

    for full trackset (10 nodes per fish) use this:
    [(0,1), (0,2), (0,3), (1,2), (1,3), (2,4), (3,5), (2,6), (3,7), (6,8), (7,8), (8,9)]
```

## Tracksets on Tank

```
visualization.addTracksOnTank( outputvideo, tracks, tank="data/tank.png", nfish = 3, fps=30, dimension=(960,720), psize=1, showvid=False, skeleton=None )

outputviedo: path to outputvideo
tracks: trackset with tracks for every frame of inputvideo
tank: tank picture in background
nfish: number of fish in trackset
psize: size of the points put on fish
showvid: show video while rendering (for debugging)
skeleton: A mapping of indices to connect points with lines,
    e.g. trackset with rows: [head1_x, head1_y, center1_x, center1_y, head2_x, ...]
    to connect center and head give [(0,1)]

    for full trackset (10 nodes per fish) use this:
    [(0,1), (0,2), (0,3), (1,2), (1,3), (2,4), (3,5), (2,6), (3,7), (6,8), (7,8), (8,9)]
```


# Given Videos

## Video Mapping

We mapped following names to the given videos:

```
diff_1 - DiffGroup1-1
diff_2 - DiffGroup5_2-cut
diff_3 - DiffGroup9_1-1
diff_4 - DiffGroup9_3-1
diff_5 - DiffGroup31_3-Cut
same_1 - SameGroup5_1-1
same_2 - SameGroup5_3-1
same_3 - SameGroup9_2-1
same_4 - SameGroup31_1-1
same_5 - SameGroup31_2-1
```

## Frames used

```
diff_1: entire video
diff_2: entire video
diff_3: frame 0 - 17000
diff_4: frame 120 - end
diff_5: not used
same_1: entire video
same_2: not used
same_3: frame 130 - end
same_4: entire video
same_5: entire video
```

# Future improvements

Improvements which would or could have improved the results.

### Data extraction
* More training on data in sleap (probably 200 - 600 frames more), specifically for edge cases
* Better interpolation method:
* Detect outliers not by constant velocity but rather by a factor depending on the current velocity
* Detect "really bad sequence" of frames more reliably
* Include consistency checks on Nodes in relation to each other (e.g. center needs to be between tail and head, and many more)

### Models
* Bin representation for Locomotion (very important)
* Create a fix for "the angle problem" (as described in presentation) (very important)
  + Either by the fix explained in presentation
  + Or by representing angles as vectors

### Organisational
* Use more time for the model itself and not for the things around it
* Start as early as possible with the model
