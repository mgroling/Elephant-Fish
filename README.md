# Elephant-Fish

![Our one and only love](http://cdn.sci-news.com/images/enlarge5/image_6632_2e-Elephantnose-Fish.jpg)

# Dependencies

These dependencies will be needed for this project to work:
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
* itertools
* os
* opencv2 4.3.0.36 (opencv-python)

# Workflow

1. Install all dependencies
2. Get the fishdata
3. Train specific Network on fishdata using this command `[to be written]`
4. Simulate specific Network using this command `[to be written]`
5. Convert simulation output to trackset using this command `[to be written]`
6. Evaluate simulation by using `evaluation.create_plots(tracksets)`
7. Happy evaluating!

Maybe: Do everything at once using this command `[]`

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

Best labels on: same_2

## Future improvements

Improvements we did not invest time in, since the data retrieval was not the main focus of this software project.

* More training on data in sleap (probably 200 - 600 frames more), specifically for edge cases
* Better interpolation method:
* Detect outliers not by constant velocity but rather by a factor dependend on the current velocity
* Detect "really bad sequence" of frames more reliably
* Include consistency checks on Nodes in relation to each other (e.g. center needs to be between tail and head, and many more)

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

## Fish emplacement

If only given center and head values, you can use
```
@TODO
```
to add static positions for the fish.