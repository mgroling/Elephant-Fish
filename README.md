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

## Bin Representation of Locomotion

For our model input and output of locomotion was in the form of bin representation.
This means that a given locomotion, e.g. linear movement of 3 is matched to bins, say bin 2 and 5 where 2 has a higher percentage cause it represents the locomotion better.
We used this form of locomotion representation because movement of fishes is not deterministic and we wanted some way of portraying that.
However this leaves one with a decision to which bin centers should be used and how many one should use.
For the problem of how which center points to use, we decided that it would make sense to apply kmeans to our given locomotion from the videos, giving us higher precision in sections where a lot of movement occured.
Now for the problem of how many center points to use, we had an objective to use as few center points as possible, that still lead to an acceptable amount of loss of information. After considering plots of locomotion, we decided that at least at 8 bins should be used for linear movement and for change in orientation (this basically gives us a wasd-keyboard for change in orientation and enough bins to have certain speeds of going forward and backward) and at least 16 for the angular change (this plot was a lot more evenly distributed, than the orientation plot, which lead us to to the belief that this one needs more ways of moving)
Now we had the minimum number of bins to use, however this would probably not be that good. We plotted the loss of information vs count of clusters and used a method to find the elbow of that curve (point that has the maximum curvature). (see figures/cluster_plots) And this gives us kind of an optimal value for how many cluster centers to use. We decided to set a max value of 50 cluster centers, the loss of information became to stagnate pretty hard at that point. 

TL;DR We used a bin representation of locomotion and used 18 bins for linear movement, 17 for change in orientation and 26 for angular change.


Best labels on: same_2

## Future improvements

Improvements we did not invest time in, since the data retrieval was not the main focus of this software project.

* More training on data in sleap (probably 200 - 600 frames more), specifically for edge cases
* Better interpolation method:
* Detect outliers not by constant velocity but rather by a factor dependend on the current velocity
* Detect "really bad sequence" of frames more reliably
* Include consistency checks on Nodes in relation to each other (e.g. center needs to be between tail and head, and many more)
