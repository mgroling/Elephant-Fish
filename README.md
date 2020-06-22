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
diff_5: let's see
```

## Possible improvements

Not done since videos are not focus

* More training on data in sleap (probably 200 - 600 frames more), specifically for edge cases
* Better interpolation method:
* Detect outliers not by constant velocity but rather by a factor dependend on the current velocity
* Detect "really bad sequence" of frames more reliably
* Include consistency checks on Nodes in relation to each other (e.g. center needs to be between tail and head, and many more)
