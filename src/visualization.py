# Visualize the tracksets into the video files

import numpy as np
import pandas
import cv2
import os
import sys

# Video paths
diff1 = "E:\\VideosSWP\\diff_1.avi"
diff2 = "E:\\VideosSWP\\diff_2_edit.avi"
diff3 = "E:\\VideosSWP\\diff_3_edit.avi"
diff4 = "E:\\VideosSWP\\diff_4.avi"
same1 = "E:\\VideosSWP\\same_1_edit.avi"
same3 = "E:\\VideosSWP\\same_3_edit.avi"
same4 = "E:\\VideosSWP\\same_4_edit.avi"
same5 = "E:\\VideosSWP\\same_5_edit.avi"

# OpenCV Version 4.3.0.36


def main():
    print( cv2.__version__)
    print( "hello world" )

    cap = cv2.VideoCapture( diff1 )
    if cap is None:
        print( "not able to open video" )
        sys.exit()

    while( cap.isOpened() ):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
