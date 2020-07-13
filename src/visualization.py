# Visualize the tracksets into the video files

import numpy as np
import pandas
import cv2
import os
import sys
import reader

# Video paths
diff1 = "E:\\VideosSWP\\diff_1.avi"
diff2 = "E:\\VideosSWP\\diff_2_edit.avi"
diff3 = "E:\\VideosSWP\\diff_3_edit.avi"
diff4 = "E:\\VideosSWP\\diff_4.avi"
same1 = "E:\\VideosSWP\\same_1_edit.avi"
same3 = "E:\\VideosSWP\\same_3_edit.avi"
same4 = "E:\\VideosSWP\\same_4_edit.avi"
same5 = "E:\\VideosSWP\\same_5_edit.avi"
diff1_out = 'C:/Users/Gabriel/Videos/diff_1_tracks.avi'
diff2_out = 'C:/Users/Gabriel/Videos/diff_2_tracks.avi'

# OpenCV Version 4.3.0.36

def addTracksOnVideo( inputvideo, outputvideo, tracks, nfish = 3, fps=30, dimension=(960,720), psize=1, showvid=False ):
    """
    Takes tracks and adds them on video
    """
    row, col = tracks.shape
    assert row >= 1
    assert col > 0
    assert col % nfish == 0
    nnodes = col // nfish

    # Set up input
    cap = cv2.VideoCapture( inputvideo )
    if cap is None:
        print( "not able to open video" )
        sys.exit(-1)

    # Set up output
    fourcc = cv2.VideoWriter_fourcc( 'D', 'I', 'V', 'X' )
    out = cv2.VideoWriter( outputvideo, fourcc, fps, dimension )

    # Process video
    colors = [(0,255,255), (0,255,0), (0,0,255)]
    i = 0
    while( cap.isOpened() ):
        if i % 1000 == 0:
            print( "Frame: ", i )
        if i == row:
            print( "trackset too short" )
            sys.exit(-1)

        success, frame = cap.read()
        if success:
            # frame = cv2.flip( frame, 0 )
            for f in range(nfish):
                for n in range(nnodes//2):
                    x = int( tracks[i,f * nnodes + 2 * n] )
                    y = int( tracks[i,f * nnodes + 2 * n + 1] )
                    frame = cv2.circle( frame, ( x, y ), psize, colors[f], -1 )
            out.write( frame )
            if showvid:
                cv2.imshow( 'fishy fish fish', frame )
                cv2.waitKey(0)
        else:
            print( "end of video" )
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        i += 1

    cap.release()
    cv2.destroyAllWindows()

    if i < row - 1:
        print( "Not the entire trackset was used" )


def main():

    tracks = reader.extract_coordinates( "data/sleap_1_diff2.h5", [b'head', b'center', b'l_fin_basis', b'r_fin_basis', b'l_fin_end', b'r_fin_end', b'l_body', b'r_body', b'tail_basis', b'tail_end'] )

    addTracksOnVideo( diff2, diff2_out, tracks )



if __name__ == "__main__":
    main()
