#! /usr/bin/env python
# coding: utf-8

import cv2
import sys
from object_detection import object_detection


# object tracking algorithm, for more details, refer to https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/#opencv-tracking-api


def object_tracking(filename="videos/riding.mp4"):
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    
    # Read video
    video = cv2.VideoCapture(filename)
    
    # establish output file, the same as video.
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)   
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT) 
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('./output.avi',fourcc, 20.0, (int(width), int(height)))
 
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
     
    # Define an initial bounding box
 
    # Uncomment the line below to select a different bounding box
    bboxes = object_detection(frame)
 
    # Initialize mutiltracker with first frame and bounding box
    trackerType = 'CSRT'
    multiTracker = cv2.MultiTracker_create()
    colors = []
    for bbox in bboxes:
        multiTracker.add(cv2.TrackerCSRT_create(), frame, bbox)
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    
    ret = True
    while ret:
        # Read a new frame
        ret, frame = video.read()
        if not ret:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        success, bboxes = multiTracker.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
 
        # Draw bounding box
        if success:
            # Tracking success
            for i, newbox in enumerate(bboxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                new_frame = cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        else :
            # Tracking failure
            cv2.putText(new_frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
        # Display tracker type on frame
        cv2.putText(new_frame, trackerType + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(new_frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        out.write(new_frame)
        cv2.imshow("Tracking", new_frame)
        
    out.release()
    video.release()
    cv2.destroyAllWindows()


object_tracking(filename='videos/riding2.mp4')

