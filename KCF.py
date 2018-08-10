
# coding: utf-8

# In[4]:


import cv2
import sys
from yolo_bicycle import object_detection


# object tracking algorithm, for more details, refer to [this](https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/#opencv-tracking-api)

# In[5]:


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
    bbox = object_detection(frame)
 
    # Initialize tracker with first frame and bounding box
    tracker = cv2.TrackerKCF_create()
    ok = tracker.init(frame, bbox)
    
    ret = True
    while ret:
        # Read a new frame
        ret, frame = video.read()
        if not ret:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            new_frame = cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(new_frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
        # Display tracker type on frame
        cv2.putText(new_frame, tracker_types[2] + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(new_frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        out.write(new_frame)
        cv2.imshow("Tracking", new_frame)
        
    out.release()
    video.release()
    cv2.destroyAllWindows()


# In[7]:


object_tracking(filename='videos/riding2.mp4')

