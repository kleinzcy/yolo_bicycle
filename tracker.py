
# coding: utf-8

import cv2
import sys
from object_detection import tfnet
from random import randint

class MultiTracker():
    """the tracker type depend on your choice"""
    def __init__(self, type):
        self.net = tfnet()
        self.colors = []
        self.trackerType = type
     
    
    """Create a tracker based on tracker name"""
    def createTrackerByName(self):
        trackerType = self.trackerType
        trackerTypes = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
        if trackerType == trackerTypes[0]:
            tracker = cv2.TrackerBoosting_create()
        elif trackerType == trackerTypes[1]: 
            tracker = cv2.TrackerMIL_create()
        elif trackerType == trackerTypes[2]:
            tracker = cv2.TrackerKCF_create()
        elif trackerType == trackerTypes[3]:
            tracker = cv2.TrackerTLD_create()
        elif trackerType == trackerTypes[4]:
            tracker = cv2.TrackerMedianFlow_create()
        elif trackerType == trackerTypes[5]:
            tracker = cv2.TrackerMOSSE_create()
        elif trackerType == trackerTypes[6]:
            tracker = cv2.TrackerCSRT_create()
        else:
            tracker = None
            print('Incorrect tracker name')
            print('Available trackers are:')
            for t in trackerTypes:
              print(t)

        return tracker  
    
    
    def create_MutilTracker(self, frame):
        # obtain box through yolo
        bboxes = self.net.object_detection(frame)
        
        self.colors = []
        # Initialize mutiltracker with first frame and bounding box
        multiTracker = cv2.MultiTracker_create()
        for bbox in bboxes:
            multiTracker.add(self.createTrackerByName(), frame, bbox)
            self.colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        return multiTracker
    
    
    """object detection every num frame"""
    def tracking(self, num=30, filename="videos/riding.mp4"):
        # Read video
        video = cv2.VideoCapture(filename)

        # establish output file, the same as video.
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)   
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT) 
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('./output/'+self.trackerType+'.avi',fourcc, 20.0, (int(width), int(height)))

        # Exit if video not opened.
        if not video.isOpened():
            print("Could not open video")
            sys.exit()

        ret = True
        num_frame = -1
        while ret:
            # Read a new frame
            ret, frame = video.read()
            if not ret:
                break

            # inintinal multitracker
            num_frame += 1
            if num_frame % num == 0:
                multitracker = self.create_MutilTracker(frame)
                num_frame = 0
                continue


            # Start timer
            timer = cv2.getTickCount()

            # Update tracker
            success, bboxes = multitracker.update(frame)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Draw bounding box
            if success:
                # Tracking success
                for i, newbox in enumerate(bboxes):
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    new_frame = cv2.rectangle(frame, p1, p2, self.colors[i], 2, 1)
            else :
                # Tracking failure
                new_frame = cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2)

            # Display tracker type on frame
            cv2.putText(new_frame, self.trackerType + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

            # Display FPS on frame
            cv2.putText(new_frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

            # Display result
            cv2.namedWindow("Video") 
            out.write(new_frame)
            cv2.imshow("Tracking", new_frame)
            k = cv2.waitKey(1) & 0xff
            if k == 27 : 
                break

        out.release()
        video.release()
        cv2.destroyAllWindows()
        
        
if __name__=='__main__':
    object_tracker = MultiTracker('MOSSE')
    object_tracker.tracking(num=100)
