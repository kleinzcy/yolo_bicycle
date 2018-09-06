
# coding: utf-8

import cv2
import sys
from object_detection import tfnet
from random import randint
from overlap_ratio import overlap
from judge import Judgement
import os



class MultiTracker():
    """the tracker type depend on your choice"""
    def __init__(self, type):
        self.net = tfnet()
        self.colors = []
        self.trackerType = type
        self.new_frame = []
        self.bboxes = []
    
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
    
    def detection(self,frame):
        # obtain box through yolo
        bboxes_person, bboxes_bicycle = self.net.object_detection(frame)
        bboxes = overlap(bboxes_person,bboxes_bicycle)
        return bboxes
    
    def create_MutilTracker(self, frame, bboxes):
        
        self.colors = []
        # Initialize mutiltracker with first frame and bounding box
        multiTracker = cv2.MultiTracker_create()
        for bbox in bboxes:
            multiTracker.add(self.createTrackerByName(), frame, bbox)
            self.colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        return multiTracker
    
    
    """object detection every num frame"""
    def tracking(self, num=30, wait_frame_num=30 , filename="videos/riding.mp4",confidence=0.02):
        # Read video
        video = cv2.VideoCapture(filename)

        # establish output file, the same as video.
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)   
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT) 
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        if os.path.exists('../output'):
            out = cv2.VideoWriter('./output/'+self.trackerType+'.avi',fourcc, 20.0, (int(width), int(height)))
            print("can't find the output file, the output video will be in the file where tracker.py locate")
        else:
            out = cv2.VideoWriter(self.trackerType+'.avi',fourcc, 20.0, (int(width), int(height)))
        # Exit if video not opened.
        if not video.isOpened():
            print("Could not open video")
            sys.exit()

        ret = True
        num_frame = -1
        bicycle_flag = False
        change_flag = False
        no_bicycle_frame = 0
        multitrackers = []
        
        while ret:
            # Read a new frame
            ret, frame = video.read()
            self.new_frame = frame
            if not ret:
                break

            '''detect whether there is bicycle
               if not, wait 30 frame and detect again
               if there is, start analysing'''
            
            if bicycle_flag is False and no_bicycle_frame > 0:
                no_bicycle_frame -= 1
            elif bicycle_flag is False and no_bicycle_frame <= 0:
                self.bboxes = self.detection(frame)
                if len(self.bboxes) > 0:
                    bicycle_flag = True
                    change_flag = True
                else:
                    bicycle_flag = False
                    no_bicycle_frame = wait_frame_num
            
            #print(bicycle_flag)
            
            if bicycle_flag is True:
                # inintinal multitracker                

                num_frame += 1
                
                if num_frame % num == 0:
                    
                    bicycle_group = 0
                    if change_flag is True:
                        change_flag = False
                    else:
                        self.bboxes = self.detection(frame)        
                
                    bicycle_groups = int(len(self.bboxes) / 2)
                    
                    #print(bicycle_groups)
                    
                    while bicycle_group < bicycle_groups:
                        current_group = []
                        multitrackers = []
                        current_group.append(self.bboxes[bicycle_group*2])
                        current_group.append(self.bboxes[bicycle_group*2 + 1])
                        multitrackers.append(self.create_MutilTracker(frame, current_group))
                        bicycle_group +=1
                        
                    num_frame = 0
                    continue

                # Start timer
                timer = cv2.getTickCount()
                # Update tracker
                multitracker_num = 0
                successes = []
                update_bboxes = []
                for multitracker in multitrackers:
                    #if multitracker is not None:
                    success, update_bbox = multitracker.update(frame)
                    successes.append(success)
                    update_bboxes.append(update_bbox)
                    multitracker_num += 1

                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                
                for update_bbox in update_bboxes:
                    bbox_leave = Judgement(update_bbox, confidence)
                
                # Draw bounding box
                
                bboxes_num = 0
                for success in successes:
                    if success : 
                        # Tracking success
                        for i, newbox in enumerate(update_bboxes[bboxes_num]):
                            p1 = (int(newbox[0]), int(newbox[1]))
                            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                            j = i%2
                            self.new_frame = cv2.rectangle(frame, p1, p2, self.colors[j], 2, 1)
                    
                    bboxes_num +=1
                    #else :
                        # Tracking failure
                        #self.new_frame = cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2)
                        
                # Display tracker type on frame
                cv2.putText(self.new_frame, self.trackerType + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

                # Display FPS on frame
                cv2.putText(self.new_frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

                # Display result
            cv2.namedWindow("Video") 
            out.write(self.new_frame)
            cv2.imshow("Tracking", self.new_frame)
            k = cv2.waitKey(1) & 0xff
            if k == 27 : 
                break

        out.release()
        video.release()
        cv2.destroyAllWindows()
        
if __name__=='__main__':
    object_tracker = MultiTracker('MOSSE')
    object_tracker.tracking(num=100)

