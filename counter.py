"""
UGM Technofest 2019 IOT Development Source Code
people counter counter using Tensorflow Lite OpenCV 
Based off tensorflow lite image classification example at 
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
"""
# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import json
import dlib
import math
# from adrian roseberg tracking lib 
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS


#oper data ke file json 
def sendtotxt(ppl,car,xmin,xplus,ymin,yplus,count):
    data='''
    {
    "crowdcar":0,
    "crowdppl":0,
    "xmin":0,
    "xplus":0,
    "ymin":0,
    "yplus":0,
    "count":0
    }'''
    datajson=json.loads(data)
    datajson["crowdcar"]=car
    datajson["crowdppl"]=ppl
    datajson["xmin"]=xmin
    datajson["xplus"]=xplus
    datajson["ymin"]=ymin
    datajson["yplus"]=yplus
    datajson["count"]=count
    with open('jsonfile.json','w+') as f:
        json.dump(datajson,f,sort_keys=True)

# Threading to increase FPS
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(400,300),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define input arguments
MODEL_NAME = 'ssdmodel'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
min_conf_threshold = 0.5
#for normal pathwalking : 0.55-0.6
#for top down view : 0.4-0.5
imW, imH = 400, 300 #resolution output
use_TPU = False

# Import TensorFlow libraries
# If tensorflow is not installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tensorflow')
if pkg is None:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]


# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
skipframes=2
#rata - rata pixel 255/2
input_mean = 127.5
input_std = 127.5

### INITIALIZE TRACKER
# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=5, maxDistance=120)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
framecount = 0
totalDown = 0
totalUp = 0
frame8=0
###
cumperson=0
cumcar=0
countperson=0
countcar=0

# set up black canvas background
canvas= cv2.imread("black.png", cv2.IMREAD_UNCHANGED)
sumdirxmin=0
sumdirymin=0
sumdirxpos=0
sumdirypos=0
basevectorXmin=[]
basevectorXpos=[]
basevectorYmin=[]
basevectorYpos=[]
sumXvectormin=0
sumXvectorpos=0
sumYvectormin=0
sumYvectorpos=0
const=80
#time avg
listhistory=[]
#treshold direction sway
tresholdX=10
tresholdY=10
# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)
#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    canvas= cv2.imread("black.png")
    # Grab frame from video stream
    frame1 = videostream.read()
    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

   # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if (framecount % skipframes)  == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []
        #print(np.arange(0, detections.shape[2]))
        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        # loop over the detections
        
        countperson=0
        countcar=0
        for i in range(len(scores)):
            condperson = (labels[int(classes[i])] == 'person') or (labels[int(classes[i])] == 'dog') or (labels[int(classes[i])] == 'cat')
            #kondisi detect car
            condcar=(labels[int(classes[i])] == 'car') or (labels[int(classes[i])] == 'truck') or (labels[int(classes[i])] == 'bus') or (labels[int(classes[i])] == 'boat')
            #kondisi detect person & car 
            condglobal= condperson or condcar
            # filter out weak detections by requiring a minimum
            # confidence
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)) and condglobal :
                # compute the (x, y)-coordinates of the bounding box
                # for the object
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                if (condperson):
                    object_name = "person" # Look up object name from "labels" array using class index
                    countperson+=1
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(int(xmin), int(ymin), int(xmax), int(ymax))
                    tracker.start_track(frame_rgb, rect)
                    trackers.append(tracker)

                else:
                    object_name = "car"
                    countcar+=1

                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255, 255, 255), 2)
                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.57, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (0, 0, 0), 2) # Draw label text
                #cv2.putText(frame, label, (xmin, label_ymin+10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2) # Draw label text
    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        #counting=0
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(frame_rgb)
            pos = tracker.get_position()

            # unpack the position object
            xmin = int(pos.left())
            ymin = int(pos.top())
            xmax = int(pos.right())
            ymax = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((xmin, ymin, xmax, ymax))

    cv2.rectangle(frame, (0,0), (400,300), (255, 255, 255), 2)
    cv2.rectangle(canvas, (0,0), (300,300), (255, 255, 255), 2)
    cv2.arrowedLine(canvas, (50, 125),(250, 125) , (255, 255, 255), 3, 8, 0, 0.05)
    cv2.arrowedLine(canvas,(150, 225), (150, 25) , (255, 255, 255), 3, 8, 0, 0.05)
    cv2.putText(canvas, "Y", (147, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(canvas, "X", (260, 130),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)
    #print(ct.nextObjectID)
    #print(objects)
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        #avg time

        to = trackableObjects.get(objectID, None)
        sumdirx=0
        sumdiry=0
        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)
        
        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            x = [c[0] for c in to.centroids]
            directiony = -1*(centroid[1] - np.mean(y))
            directionx = centroid[0] - np.mean(x)
            #convert to base vector
            if (abs(directiony)>7.5) or (abs(directionx)>7.5):
                dirrootsquare=math.sqrt(math.pow(directionx,2)+math.pow(directiony,2))
                basex= directionx/dirrootsquare
                basey=directiony/dirrootsquare
                if not(objectID in listhistory):
                    listhistory.append(objectID)
                    

           
                to.centroids.append(centroid)
                """
                print("object :" + str(objectID))
                print("directionx :" + str(directionx))
                print("directiony :" + str(-1*directiony))
                """
                if not((np.isnan(basex)) or (np.isnan(basey))):
                    if(basex<0):
                        sumdirxmin+=basex
                    else:
                        sumdirxpos+=basex
                    if(basey<0) :
                        sumdirymin+=basey
                    else:
                        sumdirypos+=basey

                cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
            # check to see if the object has been counted or not
            #if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                #if direction < -10 and centroid[1] < 150:
                #    totalUp += 1
                #    to.counted = True

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                #elif direction > 10 and centroid[1] > 150:
                #    totalDown += 1
                #    to.counted = True
        
        #avg time

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        
        #canvas
        basevectorXmin.append(sumdirxmin)
        basevectorXpos.append(sumdirxpos)
        basevectorYmin.append(sumdirymin)
        basevectorYpos.append(sumdirypos)
        
        """
        print("vectorxmin :" + str(np.mean(basevectorXmin)))
        print("vectorxpos :" + str(np.mean(basevectorXpos)))
        print("vectorymin :" + str(np.mean(basevectorYmin)))
        print("vectorypos :" + str(np.mean(basevectorYpos)))"""
        #print("vectorX : " + str(np.mean(vectorX)))
        #print(vectorX)
        #print("vectorY : " + str(np.mean(vectorY)))
        #print(vectorY)
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        #text = "ID {}".format(objectID)
        #cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
        #   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        #cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


    # construct a tuple of information we will be displaying on the
    # frame
    if ((framecount)%skipframes)==0:
        frame8+=1
        #Average crowd
        cumperson+=countperson
        avgperson=cumperson/frame8
        avgperson=("{0:.2f}".format(avgperson))
        cumcar+=countcar
        avgcar=cumcar/frame8
        avgcar = ("{0:.2f}".format(avgcar))
        
        #canvas calculation
        if len(basevectorXmin)!=0:
            sumXvectormin=np.sum(basevectorXmin)
            sumXvectorpos=np.sum(basevectorXpos)
            sumYvectormin=np.sum(basevectorYmin)
            sumYvectorpos=np.sum(basevectorYpos)
            maxvalue=np.max([abs(sumXvectormin),sumXvectorpos,abs(sumYvectormin),sumYvectorpos])
            if(maxvalue!=0):
                sumXvectormin=(sumXvectormin/maxvalue)*const
                sumXvectorpos=(sumXvectorpos/maxvalue)*const
                sumYvectormin=(sumYvectormin/maxvalue)*const
                sumYvectorpos=(sumYvectorpos/maxvalue)*const
            else:
                sumXvectormin=0
                sumXvectorpos=0
                sumYvectormin=0
                sumYvectorpos=0
        
    #print("people count : "+ str(len(listhistory)))
    text = "People Density:      people/frame"
    cv2.putText(canvas, text, (10, 260),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    text="{}".format(countperson)
    cv2.putText(canvas, text, (135, 260),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    text = "Car Density:      car/frame"
    cv2.putText(canvas, text, (10, 285),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    text="{}".format(countcar)
    cv2.putText(canvas, text, (110, 285),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    text = "Count: "
    cv2.putText(canvas, text, (10, 235),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    text = "{}".format(len(listhistory))
    cv2.putText(canvas, text, (75, 235),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
     
    
    xmin=float(sumXvectormin)
    xpos=float(sumXvectorpos)
    ymin=float(sumYvectormin)
    ypos=float(sumYvectorpos)
    xminmap=xmin+150
    xposmap=xpos+150
    yminmap=-ymin+125
    yposmap=-ypos+125
    if (np.isnan(xminmap)):
        xminmap=150
    if (np.isnan(xposmap)):
        xposmap=150
    if (np.isnan(yminmap)):
        yminmap=125
    if (np.isnan(yposmap)):
        yposmap=125
        
    cv2.arrowedLine(canvas, (150, 125), (int(xminmap), 125), (255, 0, 0), 3, 8, 0, 0.1)
    cv2.putText(canvas, "{0:.2f}".format(xmin/const), (40, 117),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
    cv2.arrowedLine(canvas, (150, 125), (int(xposmap), 125), (255, 0, 0), 3, 8, 0, 0.1)
    cv2.putText(canvas, "{0:.2f}".format(xpos/const), (245, 117),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
    cv2.arrowedLine(canvas, (150, 125), (150, int(yminmap)), (255, 0, 0), 3, 8, 0, 0.1)
    cv2.putText(canvas, "{0:.2f}".format(ymin/const), (153, 225),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
    cv2.arrowedLine(canvas, (150, 125), (150, int(yposmap)), (255, 0,0), 3, 8, 0, 0.1)
    cv2.putText(canvas, "{0:.2f}".format(ypos/const), (153, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
    
    #cv2.arrowedLine(frame, (200, 258), (int(xmap), int(ymap)), (0, 0, 255), 3, 8, 0, 0.1)
    #cv2.arrowedLine(canvas, (200, 200), (int(xmap), int(ymap-58)), (0, 0, 255), 3, 8, 0, 0.1)
    # check to see if we should write the frame to disk
    # show the output frame
    cv2.imshow("Frame", frame)
    cv2.imshow("Vector",canvas)
    
    if ((framecount)%skipframes)==0:
        cv2.imwrite('send.jpg',frame)
        cv2.imwrite('vector.jpg',canvas)
        sendtotxt(avgperson,avgcar,xmin/const,xpos/const,ymin/const,ypos/const,len(listhistory))
    key = cv2.waitKey(1) & 0xFF
    # oper ke mainiot json format 
    # def sendtotxt(pup,pdo,ppl,car):
    framecount+=1
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    
   


# check to see if we need to release the video writer pointer

# if we are not using a video file, stop the camera video stream
cv2.destroyAllWindows()
videostream.stop()

