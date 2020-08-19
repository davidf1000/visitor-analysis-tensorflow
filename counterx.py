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
# from adrian roseberg tracking lib 
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS

# Threading to increase FPS
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/

#oper data ke file json 
def sendtotxt(pup,pdo,ppl,car):
    data='''
    {
    "crowdcar":0,
    "crowdppl":0,
    "peopleup":0,
    "peopledown":0
    }'''
    datajson=json.loads(data)
    datajson["crowdcar"]=car
    datajson["crowdppl"]=ppl
    with open('jsonfile.json','w+') as f:
        json.dump(datajson,f,sort_keys=True)


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
min_conf_threshold = 0.45
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
skipframes=8
#rata - rata pixel 255/2
input_mean = 127.5
input_std = 127.5

### INITIALIZE TRACKER
# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=12, maxDistance=50)
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
# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)
#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

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
            # filter out weak detections by requiring a minimum
            # confidence
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)) :
#                 if (condperson):
#                     object_name = "person" # Look up object name from "labels" array using class index
#                     countperson+=1
#                 else:
#                     object_name = "car"
#                     countcar+=1
                # compute the (x, y)-coordinates of the bounding box
                # for the object
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(int(xmin), int(ymin), int(xmax), int(ymax))
                tracker.start_track(frame_rgb, rect)
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 255, 255), 2)
                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                object_name = labels[int(classes[i])]
                #trackers.append(tracker)
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
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

    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    cv2.line(frame, (200, 300), (200, 0), (0, 255, 255), 2)
    
    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)
    #print(ct.nextObjectID)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

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
            x = [c[0] for c in to.centroids]
            
            direction = centroid[0] - np.mean(x)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
#             if not to.counted:
#                 # if the direction is negative (indicating the object
#                 # is moving up) AND the centroid is above the center
#                 # line, count the object
#                 if direction < -10 and centroid[1] < 150:
#                     totalUp += 1
#                     to.counted = True
# 
#                 # if the direction is positive (indicating the object
#                 # is moving down) AND the centroid is below the
#                 # center line, count the object
#                 elif direction > 10 and centroid[1] > 150:
#                     totalDown += 1
#                     to.counted = True

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        #text = "ID {}".format(objectID)
        #cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
        #   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # construct a tuple of information we will be displaying on the
    # frame
#     if ((framecount)%skipframes)==0:
#         frame8+=1
#         cumperson+=countperson
#         avgperson=cumperson/frame8
#         avgperson=("{0:.2f}".format(avgperson))
#         cumcar+=countcar
#         avgcar=cumcar/frame8
#         avgcar = ("{0:.2f}".format(avgcar))
#         sendtotxt(totalUp,totalDown,avgperson,avgcar)  
#     info = [
#     ("Up", totalUp),
#     ("Down", totalDown),
#     ("crowdppl",avgperson),
#     ("crowdcar",avgcar),
#     ("Status", status),
#     ]
#     # loop over the info tuples and draw them on our frame
#     for (i, (k, v)) in enumerate(info):
#         text = "{}: {}".format(k, v)
#         cv2.putText(frame, text, (10, 300 - ((i * 20) + 20)),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # check to see if we should write the frame to disk
    # show the output frame
    cv2.imshow("Frame", frame)
    if ((framecount)%skipframes)==0:
        out = cv2.imwrite('send.jpg',frame)
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


