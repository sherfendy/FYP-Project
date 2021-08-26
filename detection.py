import cv2
import os
import math
#import random
#import numpy as np
import time
import datetime
from itertools import combinations

def is_close(p1,p2):
    dist = math.sqrt(p1**2 + p2**2)
    return dist

def convertBack(x, y, w, h):
    xcent = int(round(x + (w / 2)))
    ycent = int(round(y + (h / 2)))
    return xcent, ycent

def drawDetections(detections, img):
    if len(detections[1]) > 0:
        centroid_dict = dict()
        objectId = 0
        for objectId in range(len(detections[2])):
            xmin, ymin, w, h = detections[2][objectId][0],\
                               detections[2][objectId][1],\
                               detections[2][objectId][2],\
                               detections[2][objectId][3]
            
            xcent, ycent = convertBack(float(xmin), float(ymin), float(w), float(h))
            centroid_dict[objectId] = (xcent, ycent, xmin, ymin, w, h)
            objectId += 1
    
    red_zone_list = []
    red_line_list = []
    for (id1, p1), (id2, p2) in combinations (centroid_dict.items(),2):
        dy, dx = p1[0] - p2[0], p1[1] - p2[1]
        distance = is_close(dy, dx)
        if distance < 80.0:
            if id1 not in red_zone_list:
                red_zone_list.append(id1)
                red_line_list.append(p2[0:2])
            if id2 not in red_zone_list:
                red_zone_list.append(id2)
                red_line_list.append(p2[0:2])
    
    for idx, box in centroid_dict.items():
        if idx in red_zone_list:
            cv2.rectangle(img, (box[2], box[3], box[4], box[5]), (0, 0, 255), 2)
        else:
            cv2.rectangle(img, (box[2], box[3], box[4], box[5]), (0, 255, 0), 2)
    
    risk = int(len(red_zone_list)/(len(detections[2]))*100)
    
    text = "Risk Percentage: {0}%".format(str(risk))
    
    img, text
    font=cv2.FONT_HERSHEY_SIMPLEX
    location = (10, img.shape[0] - 30)
    font_scale=1
    font_thickness=2
    text_color_bg=(0, 0, 0)
    
    if risk >= 60:
        text_color= (0, 0, 255)
    else:
        text_color = (0, 255, 0)
    

    x, y = location
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, location, (x + text_w + 3, y + text_h + 3), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    
    if risk >= 60 and len(detections[2]) > 10:
        text_alert = "Assistance needed"
        location_alert = (10, img.shape[0] - 60)
        x, y = location_alert
        cv2.rectangle(img, location_alert, (x + text_w + 3, y + text_h + 3), text_color_bg, -1)
        cv2.putText(img, text_alert, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

   # cv2.putText(img, text, location, font, font_scale, text_color, font_thickness, cv2.LINE_AA)  #bgr

    
    for check in range(0, len(red_zone_list)-1):
        start_point = red_line_list[check]
        end_point = red_line_list[check+1] 
        check_line_x = abs(end_point[0] - start_point[0])
        check_line_y = abs(end_point[0] - start_point[0]) 
        #if(check_line_x < 75) and (check_line_y < 25):
            #cv2.line(img, start_point, end_point, (0, 0, 255), 2)
    
    return img, risk


############################### Start Execution #######################################
net = cv2.dnn_DetectionModel('./yolov4-obj.cfg', './yolov4-obj_best.weights')
net.setInputSize(704, 704)
net.setInputScale(1.0/ 255)
net.setInputSwapRB(True)

f = open('./obj.names', 'r')
names = f.read()


# img = cv2.imread('detect/cctv.jpg')
#img = cv2.VideoCapture("detect/testvid.mp4")

# classes, confidences, boxes = net.detect(img, confThreshold = 0.1, nmsThreshold = 0.3)

# detections = net.detect(img, confThreshold = 0.1, nmsThreshold = 0.5)
# # detections[0] = class
# # detections[1] = confidence
# # detections[2] = xmin,ymin,w,h
# img = drawDetections(detections, img)
# cv2.imshow('detect', img)
# cv2.waitKey(0)

################################## For Video #################################################
#cap = cv2.VideoCapture(0)  # webcam
cap = cv2.VideoCapture("./testvid.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# count the number of frames
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = int(cap.get(cv2.CAP_PROP_FPS))
  
# calculate dusration of the video
seconds = int(frames / fps)
video_time = str(datetime.timedelta(seconds=seconds))

print("duration in seconds:", seconds)
#print("video time:", video_time)

#new_height, new_width = frame_height // 2, frame_width // 2
#print("Video Resolution: ",(frame_width, frame_height))

# create detection video in
out = cv2.VideoWriter("./test_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,(frame_width, frame_height))

risk_cap = []
frame_cap = []
count_frame = 0

# print("Starting the YOLO loop...")
while True:
    
    prev_time = time.time()
    ret, frame_read = cap.read()
    # Check if frame present :: 'ret' returns True if frame present, otherwise break the loop.
    if not ret:
        break
    
    count_frame = count_frame + 1
    frame_cap.append(count_frame)
    detections = net.detect(frame_read, confThreshold = 0.1, nmsThreshold = 0.5)
    img, risk = drawDetections(detections, frame_read)
    risk_cap.append(risk)
    print("frame:", count_frame)    # frame number
    print(1/(time.time()-prev_time))    # print fps
    cv2.imshow('Demo', img)
    cv2.waitKey(3)
    out.write(img)

cap.release()
out.release()
print(":::Video Write Completed")

# Display risk trend graph
import matplotlib.pyplot as plt
   
plt.plot(frame_cap, risk_cap)
plt.title('Risk Trend')
plt.xlabel('Frame')
plt.ylabel('Risk Percentage (%)')
plt.show()