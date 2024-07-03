#!/usr/bin/env python3

# Copyright 2024 JetsonAI CO., LTD.
#
# Author: Kate Kim

import rclpy 
from rclpy.node import Node 
import cv2, time
import numpy as np
from ros_yolosort.darknet import *
from sensor_msgs.msg import Image 
import cv2 
from cv_bridge import CvBridge, CvBridgeError
from queue import Queue
import numpy as np
from ros_yolosort.sort import *
import math


gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

packet_path = "/home/nvidia/cv_selfdriving_ws/src/ros_yolosort/"

data_path = packet_path + "yolo/challenge.mp4"
coco_path = packet_path +  "/yolo/coco.names"
cfg_path = packet_path +  "/yolo/yolov4-tiny.cfg"
weight_path = packet_path +  "/yolo/yolov4-tiny.weights"

class_colors_num = class_colors_idx()   

def Dist(src,dst):
    Homo,mask = cv2.findHomography(src,dst,cv2.RANSAC)
    return Homo

######################  
### 320 x 240 ###
#src = np.float32([[ 119, 102],[ 119, 124],[ 121, 150], [ 121, 175], [ 121, 199],  [ 122, 220],
#                  [ 155,  88],[ 155, 119],[ 159, 150], [ 159, 183], [ 159, 213],  [ 159, 240],
#                  [ 183,  77],[ 185, 113],[ 188, 151], [ 189, 188], [ 187, 222],  [ 186, 253],
#                ])
######################                
### 640 x 480 ###
src = np.float32([[ 285, 205],[ 286, 253],[ 290, 304], [ 292, 355], [ 293, 404],  [ 293, 446],
                  [ 360, 179],[ 365, 239],[ 368, 305], [ 369, 370], [ 369, 429],  [ 365, 484],
                  [ 417, 161],[ 422, 231],[ 429, 304], [ 428, 380], [ 424, 447],  [ 416, 507],
                ])
######################                  
dst = np.float32([[16.8, 4.8],[16.8, 2.4],[16.8, 0], [16.8, -2.4], [16.8, -4.8],  [16.8, -7.2],
                  [12.0, 4.8],[12.0, 2.4],[12.0, 0], [12.0, -2.4], [12.0, -4.8],  [12.0, -7.2],
                  [ 9.6, 4.8],[ 9.6, 2.4],[ 9.6, 0], [ 9.6, -2.4], [ 9.6, -4.8],  [ 9.6, -7.2],
                ])
       
Homo = Dist(src,dst)
print("Homo====")
print(Homo)

colors_idx = class_colors_idx()

def convert2relative(bbox, darknet_width, darknet_height):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox

    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height

def convert2original(image, bbox, darknet_width, darknet_height):
    x, y, w, h = convert2relative(bbox, darknet_width, darknet_height)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted
  
    
    
def draw_boxes_dist(detections, image):
    global Homo
    global colors_idx

    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        cx, cy, w, h = bbox
        cbX = cx
        cbY = bottom
        
        img_center = np.array([cbX, cbY,1])
        V_Center = np.dot(Homo, img_center)
        X_dist = V_Center[0]/V_Center[2]
        Y_dist = V_Center[1]/V_Center[2]

        V_Center_text = "{} X: {:.2f} Y: {:.2f}".format(label, X_dist, Y_dist)
        
        cv2.putText(image, V_Center_text, (left + 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
      
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        #cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
        #            (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #            colors[label], 2)
    return image

##################################           
  
mot_tracker = Sort()

def class_to_label(x, class_names):
    # x 숫자 레이블 -> 문자열 레이블로 반환
    #classes = class_names
    #print(len(classes))
    idx = int(x) % len(class_names)
    #print(idx)
    return class_names[idx]

def ID_to_color(x):
    global class_colors_num
    # x 숫자 레이블 -> 문자열 레이블로 반환
    idx = int(x) % 80
    #print("---ID_to_color: idx {} len {}".format(idx, 80))
    color = class_colors_num[idx]
    return color
   
prev_timev = time.time()     
prev_X_dist = {}     
#################################   

def image_detection2(frame, network, class_names, class_colors, thresh):
    global Homo
    global prev_timev
    global prev_X_dist
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    darknet_width = network_width(network)
    darknet_height = network_height(network)
    darknet_image = make_image(darknet_width, darknet_height, 3)
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (darknet_width, darknet_height),
                               interpolation=cv2.INTER_LINEAR)

    copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = detect_image2(network, class_names, darknet_image, thresh=thresh)
    
    current_time = time.time() ###

    detections_adjusted = []
    detections_md = []
    i=0
    for label, confidence, bbox in detections:
        if(label>= 0) & (label <= 7):
            bbox_adjusted = convert2original(frame, bbox, darknet_width, darknet_height)
            detections_adjusted.append((str(label), confidence, bbox_adjusted))

            detections_md.append([])
            left, top, right, bottom = bbox2points(bbox_adjusted)
           
            detections_md[i].append(left)
            detections_md[i].append(top)
            detections_md[i].append(right)
            detections_md[i].append(bottom)  
                          
            detections_md[i].append(confidence)
            detections_md[i].append(label)
            i = i+1
    detect_num = len(detections_md)
    if(detect_num > 0) :
        detections = np.array(detections_md)
        
        #print(detections)
      
        #detections = arr[(clnum >= 0) & (clnum <= 7)]
        track_bbs_ids = mot_tracker.update(detections)
        for j in range(len(track_bbs_ids.tolist())):
            coords = track_bbs_ids.tolist()[j]
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]),int(coords[2]),int(coords[3])
            class_idx = detections[j,5]
            name = "ID : {}".format(str(class_idx))
            
            track_ID = int(coords[4])
            name = "ID : {}".format(str(track_ID))

            cv2.rectangle(frame,(x1,y1),(x2,y2),2)

            clabel = class_to_label(class_idx, class_names)
            #print("class_idx {} clabel {}".format(class_idx, clabel))
            color = ID_to_color(track_ID) 
           
            cbX = y2
            cbY = (x2-x1)/2
                    
            img_center = np.array([cbX, cbY,1])
            V_Center = np.dot(Homo, img_center)
            X_dist = V_Center[0]/V_Center[2]
            Y_dist = V_Center[1]/V_Center[2]
            
            #cv2.circle(image, (cbY, cbX), 5, (0, 0, 255), -1)

            V_Center_text1 = "{} : {} ".format(clabel,track_ID,)
            V_Center_text2 = "X: {:.2f} Y: {:.2f}".format(X_dist, Y_dist)
            #print(V_Center_text)
            
            #cv2.putText(image, V_Center_text, (x1+10, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            
            #cv2.putText(frame, clabel
            #+ ': ' + "{:.1f}".format(X_dist) + ', ' + "{:.1f}".format(Y_dist),
            #(x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)   
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,V_Center_text1,(x1+10, y1-40),cv2.FONT_HERSHEY_SIMPLEX,0.5,color, 2)
            cv2.putText(frame,V_Center_text2,(x1+10, y1-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,color, 2)                 
            ###################
            '''
            TTC = Dist/RelativeSpeed
            RelativeSpeed 두물체의 속도 차이 ( |자차의 속도 - 타겟 차량의 속도|)
            내차의 속도 : 20m/s
            상대차의 속도 : 10m/s
            거리 : 100m
            TTC = 100m / 10m/s = 10s (충돌까지의 시간)
            '''
            if track_ID in prev_X_dist and -15 <= Y_dist <= 15:
                if track_ID in prev_X_dist:
                    prev_dist = prev_X_dist[track_ID]
                    time_between_frames = current_time - prev_timev  # 프레임 사이의 실제 시간 계산
                    speed = abs(X_dist - prev_dist)/time_between_frames  # Speed in X direction
                    TTC = X_dist / speed if speed != 0 else float('inf')
                    V_Center_text3 = "TTC: {:.2f}".format(TTC)
                    print(TTC)
                    cv2.putText(frame,V_Center_text3,(x1+10, y1-3),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255), 2)

            # Update the X_dist for this ID
            prev_X_dist[track_ID] = X_dist
            ###################                        

    #image = draw_boxes(detections_adjusted, frame, class_colors)
    #image = draw_boxes_dist(detections_adjusted, frame, class_colors)
    prev_timev = current_time  
    free_image(darknet_image)
    return frame



def model_detect(img):
    classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)

    print(len(classIds))
    print(classIds)
    print(scores)

    for (classId, score, box) in zip(classIds, scores, boxes):
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
        color=(0, 255, 0), thickness=2)
        #text = '%s: %.2f' % (classes[classId[0]], score)
        text = '%s: %.2f' % (classes[classId], score)
        cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(0, 255, 0), thickness=2)

    return img

    
def main(args=None):
  
    rclpy.init()
    node = rclpy.create_node("ros_yolo_TTC")
    
    
    print(coco_path)

    global bridge
    bridge = CvBridge()

    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(gst_str)
    if not (cap.isOpened()):
        print("Could not open video device")
    # To set the resolution
    
    width = 640
    height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    print('width:{} height:{}'.format(width,height))

    network, class_names, class_colors = load_network3(
            cfg_path, coco_path,
            weight_path, batch_size=1)

    thresh=.25 
    #class_colors_num = class_colors_idx()   

    while(True):
        # Capture frame-by-frame
        ret, cv_image = cap.read()
        
        output= image_detection2(
                cv_image, network, class_names, class_colors, thresh)

        # Display the resulting frame
        cv2.imshow('frame',output)
        
        #print('view video frame')
        # Waits for a user input to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    node.destroy_node()
    rclpy.shutdown()
  
if __name__ == '__main__':
  main()
