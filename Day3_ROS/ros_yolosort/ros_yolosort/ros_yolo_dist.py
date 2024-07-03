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


gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")  

packet_path = "/home/nvidia/cv_selfdriving_ws/src/ros_yolosort/"

data_path = packet_path + "yolo/challenge.mp4"
coco_path = packet_path +  "/yolo/coco.names"
cfg_path = packet_path +  "/yolo/yolov4-tiny.cfg"
weight_path = packet_path +  "/yolo/yolov4-tiny.weights"

#colors_idx = darknet.class_colors_idx()


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
  
    
    
def draw_boxes_dist(detections, image, colors):
    global Homo
    global colors_idx

    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        cx, cy, w, h = bbox
        cbX = bottom
        cbY = cx 
        
        #print("label:{}".format(label))
        if label == "car":
        
            img_center = np.array([cbX, cbY,1])
            V_Center = np.dot(Homo, img_center)
            X_dist = V_Center[0]/V_Center[2]
            Y_dist = V_Center[1]/V_Center[2]

            V_Center_text = "{} X: {:.2f} Y: {:.2f}".format(label, X_dist, Y_dist)
            
            cv2.putText(image, V_Center_text, (left + 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
          
            cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
            #cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
            #            (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #            colors[label], 2)
    return image


def image_detection2(frame, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    #darknet_width = darknet.network_width(network)
    #darknet_height = darknet.network_height(network)
    #darknet_image = darknet.make_image(darknet_width, darknet_height, 3)
    darknet_width = network_width(network)
    darknet_height = network_height(network)
    darknet_image = make_image(darknet_width, darknet_height, 3)
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (darknet_width, darknet_height),
                               interpolation=cv2.INTER_LINEAR)

    #darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    #detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = detect_image(network, class_names, darknet_image, thresh=thresh)

    detections_adjusted = []
    i=0
    for label, confidence, bbox in detections:
        bbox_adjusted = convert2original(frame, bbox, darknet_width, darknet_height)
        detections_adjusted.append((str(label), confidence, bbox_adjusted))
                                          
    #image = draw_boxes(detections_adjusted, frame, class_colors)
    image = draw_boxes_dist(detections_adjusted, frame, class_colors)
    
    free_image(darknet_image)
    return image



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
    node = rclpy.create_node("ros_yolo_node")
    
    
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

    #network, class_names, class_colors = darknet.load_network3(
    network, class_names, class_colors = load_network3(
            cfg_path, coco_path,
            weight_path, batch_size=1)

    thresh=.25    

    while(True):
        # Capture frame-by-frame
        ret, cv_image = cap.read()
        
        output= image_detection2(cv_image, network, class_names, class_colors, thresh)

        # Display the resulting frame
        cv2.imshow('frame',cv_image)
        
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
