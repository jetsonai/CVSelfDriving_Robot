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

#gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)320, height=(int)240, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)320, height=(int)240, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

packet_path = "/home/nvidia/cv_selfdriving_ws/src/ros_yolosort/"

data_path = packet_path + "yolo/challenge.mp4"
coco_path = packet_path +  "/yolo/coco.names"
cfg_path = packet_path +  "/yolo/yolov4-tiny.cfg"
weight_path = packet_path +  "/yolo/yolov4-tiny.weights"


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
    for label, confidence, bbox in detections:
        bbox_adjusted = convert2original(frame, bbox, darknet_width, darknet_height)
        detections_adjusted.append((str(label), confidence, bbox_adjusted))

    #darknet.free_image(darknet_image)
    #image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
    free_image(darknet_image)
    image = draw_boxes(detections_adjusted, frame, class_colors)
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
    input_path = gst_str
    cap = cv2.VideoCapture(input_path)

    if cap.isOpened():
        print("Video Opened")
    else:
        print("Video Not Opened")
        print("Program Abort")
        exit()
    
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
