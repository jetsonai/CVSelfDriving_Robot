#!/usr/bin/env python3

# Copyright 2024 JetsonAI CO., LTD.
#
# Author: Kate Kim

import rclpy 
from rclpy.node import Node 
import cv2, time
import numpy as np
from ros_yolo_pack.darknet import *
from sensor_msgs.msg import Image 
import cv2 
from cv_bridge import CvBridge, CvBridgeError

gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

packet_path = "/home/nvidia/ros_app_ws/src/ros_yolo_pack/"

data_path = packet_path + "yolo/challenge.mp4"
coco_path = packet_path +  "/yolo/coco.names"
cfg_path = packet_path +  "/yolo/yolov4-tiny.cfg"
weight_path = packet_path +  "/yolo/yolov4-tiny.weights"

def Dist(src,dst):
    Homo,mask = cv2.findHomography(src,dst,cv2.RANSAC)
    return Homo

src = np.float32([[94,90],[114, 86], [138, 82], [163, 79]])
dst = np.float32([[9,5],[16.5,5],[16.5,-7.5],[9,-5]])

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

colours = np.random.rand(32, 3)*255 #used only for display
prev_X_dist = {}
prev_time = time.time()

def draw_boxes_dist(detections, image, colors):
    global Homo
    global prev_time

    current_time = time.time()
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        cx, cy, w, h = bbox
        
        img_center = np.array([cx, cy,1])
        V_Center = np.dot(Homo, img_center)
        X_dist = V_Center[0]/V_Center[2]
        Y_dist = V_Center[1]/V_Center[2]      

        # Calculate speed and TTC if we have a previous X_dist for this ID
        if name_idx in prev_X_dist and -5 <= Y_dist <= 5:
            if name_idx in prev_X_dist:
                prev_dist = prev_X_dist[name_idx]
                time_between_frames = current_time - prev_time  # 프레임 사이의 실제 시간 계산
                speed = abs(X_dist - prev_dist)/time_between_frames  # Speed in X direction
                TTC = X_dist / speed if speed != 0 else float('inf')
                print("TTC for ID {}: {}".format(name_idx, TTC))

        prev_X_dist[name_idx] = X_dist 
        '''
        V_Center_text = "{} X: {:.2f} Y: {:.2f}".format(label, X_dist, Y_dist)
        
        #lane_center = int(lane_center)
        #lane_center_y = int(lane_center_y)
        cv2.putText(image, V_Center_text, (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
      
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        #cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
        #            (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #            colors[label], 2)
    prev_time = current_time  # 현재 시간을 이전 시간으로 설정
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
    for label, confidence, bbox in detections:
        bbox_adjusted = convert2original(frame, bbox, darknet_width, darknet_height)
        detections_adjusted.append((str(label), confidence, bbox_adjusted))

    free_image(darknet_image)
    #image = draw_boxes(detections_adjusted, frame, class_colors)
    image = draw_boxes_dist(detections_adjusted, frame, class_colors)
                    
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
    node = rclpy.create_node("ros_yolo_node2_dist")
    
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
