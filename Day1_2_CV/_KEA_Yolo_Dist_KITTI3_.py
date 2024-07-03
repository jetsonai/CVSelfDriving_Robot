#from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
import numpy as np
from sort import *
import math

gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)320, height=(int)240, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=gst_str,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="./yolo/yolov4-tiny.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./yolo/yolov4-tiny.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./yolo/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def video_capture(frame_queue, darknet_image_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_queue.put(frame)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue):
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image2(network, class_names, darknet_image, thresh=args.thresh)
        detections_queue.put(detections)
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        #print("FPS: {}".format(fps))
        #darknet.print_detections(detections, args.ext_output)
        darknet.free_image(darknet_image)
    cap.release()

def draw_boxes(detections, image, colors, frame_number):
    origin = np.copy(image)
    detection_index = 0
    for label, confidence, bbox in detections:
        left, top, right, bottom = darknet.bbox2points(bbox)

        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image
  
mot_tracker = Sort()
  
def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.out_filename, (video_width, video_height))
    frame_number = 0
    os.makedirs("./detected",exist_ok=True)
    while cap.isOpened():
        frame = frame_queue.get()
        fps = fps_queue.get()     
        detections = detections_queue.get()

        if frame is not None:
            detections_md = []
            i=0        
            for label, confidence, bbox in detections:

                detections_md.append([])
                bbox_adjusted = convert2original(frame, bbox)
                '''
                detections_md[i].append(bbox_adjusted[0])
                detections_md[i].append(bbox_adjusted[1])
                detections_md[i].append(bbox_adjusted[2])
                detections_md[i].append(bbox_adjusted[3])
                '''
                left, top, right, bottom = darknet.bbox2points(bbox_adjusted)
               
                detections_md[i].append(left)
                detections_md[i].append(top)
                detections_md[i].append(right)
                detections_md[i].append(bottom)  
                              
                detections_md[i].append(confidence)
                detections_md[i].append(label)
                i = i+1
            detect_num = len(detections_md)
            if(detect_num > 0) :
                arr = np.array(detections_md)
                
                detections = arr[(arr[:, 5] >= 0) & (arr[:, 5] <= 7)]
                track_bbs_ids = mot_tracker.update(detections)
                #print(len(track_bbs_ids.tolist()))
                for j in range(len(track_bbs_ids.tolist())):
                    coords = track_bbs_ids.tolist()[j]
                    x1, y1, x2, y2 = int(coords[0]), int(coords[1]),int(coords[2]),int(coords[3])
                    name_idx = int(coords[4])
                    name = "ID : {}".format(str(name_idx))

                    cv2.rectangle(frame,(x1,y1),(x2,y2),2)
                    cv2.putText(frame,name,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,2)
            cv2.imshow('Image',frame)   
            if cv2.waitKey(1) == 27:
                break  



        '''
        detections_adjusted = []
        if frame is not None:
            for label, confidence, bbox in detections:
                bbox_adjusted = convert2original(frame, bbox)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
            image = draw_boxes(detections_adjusted, frame, class_colors, frame_number)
            frame_number += 1
            if not args.dont_show:
                cv2.imshow('Inference', image)
         
            if args.out_filename is not None:
                video.write(image)
            if cv2.waitKey(1) == 27:
                break
         '''
         
    cap.release()
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = './test_img/video/KITTI_test11.mp4'
    #input_path = "../track.avi"
    cap = cv2.VideoCapture(input_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Thread(target=video_capture, daemon=True, args=(frame_queue, darknet_image_queue)).start()
    Thread(target=inference, daemon=True, args=(darknet_image_queue, detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()
    
    
#export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0:$LD_PRELOAD
#rm ~/.cache/gstreamer-1.0/ -fr    
# sudo service nvargus-daemon restart
