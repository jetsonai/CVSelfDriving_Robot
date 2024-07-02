gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")  

import cv2
import numpy as np

clicCnt = 0
maxP = 18
coord_x = np.zeros(maxP)        
coord_y = np.zeros(maxP) 

def mouse_callback(event, x, y, flags, param): 
    global coord_x	
    global coord_y   
    global clicCnt
    
    if event == cv2.EVENT_LBUTTONDOWN :
        print("mouse click, x:", x ," y:", y)

        if(clicCnt < maxP):
            coord_x[clicCnt] = x		
            coord_y[clicCnt] = y
            clicCnt = clicCnt+1
        
def Video(openpath):
    global coord_x	
    global coord_y   
    global clicCnt
    cap = cv2.VideoCapture(openpath)
    if cap.isOpened():
        print("Video Opened")
    else:
        print("Video Not Opened")
        print("Program Abort")
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    #fourcc = cv2.VideoWriter_fourcc('m','p','4','v') with *.mp4 save

    cv2.namedWindow("Input", cv2.WINDOW_GUI_NORMAL)
    #cv2.resizeWindow("Input", 1280, 760)
    cv2.setMouseCallback('Input', mouse_callback)

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if ret:
                for cn in range(0, clicCnt):
                    cv2.circle(frame, (int(coord_x[cn]), int(coord_y[cn])), 5, (0, 255, 0), -1)
                cv2.imshow("Input", frame)

            else:
                break
            # waitKey(int(1000.0/fps)) for matching fps of video
            if cv2.waitKey(int(1000.0/fps)) & 0xFF == ord('q'):
                break
        except KeyboardInterrupt:  
            print("key int")  
            break  
    # When everything done, release the capture
    cap.release()

    cv2.destroyAllWindows()
    return
   
if __name__=="__main__":
    Video(gst_str)
