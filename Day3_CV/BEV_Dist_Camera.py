import cv2
import numpy as np
import sys

maxP = 18
coord_x = 159       
coord_y = 150

def mouse_callback(event, x, y, flags, param): 
    global coord_x	
    global coord_y   
    global maxP
    
    if event == cv2.EVENT_LBUTTONDOWN :
        print("mouse click, x:", x ," y:", y)
        coord_x = y
        coord_y = x       

def Dist(src,dst):
    Homo,mask = cv2.findHomography(src,dst,cv2.RANSAC)
    return Homo


'''
mouse click, x: 205  y: 285
mouse click, x: 253  y: 286
mouse click, x: 304  y: 290
mouse click, x: 355  y: 292
mouse click, x: 404  y: 293
mouse click, x: 446  y: 293

mouse click, x: 179  y: 360
mouse click, x: 239  y: 365
mouse click, x: 305  y: 368
mouse click, x: 370  y: 369
mouse click, x: 429  y: 369
mouse click, x: 484  y: 365

mouse click, x: 161  y: 417
mouse click, x: 231  y: 422
mouse click, x: 304  y: 429
mouse click, x: 380  y: 428
mouse click, x: 447  y: 424
mouse click, x: 507  y: 416


'''
maxP = 18
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


def pers_transform(image):
    # Grab the image shape
    image_size = (image.shape[1],   image.shape[0])
    src = np.float32([[80, 187], [101, 121], [223, 125], [254, 186]])
    offset = [0,0]
    dst = np.float32([src[0] + offset, np.array([src[0, 0], 0]) + offset, 
                      np.array([src[3, 0], 0]) - offset, src[3] - offset]) 
                      
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(image, M, image_size)
    # Return the resulting image and matrix
    Minv = cv2.getPerspectiveTransform(dst, src)

    return warped, M, Minv
    
def process_img(img):
 
    # Perspective transform image
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    warped, M, Minv = pers_transform(img)
    return warped
    
def draw_image_dist(image):
    global Homo
    global src

    #for label, confidence, bbox in detections:
    #left, top, right, bottom = bbox2points(bbox)
    #cx, cy, w, h = bbox
    #cbX = cx
    #cbY = bottom
    cbX = coord_x
    cbY = coord_y
            
    img_center = np.array([cbX, cbY,1])
    V_Center = np.dot(Homo, img_center)
    X_dist = V_Center[0]/V_Center[2]
    Y_dist = V_Center[1]/V_Center[2]
    
    for sc in src:
        #print("{} {}".format(sc[1], sc[0]))
        cv2.circle(image, (int(sc[1]), int(sc[0])), 5, (0, 255, 0), -1)
    cv2.circle(image, (cbY, cbX), 5, (0, 0, 255), -1)

    V_Center_text = "X: {:.2f} Y: {:.2f}".format( X_dist, Y_dist)
    #print(V_Center_text)
    
    cv2.putText(image, V_Center_text, (cbY + 10, cbX - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
      
    return image    

def Video(openpath):
    global coord_x	
    global coord_y   
    global maxP
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
    
                frame = draw_image_dist(frame)
                #for cn in range(0, maxP):
                #   cv2.circle(frame, (int(src[cn]), int(src[cn])), 5, (0, 255, 0), -1)
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
    
gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")     
   
if __name__=="__main__":
    Video(gst_str)
