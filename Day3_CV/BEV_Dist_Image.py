import cv2
import numpy as np
import sys

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
    cbX = 152
    cbY = 139
            
    img_center = np.array([cbX, cbY,1])
    V_Center = np.dot(Homo, img_center)
    X_dist = V_Center[0]/V_Center[2]
    Y_dist = V_Center[1]/V_Center[2]
    
    cv2.circle(image, (cbX, cbY), 5, (255, 255, 0), -1)
    for cn in range(0, maxP):
        print("{} {}".format(src[cn][1], src[cn][0]))
        cv2.circle(image, (int(src[cn][1]), int(src[cn][0])), 5, (0, 255, 0), -1)

    V_Center_text = "X: {:.2f} Y: {:.2f}".format( X_dist, Y_dist)
    print(V_Center_text)
    
    cv2.putText(image, V_Center_text, (cbX + 10, cbY - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
      
    return image    

inputfile_path = sys.argv[1]
print("inputfile_path:{}".format(inputfile_path))

img = cv2.imread(inputfile_path)   

result = process_img(img)
image = draw_image_dist(img)
cv2.imshow('Dist', image)    
#cv2.imshow('BEV image', result)

cv2.waitKey() 

cv2.destroyAllWindows()
