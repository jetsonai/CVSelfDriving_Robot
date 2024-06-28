import cv2
import numpy as np

import glob

import rclpy 
from rclpy.node import Node 
from cv_bridge import CvBridge, CvBridgeError

from custom_msgctl.msg import MsgCtl

from collections import deque

class Line():
    def __init__(self, maxSamples=4):
        
        self.maxSamples = maxSamples 
        # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=self.maxSamples)
        # Polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        # Polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        # Average x values of the fitted line over the last n iterations
        self.bestx = None
        # Was the line detected in the last iteration?
        self.detected = False 
        # Radius of curvature of the line in some units
        self.radius_of_curvature = None 
        # Distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #-----------------------------
        self.last_best_fit = None
        self.last_current_fit = None
        self.last_recent_xfitted = deque(maxlen=self.maxSamples)

         
    def update_lane(self, ally, allx):
        # Updates lanes on every new frame
        # Mean x value 
        self.bestx = np.mean(allx, axis=0)
        # Fit 2nd order polynomial
        new_fit = np.polyfit(ally, allx, 2)
        # Update current fit
        self.current_fit = new_fit
        # Add the new fit to the queue
        self.recent_xfitted.append(self.current_fit)
        # Use the queue mean as the best fit
        self.best_fit = np.mean(self.recent_xfitted, axis=0)
        # meters per pixel in y dimension
        #ym_per_pix = 30/720
        # meters per pixel in x dimension
        #xm_per_pix = 3.7/700
        # Calculate radius of curvature
        #fit_cr = np.polyfit(ally*ym_per_pix, allx*xm_per_pix, 2)
        #y_eval = np.max(ally)
        #self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    
    def last_lane_update(self,last_fit):
        last_new_fit = np.polyfit(last_fit[0], last_fit[1], 2)
        self.last_current_fit = last_new_fit
        self.last_recent_xfitted.append(self.last_current_fit)
        self.last_best_fit = np.mean(self.last_recent_xfitted, axis=0)
        
left_line = Line()
right_line = Line()

line = Line()

global left_fit, right_fit
left_fit = None
right_fit = None

def validate_lane_update(img, left_lane_inds, right_lane_inds):
    if line.detected == False:
        print("Validate Update X No lane detected")
        return
    # Checks if detected lanes are good enough before updating
    img_size = (img.shape[1], img.shape[0])#width, height
    
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Extract left and right line pixel positions
    left_line_allx = nonzerox[left_lane_inds]
    left_line_ally = nonzeroy[left_lane_inds] 
    right_line_allx = nonzerox[right_lane_inds]
    right_line_ally = nonzeroy[right_lane_inds]

    
    # Discard lane detections that have very little points, 
    # as they tend to have unstable results in most cases
    if len(left_line_allx) <= 300 or len(right_line_allx) <= 300:
        print("Discarding detections points")
        left_line.detected = False
        right_line.detected = False
        #print("Update X")
        return
    
    left_x_mean = np.mean(left_line_allx, axis=0)
    right_x_mean = np.mean(right_line_allx, axis=0)
    lane_width = np.subtract(right_x_mean, left_x_mean)
    # Fit a second order polynomial to each lane line
    left_fit = np.polyfit(left_line_ally, left_line_allx, 2)
    right_fit = np.polyfit(right_line_ally, right_line_allx, 2)
    max_y = img.shape[0]  # y is maximum at the bottom of the image
    right_x_at_max_y = right_fit[0]*max_y**2 + right_fit[1]*max_y + right_fit[2]
    left_x_at_max_y = left_fit[0]*max_y**2 + left_fit[1]*max_y + left_fit[2]
    #print("Validate_right_x_at_max_y",right_x_at_max_y)
    
    if right_x_at_max_y < 160 or left_x_at_max_y > 160:
        print("lane width error!!!")
        left_line.detected = False
        right_line.detected = False
        return

    # Discard the detections if lanes are not in their repective half of their screens
    if left_x_mean > 160 or right_x_mean < 160:
        
        #print("Discarding detections screens")
        left_line.detected = False
        right_line.detected = False
        #print("Update X")
    
        return
    
    
    # Discard the detections if the lane width is too large or too small
    if  lane_width < 100 or lane_width > 320:
        #print("Discarding detections lane width")
        left_line.detected = False
        right_line.detected = False
        #print("Update X")
       
        return 
    
        
    # If this is the first detection or 
    # the detection is within the margin of the averaged n last lines 
    if left_line.bestx is None or np.abs(np.subtract(left_line.bestx, np.mean(left_line_allx, axis=0))) < 100:
        #print("left_line_update")
        left_line.update_lane(left_line_ally, left_line_allx)
        left_line.detected = True
    else:
        left_line.detected = False
    if right_line.bestx is None or np.abs(np.subtract(right_line.bestx, np.mean(right_line_allx, axis=0))) < 100:
        #print("right_line_update")
        right_line.update_lane(right_line_ally, right_line_allx)
        right_line.detected = True
    else:
        right_line.detected = False 


    # Calculate vehicle-lane offset
    xm_per_pix = 3.7/610 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    car_position = img_size[0]/2
    l_fit = left_line.current_fit
    r_fit = right_line.current_fit
    left_lane_base_pos = l_fit[0]*img_size[1]**2 + l_fit[1]*img_size[1] + l_fit[2]
    right_lane_base_pos = r_fit[0]*img_size[1]**2 + r_fit[1]*img_size[1] + r_fit[2]
    lane_center_position = (left_lane_base_pos + right_lane_base_pos) /2
    left_line.line_base_pos = (car_position - lane_center_position) * xm_per_pix +0.2
    right_line.line_base_pos = left_line.line_base_pos


### WARP
def pers_transform(image):
    # Grab the image shape
    image_size = (image.shape[1], image.shape[0])
    '''
    src = np.float32([[7, 98], [60, 20], [212, 16], [270,94]])

    dst = np.float32([[57, 98], [57, 20], [263, 16], [263,94]])
    '''
    src = np.float32([[6, 102], [60, 27], [216, 22], [272, 99]])
    offset = [50,0]
    dst = np.float32([[55, 102], [55, 27], [260, 27], [260, 102]]) 
 

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(image, M, image_size)
    # Return the resulting image and matrix
    Minv = cv2.getPerspectiveTransform(dst, src)

    return warped, M, Minv

### threshold for hls
def hls_thresh(img, thresh_min=0, thresh_max=255):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,1]
    
    # Creating image masked in S channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh_min) & (s_channel <= thresh_max)] = 1

    return s_binary

def lab_thresh(img, thresh=(0,255)):
    # Normalises and thresholds to the B channel
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab_b = lab[:,:,2]
    # Don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    #  Apply a threshold
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    return binary_output

#####################
## Window searching

def window_search(binary_warped):

    # Take a histogram of the bottom half of the image
    bottom_half_y = binary_warped.shape[0]/2
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    #print("midpoint",midpoint)
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 30
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one 
    # inds 완쪽과 오른쪽 차선을 구성하는 픽셀의 인덱스를 저장하는 리스트
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
  
    '''
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2) 
    right_fit = np.polyfit(righty, rightx, 2) 
    '''
    right_fit_new = np.array([0, 0, 0])
    left_fit_new = np.array([0, 0, 0])
    line.detected = True
    # 각각에 대해 2차 다항식을 적합합니다. 계수 생성
    if len(leftx) > 800 and len(lefty) > 800 and len(rightx) > 800 and len(righty) > 800:
        #print("left충분")
        #print("search_leftx",len(leftx))
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2) 
    elif len(leftx) < 800 and len(lefty) < 800 and len(rightx) > 800 and len(righty) > 800:
        #print("leftx불충분")
        #print("search_leftx = ",len(leftx))
        right_fit = np.polyfit(righty, rightx, 2)
        left_fit = np.array([right_fit[0], right_fit[1], right_fit[2] - 150])   
    elif len(leftx) > 800 and len(lefty) > 800 and len(rightx) < 800 and len(righty) < 800:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.array([left_fit[0], left_fit[1], left_fit[2] + 150])
    else:
        print("No lane detected Return!!!")
        left_fit = np.array([0, 0, 0])
        right_fit = np.array([0, 0, 0])
        line.detected = False
        return left_lane_inds, right_lane_inds, out_img,left_fit,right_fit

   
    #print("search_leftx_fit_new = ",left_fit_new)
    #print("search_right_fit_new = ",right_fit_new)
    #print("search_leftx_fit = ",left_fit)
    #print("search_right_fit = ",right_fit)    
    '''
    # 새로운 적합이 유효하면 전역 변수를 업데이트합니다.
    if left_fit_new is not None:
        left_fit = left_fit_new
    if right_fit_new is not None:
        right_fit = right_fit_new
    '''

    # Generate x and y values for plotting, x = f(y)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    # Generate black image and colour lane lines
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 255, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 255, 1]
        
    # Draw polyline on image x,y좌표
    right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
    left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)


    cv2.polylines(out_img, [right], False, (255,1,0), thickness=10)
    cv2.polylines(out_img, [left], False, (255,1,0), thickness=10)
    cv2.putText(out_img, 'window_search', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,1,0), 2, cv2.LINE_AA)
    # Get the x coordinates when y is at its maximum
    left_x_at_max_y = left_fitx[-1]
    right_x_at_max_y = right_fitx[-1]
    # Draw a dot at the x coordinates when y is at its maximum
    cv2.circle(out_img, (int(left_x_at_max_y), int(ploty[-1])), 10, (255,0,0), -1)
    cv2.circle(out_img, (int(right_x_at_max_y), int(ploty[-1])), 10, (0,255,0), -1)
    '''
    if right_x_at_max_y < 160:
        print("window_warining!!!")
        cv2.imshow('out_img', out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()   
    '''
    #print("Left lane x coordinate at max y: ", left_x_at_max_y)
    #print("Right lane x coordinate at max y: ", right_x_at_max_y)
    
    #cv2.imshow('out_img_window',out_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    


    return left_lane_inds, right_lane_inds, out_img,left_fit,right_fit
    
#####################
## Margin search
def margin_search(binary_warped):
    global left_fit, right_fit
    # Performs window search on subsequent frame, given previous frame.
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 10

    left_lane_inds = ((nonzerox > (left_line.current_fit[0]*(nonzeroy**2) + left_line.current_fit[1]*nonzeroy + left_line.current_fit[2] - margin)) & (nonzerox < (left_line.current_fit[0]*(nonzeroy**2) + left_line.current_fit[1]*nonzeroy + left_line.current_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_line.current_fit[0]*(nonzeroy**2) + right_line.current_fit[1]*nonzeroy + right_line.current_fit[2] - margin)) & (nonzerox < (right_line.current_fit[0]*(nonzeroy**2) + right_line.current_fit[1]*nonzeroy + right_line.current_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]


    # 각각에 대해 2차 다항식을 적합합니다. 계수 생성
    if len(leftx) > 0 and len(lefty) > 0:
        left_fit_new = np.polyfit(lefty, leftx, 2)
        #print("left_fit",left_fit_new)
    else:
        left_fit_new = left_fit  # 새로운 값이 유효하지 않으면 이전 값을 유지합니다.

    if len(rightx) > 0 and len(righty) > 0:
        right_fit_new = np.polyfit(righty, rightx, 2)
        #print("right_fit",right_fit_new)
    else:
        right_fit_new = right_fit  # 새로운 값이 유효하지 않으면 이전 값을 유지합니다.
      

    # 새로운 적합이 유효하면 전역 변수를 업데이트합니다.
    if left_fit_new is not None:
        left_fit = left_fit_new
    if right_fit_new is not None:
        right_fit = right_fit_new


    max_y = binary_warped.shape[0] - 1
    left_x_at_max_y = left_fit[0]*max_y**2 + left_fit[1]*max_y + left_fit[2]
    right_x_at_max_y = right_fit[0]*max_y**2 + right_fit[1]*max_y + right_fit[2]
    #print("left_x_at_max_y",left_x_at_max_y)
    #print("right_x_at_max_y",right_x_at_max_y)
    if right_x_at_max_y > 160 and left_x_at_max_y < 160:
        pass
    elif right_x_at_max_y < 160 and left_x_at_max_y < 160:
        print("Right Lane Warning!!!")
        right_fit = np.array([left_fit[0], left_fit[1], left_fit[2] + 150])
    elif right_x_at_max_y > 160 and left_x_at_max_y > 160:
        print("Left Lane Warning!!!")
        left_fit = np.array([right_fit[0], right_fit[1], right_fit[2] - 150])

   
    # Generate a blank image to draw on
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255


    # Generate x and y values for plotting, x = f(y)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    window_img = np.zeros_like(out_img)


    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.intc([left_line_pts]), (0,255,1))
    cv2.fillPoly(window_img, np.intc([right_line_pts]), (0,255,1))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 1]
        
    # Draw polyline on image
    right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
    left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
    cv2.polylines(out_img, [right], False, (1,1,255), thickness=5)
    cv2.polylines(out_img, [left], False, (1,1,255), thickness=5)
    

    cv2.putText(out_img, 'margin_search', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,1,0), 2, cv2.LINE_AA)

    return left_lane_inds, right_lane_inds, out_img,left_fit,right_fit
    
    #return left_lane_inds, right_lane_inds, out_img
   
    
#### lane finding        
def find_lanes(img):
    if left_line.detected and right_line.detected:  # Perform margin search if exists prior success.
        # Margin Search
        left_lane_inds, right_lane_inds,out_img,left_fit,right_fit = margin_search(img)
        # Update the lane detections
        #print("margin_search_update")
        validate_lane_update(img, left_lane_inds, right_lane_inds)
        
    else:  # Perform a full window search if no prior successful detections.
        # Window Search
        left_lane_inds, right_lane_inds,out_img,left_fit,right_fit= window_search(img)
        # Update the lane detections
        #print("window_search_update")
        validate_lane_update(img, left_lane_inds, right_lane_inds)
        
    return out_img,left_fit,right_fit


#### calculate the center of lanes
  
def calculate_lane_center(left_lane_inds, right_lane_inds):
    # Calculate the x position of the center of the left lane
    left_lane_center = np.mean(left_lane_inds)

    # Calculate the x position of the center of the right lane
    right_lane_center = np.mean(right_lane_inds)

    # Calculate the x position of the center of the lane
    lane_center = (left_lane_center + right_lane_center) / 2

    return lane_center
    
def write_stats(img):
    font = cv2.FONT_HERSHEY_PLAIN
    size = 3
    weight = 2
    color = (255,255,255)
    
    #radius_of_curvature = (right_line.radius_of_curvature + right_line.radius_of_curvature)/2
    #cv2.putText(img,'Lane Curvature Radius: '+ '{0:.2f}'.format(radius_of_curvature)+'m',(30,60), font, size, color, weight)
        
##### draw Lame
     
def draw_lane(undist, img, Minv,W_left_fit,W_right_fit):
    lane_center = 0
    lane_center_y = 0
    # Generate x and y values for plotting
    ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.stack((warp_zero, warp_zero, warp_zero), axis=-1)
    if line.detected == False:
        print("No lane detected")
        lane_center = -1
        lane_center_y = -1
        cv2.putText(undist, 'no lane detected', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (1,255,0), 2, cv2.LINE_AA)
        return undist, lane_center, lane_center_y

    if left_line.detected and right_line.detected:
        left_fit = left_line.best_fit
        right_fit = right_line.best_fit
    else:
        left_fit = W_left_fit
        right_fit = W_right_fit


    #print("###draw_lane_left_fit=",left_fit)
    #print("###draw_lane_right_fit=",right_fit)
    
    if left_fit is not None and right_fit is not None:
               
        # Recast the x and y points into usable format for cv2.fillPoly()
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
       
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)        
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        
        pts = np.hstack((pts_left, pts_right))

      
        # Calculate the center of the lane
        lane_center = calculate_lane_center(left_fitx, right_fitx)
        # Calculate the y position of the center of the lane
        lane_center_y = color_warp.shape[0] // 2   
        #print("lane_center",lane_center)


        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 224, 0))
        cv2.polylines(color_warp, [right], False, (0,0,255), thickness=20)
        cv2.polylines(color_warp, [left], False, (0,0,255), thickness=20) 

        # Draw a circle at the center of the lane
        # Draw a circle at the center of the lane
        #cv2.circle(color_warp, (int(lane_center), lane_center_y), 20, (0, 0, 255), -1) 
        for y in range(0, color_warp.shape[0], 20):
            center_x = (left_fit[0]*y**2 + left_fit[1]*y + left_fit[2] + right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]) // 2
            if(y==80):
                lane_center = center_x
                lane_center_y = y
            cv2.circle(color_warp, (int(center_x), y), 5, (255, 0, 0), -1)
            
        cv2.circle(color_warp, (int(lane_center), lane_center_y), 20, (0, 0, 255), -1) 
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
        
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.6, 0)
        

        write_stats(result)
        return result, lane_center, lane_center_y
    return undist
         
            
def process_img(img):
 
    # Perspective transform image
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    warped, M, Minv = pers_transform(img)
    
    # Colour thresholding in S channel
    s_bin = hls_thresh(warped,200,255)


    # Colour thresholding in B channel of LAB
    b_bin = lab_thresh(warped, thresh = (185, 255))


    # Combining both thresholds
    combined = np.zeros_like(s_bin)
    combined[(s_bin==1) | (b_bin == 1)] = 1
    scaled_combine_image = np.uint8(255*combined)


    # Find Lanes
    output_img,left_fit,right_fit = find_lanes(combined)

    # Draw lanes on image
    lane_img,lane_center,lane_center_y = draw_lane(img, combined, Minv,left_fit,right_fit);  

    return lane_img, combined, output_img, warped,lane_center,lane_center_y



'''
left_line = Line()
right_line = Line()

line = Line()


# Create VideoWriter object to save the output video
#out = cv2.VideoWriter('output_video2.avi', fourcc, fps, (1707, 720))

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Process the frame
    result, combined, output_img, warped,lane_center,lane_center_y = process_img(frame)
    print("lane_center",lane_center)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    #print("result",result.shape)

    
    # Display the processed frame
    cv2.imshow('Processed Frame', result)
    cv2.imshow('Combined', combined*255)
    cv2.imshow('Output', output_img)
    cv2.imshow('Warped', warped)


    # Write the processed frame to the output video
    #out.write(result)
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
#cap.release()
#out.release()

# Close all windows
cv2.destroyAllWindows()
'''

class MotorCtlPublisher(Node):

    def __init__(self):
        super().__init__('motorctl_publisher')
        #self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.publisher_ = self.create_publisher(MsgCtl, 'MotorCtl', 10)     # CHANGE
        self.angle_last = 0.0
        self.steering_value = 0.0001
        self.steering_dgain_value = 0.0
        self.steering_bias_value = 0.0

        self.speed_gain_value = 0.18
        self.steering_gain_value = 0.03

    def motor_drive(self, goalx, goaly):
	
        center = 320./2.
        goalX = (float(goalx)/100.) - (center/100.)
        _goalY1 = float(goaly)/100.
        goalY = -((0.5 - _goalY1) / 2.0)
        speed_value = self.speed_gain_value

        angle = np.arctan2(goalX, goalY)

        pid = angle * self.steering_gain_value + (angle - self.angle_last) * self.steering_dgain_value
        self.angle_last = angle

        steering_value = pid + self.steering_bias_value 
        #print("goalX {}, goalY {} angle {} steering_value {}".format(goalX, goalY, angle, steering_value))    
        print("angle {} steering_value {}".format(angle, steering_value))  
        left_motor_value = max(min(speed_value + steering_value, 1.0), 0.0)
        right_motor_value = max(min(speed_value - steering_value, 1.0), 0.0)
        print("left_motor_value {}, right_motor_value {} ".format(left_motor_value, right_motor_value)) 


        #msg = String()
        msg = MsgCtl()                                           # CHANGE
        #msg.data = 'Hello World: %d' % self.i
        #msg.num = self.i  
        msg.speed_cmd = speed_value;
        msg.heading_cmd = steering_value;
        msg.dir_cmd = 0;
        msg.stop_cmd = 0;
       
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: speedCmd: "%0.2f"' % msg.speed_cmd)
        self.get_logger().info('Publishing: heading_cmd: "%0.2f"' % msg.heading_cmd)
        self.get_logger().info('Publishing: dir_cmd: "%d"' % msg.dir_cmd)
        self.get_logger().info('Publishing: stop_cmd: "%d"' % msg.stop_cmd)

gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

packet_path = "/home/nvidia/cv_selfdriving_ws/src/cv_package/"
#input_path = gst_str
input_path = packet_path + "video/track.avi"

def main(args=None):
  
    rclpy.init()
    node = rclpy.create_node("cam_viewer")

    global bridge
    bridge = CvBridge()

    #left_line = Line()
    #right_line = Line()

    #line = Line()
		
    #cap = cv2.VideoCapture(1)
    
    print(input_path)
    
    cap = cv2.VideoCapture(input_path)
    if not (cap.isOpened()):
        print("Could not open video device")
    # To set the resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("width {}  height {} ".format(width, height))
    print('motorctl_publisher')
    
    motorctl_publisher = MotorCtlPublisher()

    while rclpy.ok():
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        result, combined, output_img, warped,lane_center,lane_center_y = process_img(frame)
        print("lane_center",lane_center)
        #result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        #print("result",result.shape)
        '''
        # Display the processed frame
        cv2.imshow('Processed Frame', result)
        cv2.imshow('Combined', combined*255)
        cv2.imshow('Output', output_img)
        cv2.imshow('Warped', warped)
        '''
        cv2.imshow('preview',result)
        
        motorctl_publisher.motor_drive(lane_center, lane_center_y)
        #print('view video frame')
        # Waits for a user input to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    motorctl_publisher.destroy_node()

    node.destroy_node()
    rclpy.shutdown()
  
if __name__ == '__main__':
  main()
