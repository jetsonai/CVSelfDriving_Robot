1. package 설치

cd

gedit .bashrc

133 line 의 cvselfdriving_ws 라인 주석 지우고 

134 line 에 ROS_DOMAIN_ID 를 본인 번호로 바꾸어 주세요

저장

colcon build

만약 하나의 패키지만 빌드할때 옵션

예) cv_package 패키지이니 경우

colcon build --packages-select cv_package


2. lane 검출 및 모터값 확인
   
python3 FindLane_Image.py ./data/boardpic.jpg

3. 영상 복사
   
cd Day3/data/

scp -r nvidia@192.168.100.115:/home/nvidia/Day3/data/track.avi ./

4. 페키지 실핼

   ros2 run cv_package lane_trace_cam_node
   
  구독자
6. ros2 run ros_rsaem_ctl rsaembot_motor

7 Homography

함수 정의

def Dist(src,dst):

    Homo,mask = cv2.findHomography(src,dst,cv2.RANSAC)
    
    return Homo

    ------------

적용

src = np.float32([[94,90],[114, 86], [138, 82], [163, 79]])

dst = np.float32([[22.5,5],[22.5,2.5],[22.5,0],[22.5,-2.5]])

Homo = Dist(src,dst)

V_Center = np.dot(Homo, img_center)

V_Center_X = V_Center[0]/V_Center[2]

V_Center_Y = V_Center[1]/V_Center[2]

#print("V_Center_X",V_Center_X)

V_Center_X_text = "X: {:.2f}".format(V_Center_X)

V_Center_Y_text = "Y: {:.2f}".format(V_Center_Y)

lane_center = int(lane_center)

lane_center_y = int(lane_center_y)   

cv2.putText(result, V_Center_X_text, (lane_center, lane_center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

cv2.putText(result, V_Center_Y_text, (lane_center, lane_center_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
 

