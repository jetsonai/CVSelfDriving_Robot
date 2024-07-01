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


-----------------------

 8 YOLO

 ros2 run ros_yolosort ros_yolo_node_dist

 colcon build --packages-select ros_yolosort

 
