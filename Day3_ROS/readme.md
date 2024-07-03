## Edit bashrc for ROS workspace setting 

cd

gedit .bashrc

133 line 의 cvselfdriving_ws 라인 주석 지우고

134 line 에 ROS_DOMAIN_ID 를 본인 번호로 바꾸어 주세요

저장

## ROS workspace setting 

cd

mkdir -p cv_selfdriving_ws/src

copy these 6 packages to cv_selfdriving_ws/src

colcon build

만약 하나의 패키지만 빌드할때 옵션

예) cv_package 패키지이니 경우

colcon build --packages-select cv_package

colcon build --packages-select ros_yolosort

## lane trace

페키지 실

ros2 run cv_package lane_trace_cam_node

구독자 

ros2 run ros_rsaem_ctl rsaembot_motor

## Yolo Sort

ros2 run ros_yolosort ros_yolo_node

![yolo](https://github.com/jetsonai/CVSelfDriving_Robot/assets/96120477/2ce93868-039c-4d0f-8b57-6d9e53ac77e5)

## Yolo Dist

ros2 run ros_yolosort ros_yolo_dist

![yolo_sort](https://github.com/jetsonai/CVSelfDriving_Robot/assets/96120477/38d397e1-8abc-484e-af2a-29569e315426)

## Yolo Sort Dist

ros2 run ros_yolosort ros_yolo_sort_dist

![yolo_sort_dist](https://github.com/jetsonai/CVSelfDriving_Robot/assets/96120477/7a1f7f11-ab43-483e-92c5-5adf46eb1129)

## Yolo TTC

ros2 run ros_yolosort ros_yolo_TTC

![yolo_ttc](https://github.com/jetsonai/CVSelfDriving_Robot/assets/96120477/7c8a4c35-a680-49e8-b2ee-662f821405b6)

## Yolo Driving (with TTC)

ros2 run ros_yolosort ros_yolo_driving



