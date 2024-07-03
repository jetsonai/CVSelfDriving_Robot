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


## lane trace


## Yolo Sort Dist


