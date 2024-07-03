import numpy as np
import math
import cv2
import matplotlib.pyplot as plt


### Camera coordinate to vehicle coordinate(ISO)
def RotX(Pitch):
      """
      Returns the rotation matrix around the X-axis (Roll) given the pitch angle in degrees.

      Parameters:
      - Pitch (float): The pitch angle in degrees.

      Returns:
      - rot_matrix (list of lists): The rotation matrix around the X-axis.

      Example:
      >>> RotX(45)
      [[1, 0, 0], [0, 0.7071067811865476, -0.7071067811865475], [0, 0.7071067811865475, 0.7071067811865476]]
      """      
      Pitch = np.deg2rad(Pitch)
      return [[1, 0, 0], [0, math.cos(Pitch), -math.sin(Pitch)], [0, math.sin(Pitch), math.cos(Pitch)]]

def RotY(Yaw):
      """
      Calculate the rotation matrix around the Y-axis.

      Parameters:
      - Yaw: The yaw angle in degrees.

      Returns:
      - A 3x3 rotation matrix representing the rotation around the Y-axis.
      """
      Yaw = np.deg2rad(Yaw)
      return [[math.cos(Yaw), 0, math.sin(Yaw)], [0, 1, 0], [-math.sin(Yaw), 0, math.cos(Yaw)]]

def RotZ(Roll):
      """
      Rotate a point around the Z-axis (yaw) by the given angle in degrees.

      Parameters:
      - Roll (float): The angle of rotation in degrees.

      Returns:
      - list: A 3x3 rotation matrix representing the rotation around the Z-axis.
      """      
      Roll = np.deg2rad(Roll)
      return [[math.cos(Roll), -math.sin(Roll), 0], [math.sin(Roll), math.cos(Roll), 0], [0, 0, 1]]


### Camera parameter setting
ImageSize = (375, 1242)
FocalLength = (721.5377, 721.5377)
PrinciplePoint = (609.5593, 172.854)
IntrinsicMatrix = ((FocalLength[0], 0, 0), (0, FocalLength[1], 0), (PrinciplePoint[0], PrinciplePoint[1], 1))
Height = 1.65
Pitch = 0
Yaw = 0
Roll = 0

### Bird's eye view setting
DistAhead = 40
SpaceToOneSide = 7
BottomOffset = 4

OutView = (BottomOffset, DistAhead, -SpaceToOneSide, SpaceToOneSide)
OutImageSize = [math.nan, 500]

WorldHW = (abs(OutView[1]-OutView[0]), abs(OutView[3]-OutView[2]))

Scale = (OutImageSize[1]-1)/WorldHW[1]
ScaleXY = (Scale, Scale)

OutDimFrac = Scale*WorldHW[0]
OutDim = round(OutDimFrac)+1
OutImageSize[0] = OutDim

### Homography Matrix Compute

#Translation Vector
Rotation = np.matmul(np.matmul(RotZ(-Yaw),RotX(90-Pitch)),RotZ(Roll))
'''
원래 카메라 좌표계에서 카메라의 광학 중심이 지상으로 내려가고 센서 위치가 설명되는 변환을 계산합니다.
이렇게 하면 차량 좌표계와 정렬된 차량 좌표계에서 카메라 좌표계로 좌표 변환이 수행됩니다.
RotX(90-Pitch) 을 통해서 카메라의 광학 중심이 지상으로(아래쪽)내려가게 됩니다. ==> 차량 좌표계와 카메라 좌표계의 정렬을 위해서
'''
TranslationinWorldUnits = (0, 0, Height)
'''
센서의 위치 정보를 넣어주는 부분입니다. 데이터 셋에 카메라의 위치는 X=0 Y=0 Z(Height)=1.65로 설정되어 있습니다.(차량 좌표계 기준)
만약에 카메라의 위치(차량좌표계)가 X= sx, Y=sy, Z=sz로 설정되어 있다면, TranslationinWorldUnits = (sy, sx, sz)로 설정하면 됩니다.
sx, sy의 위치가 바뀐 것은 위 rotation 을 통해서 카메라가 아래를 볼때 X와 Y가 바뀌기 때문입니다.
'''
Translation = [np.matmul(TranslationinWorldUnits, Rotation)]

#Rotation Matrix
RotationMatrix = np.matmul(RotY(180), np.matmul(RotZ(-90), np.matmul(RotZ(-Yaw), np.matmul(RotX(90-Pitch), RotZ(Roll)))))
'''
RotX(90-Pitch) : Pitch에 대한 회전
   z^                             zX------>x
    '            ---------->       '
    '           RotX(90-Pitch)     '
   yX------>x                      'y

                                               ^
------->    y<-----X z  ----------->           'x          
RotZ(-90)          '     RotY(180)             '        
                   '                           '        
                   'x                   y<-----0z             


'''
#Camera Matrix
CameraMatrix = np.matmul(np.r_[RotationMatrix, Translation], IntrinsicMatrix)
'''
This transform is meant for converting 3D vehicle coordinate sytem locations to image coordinates.
RoationMatrix * Translation = Compute extrinsics based on camera setup
CameraMatrix = Construct a camera matrix
'''
CameraMatrix2D = np.r_[[CameraMatrix[0]], [CameraMatrix[1]], [CameraMatrix[3]]]

#Compute Vehicle-to-Image Projective Transform
VehicleHomo = np.linalg.inv(CameraMatrix2D)

AdjTransform = ((0, -1, 0), (-1, 0, 0), (0, 0, 1))
BevTransform = np.matmul(VehicleHomo, AdjTransform)

DyDxVehicle = (OutView[3], OutView[1])
tXY = [a*b for a,b in zip(ScaleXY, DyDxVehicle)]

#test = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
ViewMatrix = ((ScaleXY[0], 0, 0), (0, ScaleXY[1], 0), (tXY[0]+1, tXY[1]+1, 1))

T_Bev = np.matmul(BevTransform, ViewMatrix)
T_Bev = np.transpose(T_Bev)

### Main
src = cv2.imread("./test_img/0000000056.png", cv2.IMREAD_COLOR)

BirdEyeView = cv2.warpPerspective(src, T_Bev, (OutImageSize[1], OutImageSize[0]))
plt.figure(figsize = (20,20))
plt.imshow(BirdEyeView)
#Image to Vehicle
'''
T_Bev는 BEV를 생성할 때 사용됩니다. (원래의 이미지 공간에서 BEV 공간으로 좌표 변환)
toOriginalImage = np.linalg.inv(np.transpose(T_Bev)) 이 부분에서는 T_BEV의 역행렬을 계산하여 원래의 이미지 공간으로 좌표를
역변환하는데 사용 됩니다. VehicleHomo는 차량 좌표계에서 이미지 좌표계로의 프로젝티브 변환을 나타내는 행렬입니다.
Trans = np.matmul(toOriginalImage, VehicleHomo) 이 부분에서는 행렬 곱셈하여 최종 변환 행렬을 얻게 되는데
이 행렬은 BEV 공간에서 원래의 이미지 공간으로 좌표를 역변환한 후, 이미지 공간에서 차량 공간으로 좌표를 변환합니다.
'''
toOriginalImage = np.linalg.inv(np.transpose(T_Bev))
Trans = np.matmul(toOriginalImage, VehicleHomo)
ImagePoint = [[120, 400]]

#=============
BEV_point = [[120,400]]
BEV_point = np.r_[BEV_point[0], np.shape(BEV_point)[0]]
ori_img_point=np.matmul(BEV_point,toOriginalImage)
ori_img_point_t = ori_img_point[0:2]/ori_img_point[2]
print("ori_img_point",ori_img_point_t) #original image point 
dist =np.matmul(ori_img_point,VehicleHomo)
dist[0:2] = dist[0:2]/dist[2]
print("dist",dist)

ori_point = [[517.84861558, 214.18404489]]
ori_img_point = np.r_[ori_point[0], np.shape(ori_point)[0]]
result = np.matmul(ori_img_point, VehicleHomo)
result[0:2] = result[0:2] / result[2]
print("Distance in vehicle coordinates:", result)
#==============
UI = ImagePoint
UI = np.r_[ImagePoint[0], np.shape(ImagePoint)[0]]
XI = np.matmul(UI, Trans)

XI[0:2] = XI[0:2]/XI[2]
XAhead = XI[0]
YAhead = XI[1]
#===============
print("XAhead",XAhead)
print("YAhead",YAhead)
#==============
tpimg = tuple(ImagePoint[0])
#annotatedBEV2 = cv2.drawMarker(BirdEyeView, ImagePoint[0], (0,0,255))
annotatedBEV2 = cv2.drawMarker(BirdEyeView, tpimg, (0,0,255))
cv2.putText(annotatedBEV2, str(round(XAhead, 2))+" meters", tpimg, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))
cv2.putText(annotatedBEV2, str(round(YAhead, 2))+" meters", (ImagePoint[0][0], ImagePoint[0][1] + 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))
cv2.namedWindow("BEV", cv2.WINDOW_NORMAL)
cv2.resizeWindow("BEV", width=960, height=960)
cv2.imshow("Original", src)
cv2.imshow("BEV", annotatedBEV2)
cv2.waitKey(0)
cv2.destroyAllWindows()
