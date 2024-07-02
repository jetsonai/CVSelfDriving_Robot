
## Track Bird eyes View

python3 getWarpPoint_Image.py ./data/goStrait.jpg

![get_warp_img_points](https://github.com/jetsonai/CVSelfDriving_Robot/assets/96120477/a21b13ae-f873-43ba-b045-f471ca371ff7)

python3 getWarpPoint_Camera.py

###Modify src 160 line  

python3 FindLane_Image.py ./data/goStrait.jpg

![findtrack](https://github.com/jetsonai/CVSelfDriving_Robot/assets/96120477/0523bfb8-42fc-4322-b0b9-2f42a15b9fe4)

FindLane_TrackVideo.py ./data/track.avi

python3 FindLane_Image.py ./data/CurveLeft.jpg

python3 FindLane_Image.py ./data/CurveRight.jpg

==============================================

## get Distance from Camera Homography

python3 getWarpPoint_Camera.py

![get_warp_cam_points](https://github.com/jetsonai/CVSelfDriving_Robot/assets/96120477/5ba16f1a-2b2c-454e-9a91-d95060c83a09)

##<Test Distance from Camera Homography>

Click mouse point on the view to check distance!!

python3 BEV_Dist_Camera.py 

![BEV_cam_Homo](https://github.com/jetsonai/CVSelfDriving_Robot/assets/96120477/75cbb550-b001-4133-b583-5f484e3039b7)

==============================================

## Yolo Sort and Distance

python3 Yolo_Sort_Camera.py

![sort](https://github.com/jetsonai/CVSelfDriving_Robot/assets/96120477/5651c7d8-004e-4785-ada0-514ce2425ae6)

python3 Yolo_Sort_Dist_Cam.py

![sort_dist](https://github.com/jetsonai/CVSelfDriving_Robot/assets/96120477/d7126cb2-192d-4045-b7ee-f2a5a1b761f5)


python3 Yolo_TTC_Cam.py


